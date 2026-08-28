// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <cassert>
#include <format>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace rocjitsu::plugins::race_detector {

namespace {

void warn_cluster_peer_writes_ignored_once() {
  static std::once_flag warned;
  std::call_once(warned, [] {
    util::Logger::warn(
        "race detector does not model cluster LDS multicast peer writes; peer writes are ignored");
  });
}

int lgkmcnt_no_wait_value(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return 15;
  default:
    return 63;
  }
}

bool models_cdna_ordered_counter_backpressure(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

MemoryCompletionOrder completion_order_for(const Instruction &inst, rj_code_arch_t arch) {
  if (!inst.is_memory_op())
    return MemoryCompletionOrder::UNORDERED;

  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.starts_with("ds_")) {
    // GDS shares LGKMCNT with LDS but is a distinct completion-order class.
    return inst.disassemble().find(" gds") == std::string::npos ? MemoryCompletionOrder::LDS
                                                                : MemoryCompletionOrder::UNORDERED;
  }

  // Generic FLAT operations can return out of order with each other and other
  // VMEM operations, even though they contribute to the same wait counters.
  if (mnemonic.starts_with("flat_"))
    return MemoryCompletionOrder::UNORDERED;

  // CDNA has no separate VSCNT. All non-FLAT VMEM events share one in-order
  // VMCNT stream. On targets with split load/store counters, keep only the
  // ordinary load class here.
  if (models_cdna_ordered_counter_backpressure(arch) &&
      (mnemonic.starts_with("global_") || mnemonic.starts_with("scratch_") ||
       mnemonic.starts_with("buffer_") || mnemonic.starts_with("tbuffer_") ||
       mnemonic.starts_with("image_"))) {
    return MemoryCompletionOrder::VMEM;
  }
  if (mnemonic.starts_with("global_load") || mnemonic.starts_with("scratch_load") ||
      mnemonic.starts_with("buffer_load") || mnemonic.starts_with("tbuffer_load") ||
      mnemonic.starts_with("image_load"))
    return MemoryCompletionOrder::VMEM;

  return MemoryCompletionOrder::UNORDERED;
}

} // namespace

MemoryCompletionOrder memoryCompletionOrderForRaceDetector(const Instruction &inst,
                                                           rj_code_arch_t arch) {
  return completion_order_for(inst, arch);
}

std::optional<PendingWaitCount> waitCountForRaceDetector(std::string_view mnemonic, int vmcnt,
                                                         int lgkmcnt) {
  PendingWaitCount waitCount;
  if (mnemonic == "s_waitcnt") {
    waitCount.vmcnt = vmcnt;
    waitCount.lgkmcnt = lgkmcnt;
  } else if (mnemonic == "s_waitcnt_vmcnt") {
    waitCount.vmcnt = vmcnt;
  } else if (mnemonic == "s_waitcnt_lgkmcnt") {
    waitCount.lgkmcnt = lgkmcnt;
  } else {
    return std::nullopt;
  }
  return waitCount;
}

// Declared in plugin.h (used by formatTrace tests in execution_plugin_test.cpp).
MarkedPc findConflict(const RaceViolation &v, RaceDetector &detector) {
  EventId event_id = v.conflictingEvent;
  if (!detector.events().contains(event_id))
    throw std::out_of_range("race violation references an unavailable conflicting event");
  return {detector.events().pc(event_id), detector.events().waveId(event_id).value, -1};
}

// Format a race trace showing the instruction stream between the memory
// operation that wrote a register (conflict) and the instruction that read
// it before the write completed (read). The trace is a rolling window of
// recent PCs; disasm maps every PC seen in the kernel to its disassembly.
//
// Output uses ==> markers for the two involved instructions and annotates
// each with wave/lane. Instructions before the first marker are trimmed.
// When the conflict fell outside the trace window, its disassembly is still
// shown (from the disasm map) with a "(before trace window)" note.
std::string formatTrace(const RingBuffer<uint64_t, 256> &trace,
                        const std::unordered_map<uint64_t, std::string> &disasm,
                        std::optional<MarkedPc> conflict, MarkedPc read) {
  auto isMarked = [&](uint64_t pc) { return pc == read.pc || (conflict && pc == conflict->pc); };

  auto lookup = [&](uint64_t pc) -> const std::string & {
    static const std::string empty;
    auto it = disasm.find(pc);
    return it != disasm.end() ? it->second : empty;
  };

  size_t n = trace.size();
  size_t first = n;
  bool conflict_found = false;
  for (size_t i = 0; i < n; ++i) {
    if (isMarked(trace[i])) {
      first = i;
      break;
    }
  }
  if (conflict) {
    for (size_t i = 0; i < n; ++i)
      if (trace[i] == conflict->pc) {
        conflict_found = true;
        break;
      }
  }

  std::ostringstream oss;

  if (conflict && !conflict_found) {
    oss << "  ==>  0x" << std::hex << conflict->pc << std::dec << "  ";
    auto &d = lookup(conflict->pc);
    if (!d.empty())
      oss << d << "  ";
    oss << "(before trace window)";
    oss << "  ; <-- wave " << conflict->wave;
    if (conflict->lane >= 0)
      oss << " lane " << conflict->lane;
    oss << "\n       ... " << first << " instructions not recorded ...\n";
  }

  size_t last = n;
  for (size_t i = first; i < n; ++i) {
    if (trace[i] == read.pc) {
      last = i + 1;
      break;
    }
  }

  constexpr size_t MAX_PRINT_SIZE = 32;
  size_t span = (last > first) ? last - first : 0;

  auto emit = [&](size_t i) {
    uint64_t pc = trace[i];
    bool is_conflict = conflict && pc == conflict->pc;
    bool is_read = pc == read.pc;
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ");
    oss << "0x" << std::hex << pc << std::dec << "  " << lookup(pc);
    if (is_conflict) {
      oss << "  ; <-- wave " << conflict->wave;
      if (conflict->lane >= 0)
        oss << " lane " << conflict->lane;
    }
    if (is_read) {
      oss << "  ; <-- wave " << read.wave;
      if (read.lane >= 0)
        oss << " lane " << read.lane;
    }
    oss << "\n";
  };

  if (span <= MAX_PRINT_SIZE) {
    for (size_t i = first; i < last; ++i)
      emit(i);
  } else {
    size_t half = MAX_PRINT_SIZE / 2;
    for (size_t i = first; i < first + half; ++i)
      emit(i);
    oss << "       ... " << (span - MAX_PRINT_SIZE) << " instructions elided ...\n";
    for (size_t i = last - half; i < last; ++i)
      emit(i);
  }
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char * /*config_json*/) : ExecutionPlugin("race") {}

RaceDetectorPlugin::~RaceDetectorPlugin() { sink().write(getSummary()); }

std::string RaceDetectorPlugin::getSummary() const {
  const char *banner = "\n========================================\n"
                       " ROCJITSU RACE DETECTION SUMMARY\n"
                       "========================================\n";
  if (!observed_races_.empty()) {
    return std::string(banner) + "  " + std::to_string(observed_races_.size()) +
           " race(s) detected\n"
           "========================================\n";
  }
  return std::string(banner) + "  No races detected.\n"
                               "========================================\n";
}

void RaceDetectorPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  std::lock_guard<std::mutex> lock(report_mutex_);
  KernelNames kernel_names{info.kernelNameOrUnknown(), info.kernelSymbolOrUnknown()};
  dispatch_kernel_names_[info.dispatch_id] = kernel_names;
  sink().write(std::format("[rocjitsu] Kernel dispatch: \"{}\" symbol=\"{}\"\n", kernel_names.name,
                           kernel_names.symbol));
}

void RaceDetectorPlugin::onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                                     uint32_t physical_vgpr_count,
                                                     uint32_t sgpr_count,
                                                     std::span<amdgpu::Wavefront *> wavefronts) {
  uint32_t num_waves = static_cast<uint32_t>(wavefronts.size());
  WorkgroupKey key{dispatch_id, wg_id};

  std::vector<amdgpu::Wavefront *> wf_ptrs(wavefronts.begin(), wavefronts.end());
  auto handler = [this, wf_ptrs, dispatch_id](RaceViolation v) {
    assert(v.wave >= 0 && static_cast<size_t>(v.wave) < wf_ptrs.size() &&
           "wave index out of range");
    amdgpu::Wavefront *wf = wf_ptrs[v.wave];
    uint64_t pc = wf->pc;

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      if (observed_races_.count({dispatch_id, pc}))
        return;
    }

    auto *ws = get_state(wf);
    assert(ws && ws->race_state && "no wavefront state for race");
    auto *detector = ws->race_state->getDetector();
    MarkedPc conflict = findConflict(v, *detector);

    std::ostringstream oss;
    if (v.space == RaceViolation::Space::VGPR)
      oss << "Race on VGPR v" << v.index;
    else if (v.space == RaceViolation::Space::SGPR)
      oss << "Race on SGPR s" << v.index;
    else
      oss << "Race on LDS byte " << v.index;
    oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
        << "), wave " << v.wave;
    if (v.space != RaceViolation::Space::SGPR)
      oss << ", lane " << v.lane;
    oss << "]\n";

    MarkedPc access_mark{pc, v.wave, v.lane};
    oss << formatTrace(ws->trace, ws->disasm->to_map(), conflict, access_mark);

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      bool is_new = !observed_races_.count({dispatch_id, pc}) &&
                    !observed_races_.count({dispatch_id, conflict.pc});
      observed_races_.emplace(dispatch_id, pc);
      if (is_new) {
        observed_races_.emplace(dispatch_id, conflict.pc);
        const char *space = v.space == RaceViolation::Space::VGPR   ? "VGPR"
                            : v.space == RaceViolation::Space::SGPR ? "SGPR"
                                                                    : "LDS";
        auto kernel_name_iter = dispatch_kernel_names_.find(dispatch_id);
        const KernelNames kernel_names =
            kernel_name_iter == dispatch_kernel_names_.end()
                ? KernelNames{kUnknownKernelIdentity, kUnknownKernelIdentity}
                : kernel_name_iter->second;
        const char *access = v.isWrite ? "write" : "read";
        sink().write(std::format(
            "RACE kernel={} symbol={} dispatch={} type={} access={} reg={} wave={} "
            "lane={} "
            "wg={},{},{} "
            "conflict=unknown\n{}END_RACE\n",
            kernel_names.name, kernel_names.symbol, dispatch_id, space, access, v.index, v.wave,
            v.lane, v.workgroupId.x, v.workgroupId.y, v.workgroupId.z, oss.str()));
      }
    }
  };

  // Multi-XCD: workgroups from the same dispatch can arrive on different
  // partition threads, so detectors_ and dispatch_disasm_ need protection.
  std::lock_guard<std::mutex> lock(dispatch_mutex_);
  detectors_[key] = std::make_unique<RaceDetector>(
      static_cast<int>(num_waves), static_cast<int>(physical_vgpr_count),
      static_cast<int>(sgpr_count), Dim3d(static_cast<int>(wg_id)), std::move(handler),
      /*vmcntNoWait=*/63, lgkmcnt_no_wait_value(wavefronts.front()->cu().arch()),
      models_cdna_ordered_counter_backpressure(wavefronts.front()->cu().arch()));

  auto &det = *detectors_[key];
  auto &dc = dispatch_disasm_[dispatch_id];
  if (!dc)
    dc = std::make_shared<DisasmCache>();
  for (uint32_t w = 0; w < num_waves; ++w) {
    auto state = std::make_unique<RaceWavefrontState>();
    state->race_state = &det.getWaveRaceState(static_cast<int>(w));
    state->disasm = dc;
    wavefronts[w]->set_plugin_state(slot_index(), std::move(state));
  }
}

void RaceDetectorPlugin::onAmdgpuRouteMemoryInstruction(const Instruction &inst,
                                                        amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  auto *rs = s->race_state;
  auto *detector = rs->getDetector();
  auto waveId = rs->getWaveId();

  if (inst.data()->tag() == amdgpu::LOCAL_MEM) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    auto type = d.is_load ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::VGPR_TO_LDS;
    const auto completionOrder = memoryCompletionOrderForRaceDetector(inst, wf.cu().arch());

    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (1ULL << lane)))
        continue;
      int addr = static_cast<int>(d.per_lane_addr[lane]);
      int nBytes = static_cast<int>(d.elem_size);
      if (d.is_load)
        detector->validateRead(addr, waveId, static_cast<int>(lane), nBytes);
      else
        detector->validateWrite(addr, waveId, static_cast<int>(lane), nBytes);
    }
    uint32_t laneAddrs[64];
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      laneAddrs[lane] = static_cast<uint32_t>(d.per_lane_addr[lane]);
    std::vector<uint32_t> registers;
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      registers.resize(d.num_elems);
      uint8_t byte_mask = d.d16_lo ? 0x3 : d.d16_hi ? 0xC : 0xF;
      for (uint32_t i = 0; i < d.num_elems; ++i) {
        registers[i] = logicalBase + i;
        rs->checkVgprWrite(static_cast<int>(logicalBase + i), wf.exec(), byte_mask,
                           completionOrder);
      }
    }
    uint8_t byte_mask = d.d16_lo ? 0x3 : d.d16_hi ? 0xC : 0xF;
    rs->registerLdsEvent(wf.pc, type, std::move(registers), wf.exec(), wf.wf_size(),
                         std::span<const uint32_t>(laneAddrs, wf.wf_size()), d.elem_size, byte_mask,
                         completionOrder);
  }

  if (inst.data()->tag() == amdgpu::GLOBAL_MEM) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    if (d.lds_dst) {
      const auto completionOrder = memoryCompletionOrderForRaceDetector(inst, wf.cu().arch());
      uint32_t perLaneBytes = d.num_elems * d.elem_size;
      if (d.cluster_multicast && d.cluster_mcast_mask != 0) {
        uint32_t selfMask = amdgpu::cluster_multicast_rank_mask(wf.cluster_rank());
        uint32_t peerMask = d.cluster_mcast_mask & ~selfMask;
        if (peerMask != 0)
          warn_cluster_peer_writes_ignored_once();
        if ((d.cluster_mcast_mask & selfMask) == 0)
          return;
      }
      uint32_t ldsAddrs[64];
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
        ldsAddrs[lane] =
            d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * perLaneBytes;
      rs->registerLdsEvent(wf.pc, MemoryEventType::GLOBAL_TO_LDS, {}, d.lane_mask, wf.wf_size(),
                           std::span<const uint32_t>(ldsAddrs, wf.wf_size()), perLaneBytes,
                           /*byteMask=*/0xF, completionOrder);
    } else if (d.is_load && d.dst_reg_base >= wf.vgpr_alloc().base) {
      constexpr MemoryEventType type = MemoryEventType::GLOBAL_TO_VGPR;
      const auto completionOrder = memoryCompletionOrderForRaceDetector(inst, wf.cu().arch());
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_elems);
      uint8_t byte_mask = d.d16_lo ? 0x3 : d.d16_hi ? 0xC : 0xF;
      for (uint32_t i = 0; i < d.num_elems; ++i) {
        registers[i] = logicalBase + i;
        rs->checkVgprWrite(static_cast<int>(logicalBase + i), wf.exec(), byte_mask,
                           completionOrder);
      }
      rs->registerEvent(wf.pc, type, std::move(registers), wf.exec(), byte_mask, completionOrder);
    } else if (!d.is_load) {
      rs->registerEvent(wf.pc, MemoryEventType::VGPR_TO_GLOBAL, {}, wf.exec(),
                        /*byteMask=*/0xF,
                        memoryCompletionOrderForRaceDetector(inst, wf.cu().arch()));
    }
  }

  if (inst.data()->tag() == amdgpu::SCALAR_MEM) {
    auto &d = *inst.data_as<amdgpu::ScalarMemState>();
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.sgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_dwords);
      for (uint32_t i = 0; i < d.num_dwords; ++i)
        registers[i] = logicalBase + i;
      rs->registerEvent(wf.pc, MemoryEventType::GLOBAL_TO_SGPR, std::move(registers), wf.exec());
    }
  }
}

void RaceDetectorPlugin::onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                               uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  s->race_state->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                                uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  s->race_state->checkVgprWriteLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->sgpr_alloc().base;
  s->race_state->checkSgprRead(static_cast<int>(logical_reg));
}

void RaceDetectorPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                          amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s);
  assert(s->race_state);
  s->race_state->prepareForMemoryEvent(memoryCompletionOrderForRaceDetector(inst, wf.cu().arch()));
  s->trace.push(pc);
  s->disasm->record(pc, inst);
}

void RaceDetectorPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                         amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);

  const std::string_view mnemonic = inst.mnemonic();
  const auto &target = wf.wait_target();
  const auto waitCount = waitCountForRaceDetector(mnemonic, static_cast<int>(target.vmcnt),
                                                  static_cast<int>(target.lgkmcnt));
  if (!waitCount)
    return;
  s->race_state->dispatch(*waitCount);
}

void RaceDetectorPlugin::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
  for (auto *wf : wavefronts) {
    auto *s = get_state(wf);
    assert(s && s->race_state);
    s->race_state->flushBarrierPendingEvents();
  }
}

} // namespace rocjitsu::plugins::race_detector
