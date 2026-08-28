// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <bit>
#include <span>

namespace rocjitsu::plugins::race_detector {
namespace {
/// RAII guard for ProfilerInterface::beginScope/endScope.
struct ProfileScope {
  ProfilerInterface &p;
  ProfileScope(ProfilerInterface &p, std::string_view key) : p(p) { p.beginScope(key); }
  ~ProfileScope() { p.endScope(); }
};
} // namespace

WaveRaceState::WaveRaceState(int vgprCount, int sgprCount, WaveId waveId, RaceDetector *detector,
                             int vmcntNoWait, int lgkmcntNoWait,
                             bool modelOrderedCounterBackpressure)
    : vmcntNoWait(vmcntNoWait), lgkmcntNoWait(lgkmcntNoWait),
      modelOrderedCounterBackpressure(modelOrderedCounterBackpressure), waveId(waveId),
      detector(detector) {
  vgprMemoryEvents.resize(vgprCount);
  sgprMemoryEvents.resize(sgprCount);
  sgprEventCount.resize(sgprCount, 0);
  for (auto &counts : regEventCount) {
    counts.resize(vgprCount, 0);
  }
}

void WaveRaceState::dispatch(PendingMemoryEvent event) {
  prepareForMemoryEvent(event.completionOrder);
  if (event.isDualOffset) {
    registerDualOffsetLdsEvent(event.pc, event.type, std::move(event.registers), event.execMask,
                               event.waveSize, event.laneBaseAddresses, event.offset0,
                               event.offset1, event.completionOrder);
  } else if (!event.laneBaseAddresses.empty()) {
    registerLdsEvent(event.pc, event.type, std::move(event.registers), event.execMask,
                     event.waveSize, event.laneBaseAddresses, event.bytesPerLane, event.byteMask,
                     event.completionOrder);
  } else {
    registerEvent(event.pc, event.type, std::move(event.registers), event.execMask, event.byteMask,
                  event.completionOrder);
  }
}

void WaveRaceState::dispatch(PendingWaitCount waitCount) {
  if (waitCount.vmcnt >= 0 && waitCount.vmcnt != vmcntNoWait) {
    sWaitCntVmcnt(waitCount.vmcnt);
  }
  if (waitCount.lgkmcnt >= 0 && waitCount.lgkmcnt != lgkmcntNoWait) {
    sWaitCntLgkmcnt(waitCount.lgkmcnt);
  }
}

void WaveRaceState::registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> regIds,
                                  uint64_t execMask, uint8_t byteMask,
                                  MemoryCompletionOrder completionOrder) {
  registerEventWithIntervals(pc, type, std::move(regIds), execMask, byteMask, {}, completionOrder);
}

void WaveRaceState::registerEventWithIntervals(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> regIds, uint64_t execMask,
                                               uint8_t byteMask, IntervalSet ldsIntervals,
                                               MemoryCompletionOrder completionOrder) {
  ProfileScope ps(*profiler_, "registerEvent");
  bool toSgpr = isToSgpr(type);
  if (!toSgpr) {
    for (auto reg : regIds) {
      regEventCountInc(type, reg);
    }
  }

  auto eventId = detector->allocateEventId(waveId, pc, type, std::move(regIds), execMask, byteMask,
                                           std::move(ldsIntervals), completionOrder);
  for (uint32_t reg : detector->events().registers(eventId)) {
    if (toSgpr) {
      sgprMemoryEvents[reg].push_back(eventId);
      sgprEventCount[reg]++;
    } else {
      vgprMemoryEvents[reg].push_back(eventId);
    }
  }
  waveMemoryEvents.push_back(eventId);
}

void WaveRaceState::prepareForMemoryEvent(MemoryCompletionOrder completionOrder) {
  if (!modelOrderedCounterBackpressure || completionOrder == MemoryCompletionOrder::UNORDERED) {
    return;
  }

  // The all-ones no-wait value is also the largest representable number of
  // outstanding operations on these counters. The physical counter is shared
  // by several event kinds, but only same-class occupancy proves which event
  // completed; mixed-class pressure remains deliberately conservative.
  const int capacity = completionOrder == MemoryCompletionOrder::VMEM ? vmcntNoWait : lgkmcntNoWait;
  int orderedPending = 0;
  for (EventId eventId : waveMemoryEvents) {
    if (detector->events().completionOrder(eventId) == completionOrder)
      ++orderedPending;
  }
  if (orderedPending < capacity)
    return;

  for (auto it = waveMemoryEvents.begin(); it != waveMemoryEvents.end(); ++it) {
    const EventId eventId = *it;
    if (detector->events().completionOrder(eventId) != completionOrder)
      continue;
    retireEventRegisters(eventId);
    detector->markEventWaveComplete(eventId);
    if (!detector->events().isTrimmable(eventId))
      barrierPendingEvents.push_back(eventId);
    waveMemoryEvents.erase(it);
    return;
  }
}

void WaveRaceState::registerLdsEvent(uint64_t pc, MemoryEventType type,
                                     std::vector<uint32_t> registers, uint64_t execMask,
                                     int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                     int bytesPerLane, uint8_t byteMask,
                                     MemoryCompletionOrder completionOrder) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    int addr = static_cast<int>(laneBaseAddresses[lane]);
    intervals.append(addr, addr + bytesPerLane);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, byteMask,
                             std::move(intervals), completionOrder);
}

void WaveRaceState::registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> registers, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1,
                                               MemoryCompletionOrder completionOrder) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    int intAddr0 = static_cast<int>(vAddr + static_cast<uint32_t>(offset0) * 8);
    intervals.append(intAddr0, intAddr0 + 8);
    int intAddr1 = static_cast<int>(vAddr + static_cast<uint32_t>(offset1) * 8);
    intervals.append(intAddr1, intAddr1 + 8);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, 0xF, std::move(intervals),
                             completionOrder);
}

void WaveRaceState::retireEventRegisters(EventId eventId) {
  ProfileScope ps(*profiler_, "retireEventRegisters");
  auto eventType = detector->events().type(eventId);
  bool toSgpr = isToSgpr(eventType);
  for (uint32_t regId : detector->events().registers(eventId)) {
    if (toSgpr) {
      removeFromUnorderedList(sgprMemoryEvents[regId], eventId);
      sgprEventCount[regId]--;
    } else {
      removeFromUnorderedList(getVgprMemoryEvents(regId), eventId);
      regEventCountDec(eventType, regId);
    }
  }
}

template <typename Pred> void WaveRaceState::resolveWaitCnt(int limit, Pred isTargetType) {
  auto retireOldest = [&](int toRetire, auto matches) {
    if (toRetire <= 0)
      return;
    int retired = 0;
    size_t write = 0;
    for (size_t read = 0; read < waveMemoryEvents.size(); ++read) {
      EventId eventId = waveMemoryEvents[read];
      if (matches(eventId) && retired < toRetire) {
        ++retired;
        retireEventRegisters(eventId);
        detector->markEventWaveComplete(eventId);
        // Trimmable WAVE_COMPLETE events may be removed from the registry
        // immediately. Only keep non-trimmable events for later barrier retire;
        // otherwise a later barrier could try to retire stale EventIds.
        if (!detector->events().isTrimmable(eventId))
          barrierPendingEvents.push_back(eventId);
      } else {
        waveMemoryEvents[write++] = eventId;
      }
    }
    waveMemoryEvents.resize(write);
  };

  auto targetsCounter = [&](EventId eventId) {
    return isTargetType(detector->events().type(eventId));
  };
  if (limit == 0) {
    retireOldest(static_cast<int>(std::ranges::count_if(waveMemoryEvents, targetsCounter)),
                 targetsCounter);
    return;
  }

  // A nonzero wait guarantees only that at most `limit` events remain on the
  // entire counter. For each independently ordered class, if more than
  // `limit` events from that class are pending, the excess oldest events must
  // have completed. No particular unordered event (notably SMEM or FLAT) can
  // be selected safely.
  for (MemoryCompletionOrder completionOrder :
       {MemoryCompletionOrder::VMEM, MemoryCompletionOrder::LDS}) {
    auto belongsToOrder = [&](EventId eventId) {
      return targetsCounter(eventId) &&
             detector->events().completionOrder(eventId) == completionOrder;
    };
    const int pending = static_cast<int>(std::ranges::count_if(waveMemoryEvents, belongsToOrder));
    retireOldest(pending - limit, belongsToOrder);
  }
}

void WaveRaceState::sWaitCntVmcnt(int vmcnt) {
  resolveWaitCnt(vmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::GLOBAL_TO_VGPR || type == MemoryEventType::VGPR_TO_GLOBAL ||
           type == MemoryEventType::GLOBAL_TO_LDS;
  });
}

void WaveRaceState::sWaitCntLgkmcnt(int lgkmcnt) {
  resolveWaitCnt(lgkmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::LDS_TO_VGPR || type == MemoryEventType::VGPR_TO_LDS ||
           type == MemoryEventType::GLOBAL_TO_SGPR;
  });
}

void WaveRaceState::flushBarrierPendingEvents() {
  ProfileScope ps(*profiler_, "removeEvents");
  for (EventId eventId : barrierPendingEvents) {
    detector->retireEvent(eventId);
  }
  barrierPendingEvents.clear();
}

void WaveRaceState::checkVgprRead(int reg, int lane, uint8_t byteMask) const {
  checkVgprReadLanes(reg, uint64_t{1} << lane, byteMask);
}

void WaveRaceState::checkVgprReadLanes(int reg, uint64_t laneMask, uint8_t byteMask) const {
  if (laneMask == 0)
    return;
  for (EventId eid : vgprMemoryEvents[reg]) {
    uint64_t conflictMask = laneMask & detector->events().execMask(eid);
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().byteMask(eid) & byteMask) != 0 && conflictMask != 0) {
      int lane = std::countr_zero(conflictMask);
      detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, false,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

void WaveRaceState::checkVgprWrite(int reg, int lane, uint8_t byteMask) const {
  checkVgprWriteLanes(reg, uint64_t{1} << lane, byteMask);
}

void WaveRaceState::checkVgprWriteLanes(int reg, uint64_t laneMask, uint8_t byteMask) const {
  if (laneMask == 0)
    return;
  for (EventId eid : vgprMemoryEvents[reg]) {
    uint64_t conflictMask = laneMask & detector->events().execMask(eid);
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().byteMask(eid) & byteMask) != 0 && conflictMask != 0) {
      int lane = std::countr_zero(conflictMask);
      detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, true,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

// Like checkVgprRead but for instructions that read all lanes (e.g. cross-lane ops).
// countr_zero picks the first active lane from the event's exec mask as the
// representative lane for the violation report.
void WaveRaceState::checkVgprReadAllLanes(int reg) const {
  if (getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, reg) != 0 ||
      getRegEventCount(MemoryEventType::LDS_TO_VGPR, reg) != 0) {
    for (EventId eid : vgprMemoryEvents[reg]) {
      if (isToVgpr(detector->events().type(eid)) && (detector->events().byteMask(eid) & 0xF) != 0) {
        int lane = std::countr_zero(detector->events().execMask(eid));
        detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, false,
                                    detector->getWorkgroupId(), eid});
      }
    }
  }
}

void WaveRaceState::checkVgprWrite(int reg, uint64_t execMask, uint8_t byteMask,
                                   MemoryCompletionOrder currentCompletionOrder) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    const MemoryEventType pendingType = detector->events().type(eid);
    const MemoryCompletionOrder pendingCompletionOrder = detector->events().completionOrder(eid);
    const bool orderedWithCurrent = currentCompletionOrder != MemoryCompletionOrder::UNORDERED &&
                                    pendingCompletionOrder == currentCompletionOrder;
    if (!isToVgpr(pendingType) || orderedWithCurrent ||
        (detector->events().byteMask(eid) & byteMask) == 0) {
      continue;
    }

    const uint64_t overlappingLanes = detector->events().execMask(eid) & execMask;
    if (overlappingLanes == 0)
      continue;

    const int lane = static_cast<int>(std::countr_zero(overlappingLanes));
    detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, true,
                                detector->getWorkgroupId(), eid});
  }
}

bool WaveRaceState::isOutstandingFromVgpr(int lane, int reg) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    if (isFromVgpr(detector->events().type(eid)) && detector->events().isActiveForLane(eid, lane)) {
      return true;
    }
  }
  return false;
}

void WaveRaceState::checkSgprRead(int reg) const {
  if (sgprEventCount[reg] == 0) {
    return;
  }
  for (EventId eid : sgprMemoryEvents[reg]) {
    if (isToSgpr(detector->events().type(eid))) {
      detector->getRaceHandler()({RaceViolation::Space::SGPR, reg, waveId.value, -1, false,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

} // namespace rocjitsu::plugins::race_detector
