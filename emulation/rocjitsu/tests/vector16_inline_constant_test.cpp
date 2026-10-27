// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector16_inline_constant_test.cpp
/// @brief Decoded 16-bit source carriers, delegated reads and destination windows.

#include "decode_test_util.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu;

template <typename Fields, auto Build>
std::array<uint32_t, 2> encode_vector16(uint16_t opcode, uint16_t source, uint8_t half) {
  Fields fields{};
  fields.vdst = 4;
  fields.src0 = source;
  fields.src1 = 128;
  fields.src2 = 128;
  if constexpr (requires { fields.opsel; })
    fields.opsel = half;
  else
    fields.op_sel = half;
  return Build(opcode, fields);
}

struct Vector16Target {
  rj_code_arch_t arch;
  const char *name;
  uint16_t add_integer;
  uint16_t add_float;
  uint16_t convert_integer;
  std::array<uint32_t, 2> (*encode)(uint16_t, uint16_t, uint8_t);
};

constexpr std::array kTargets{
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA1, "Gfx908", cdna1::kVAddU16Vop3, cdna1::kVAddF16Vop3,
                   cdna1::kVCvtF16U16Vop3,
                   encode_vector16<cdna1::Vop3BuilderFields, cdna1::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA2, "Gfx90a", cdna2::kVAddU16Vop3, cdna2::kVAddF16Vop3,
                   cdna2::kVCvtF16U16Vop3,
                   encode_vector16<cdna2::Vop3BuilderFields, cdna2::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA3, "Gfx942", cdna3::kVAddU16Vop3, cdna3::kVAddF16Vop3,
                   cdna3::kVCvtF16U16Vop3,
                   encode_vector16<cdna3::Vop3BuilderFields, cdna3::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA4, "Gfx950", cdna4::kVAddU16Vop3, cdna4::kVAddF16Vop3,
                   cdna4::kVCvtF16U16Vop3,
                   encode_vector16<cdna4::Vop3BuilderFields, cdna4::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA1, "Gfx1010", rdna1::kVAddNcU16Vop3, rdna1::kVAddF16Vop3,
                   rdna1::kVCvtF16U16Vop3,
                   encode_vector16<rdna1::Vop3BuilderFields, rdna1::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA2, "Gfx1030", rdna2::kVAddNcU16Vop3, rdna2::kVAddF16Vop3,
                   rdna2::kVCvtF16U16Vop3,
                   encode_vector16<rdna2::Vop3BuilderFields, rdna2::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA3, "Gfx1100", rdna3::kVAddNcU16Vop3, rdna3::kVAddF16Vop3,
                   rdna3::kVCvtF16U16Vop3,
                   encode_vector16<rdna3::Vop3BuilderFields, rdna3::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA3_5, "Gfx1150", rdna3_5::kVAddNcU16Vop3,
                   rdna3_5::kVAddF16Vop3, rdna3_5::kVCvtF16U16Vop3,
                   encode_vector16<rdna3_5::Vop3BuilderFields, rdna3_5::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201", rdna4::kVAddNcU16Vop3, rdna4::kVAddF16Vop3,
                   rdna4::kVCvtF16U16Vop3,
                   encode_vector16<rdna4::Vop3BuilderFields, rdna4::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250", cdna5::kVAddNcU16Vop3, cdna5::kVAddF16Vop3,
                   cdna5::kVCvtF16U16Vop3,
                   encode_vector16<cdna5::Vop3BuilderFields, cdna5::build_vop3>},
};

class Vector16InlineConstantTest : public ::testing::TestWithParam<Vector16Target> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam().arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("vector16_inline", config, &memory, &cache);
    decoder = Decoder::create(config.arch);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(kExec);
  }

  void TearDown() override {
    if (wave)
      wave->halt();
  }

  bool legacy_cdna() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_CDNA1 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA3 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA4;
  }

  bool legacy_rdna() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_RDNA2;
  }

  std::unique_ptr<Instruction> decode(uint16_t opcode, uint16_t source, uint8_t half = 0,
                                      std::optional<uint32_t> literal = std::nullopt) {
    const auto encoded = GetParam().encode(opcode, source, half);
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
  }

  void expect_reads(const Operand &source, uint32_t expected, uint32_t mask = UINT32_MAX) {
    amdgpu::RegisterAccess regs(*wave);
    EXPECT_EQ(regs.read_lane(source, 0) & mask, expected & mask);
    std::array<uint32_t, 3> chunk{};
    regs.read_chunk(source, 0, chunk.size(), chunk.data());
    for (uint32_t value : chunk)
      EXPECT_EQ(value & mask, expected & mask);
  }

  void expect_raw_scalar(const Operand &source, uint32_t expected) {
    amdgpu::RegisterAccess regs(*wave);
    EXPECT_EQ(regs.read_scalar(source), expected);
    ASSERT_TRUE(source.const_value().has_value());
    EXPECT_EQ(*source.const_value(), expected);
  }

  void expect_word_result(Instruction &instruction, uint32_t expected) {
    const auto destination = wave->vgpr_alloc().base + 4;
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
      cu->write_vgpr(destination, lane, 0xDEADBEEFu);
      cu->write_vgpr(destination + 1, lane, 0xA5A50000u | lane);
    }
    ASSERT_TRUE(cu->execute_instruction(&instruction, *wave).succeeded());
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
      EXPECT_EQ(cu->read_vgpr(destination, lane),
                (kExec & (uint64_t{1} << lane)) ? expected : 0xDEADBEEFu)
          << "lane=" << lane;
      EXPECT_EQ(cu->read_vgpr(destination + 1, lane), 0xA5A50000u | lane);
    }
    EXPECT_EQ(wave->exec(), kExec);
  }

  void expect_result(Instruction &instruction, uint16_t expected) {
    const auto destination = wave->vgpr_alloc().base + 4;
    auto canary = [](uint32_t lane) { return 0xDEAD0000u | (0xB000u + lane); };
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
      cu->write_vgpr(destination, lane, canary(lane));
    ASSERT_TRUE(cu->execute_instruction(&instruction, *wave).succeeded());
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
      const uint32_t expected_word =
          (kExec & (uint64_t{1} << lane))
              ? (legacy_cdna() ? 0u : canary(lane) & 0xffff0000u) | expected
              : canary(lane);
      EXPECT_EQ(cu->read_vgpr(destination, lane), expected_word) << "lane=" << lane;
    }
    EXPECT_EQ(wave->exec(), kExec);
  }

  static constexpr uint64_t kExec = (uint64_t{1} << 0) | (uint64_t{1} << 2) | (uint64_t{1} << 17);
  amdgpu::GpuMemory memory{"vector16_inline_memory"};
  amdgpu::L2Cache cache{"vector16_inline_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
};

// gfx1100 and gfx1201 hardware agree with LLVM's INT16 source policy: float
// inlines retain FP32 bits for integer/bit consumers. Older targets follow the
// same LLVM operand contract without physical qualification here.
TEST_P(Vector16InlineConstantTest, IntegerAndFloatInputsUseTheirOwnInlineFormats) {
  struct InlineCase {
    uint16_t selector;
    uint32_t fp32;
    uint16_t fp16;
  };
  constexpr std::array cases{
      InlineCase{240, 0x3F000000u, 0x3800}, InlineCase{242, 0x3F800000u, 0x3C00},
      InlineCase{244, 0x40000000u, 0x4000}, InlineCase{248, 0x3E22F983u, 0x3118}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.selector);
    auto integer = decode(GetParam().add_integer, test.selector);
    auto numeric = decode(GetParam().add_float, test.selector);
    ASSERT_NE(integer, nullptr);
    ASSERT_NE(numeric, nullptr);
    ASSERT_NE(integer->src_operand(0), nullptr);
    ASSERT_NE(numeric->src_operand(0), nullptr);
    EXPECT_EQ(integer->src_operand(0)->lane_inline_float_format(),
              Operand::LaneInlineFloatFormat::FP32);
    EXPECT_EQ(numeric->src_operand(0)->lane_inline_float_format(),
              Operand::LaneInlineFloatFormat::F16);
    EXPECT_EQ(integer->dst_operand(0)->preserve_unwritten_bytes(), legacy_rdna());
    EXPECT_EQ(numeric->dst_operand(0)->preserve_unwritten_bytes(), legacy_rdna());
    expect_reads(*integer->src_operand(0), test.fp32);
    expect_reads(*numeric->src_operand(0), test.fp16);
    expect_raw_scalar(*integer->src_operand(0), test.fp32);
    expect_raw_scalar(*numeric->src_operand(0), test.fp32);
    expect_result(*integer, static_cast<uint16_t>(test.fp32));
    expect_result(*numeric, test.fp16);
  }
}

TEST_P(Vector16InlineConstantTest, IntegerToFloatConversionKeepsIntegerSourceFormat) {
  auto conversion = decode(GetParam().convert_integer, 242);
  ASSERT_NE(conversion, nullptr);
  expect_reads(*conversion->src_operand(0), 0x3F800000u);
  expect_result(*conversion, 0);
}

TEST_P(Vector16InlineConstantTest, RegisterAndLiteralCarriersKeepTheirStoredBits) {
  for (const auto opcode : {GetParam().add_integer, GetParam().add_float}) {
    SCOPED_TRACE(opcode);
    cu->write_sgpr(wave->sgpr_alloc().base, 0x40003C00u);
    auto reg = decode(opcode, 0);
    ASSERT_NE(reg, nullptr);
    expect_reads(*reg->src_operand(0), 0x40003C00u);
    expect_result(*reg, 0x3C00u);
    if (!legacy_rdna()) {
      auto high = decode(opcode, 0, 1);
      ASSERT_NE(high, nullptr);
      expect_result(*high, 0x4000u);
    }
    if (legacy_cdna())
      continue; // GFX9 VOP3 does not encode an appended literal.
    for (uint32_t literal : {0x3F800000u, 0x40003C00u}) {
      SCOPED_TRACE(literal);
      auto instruction = decode(opcode, 255, 0, literal);
      ASSERT_NE(instruction, nullptr);
      EXPECT_EQ(instruction->size(), 12u);
      expect_reads(*instruction->src_operand(0), literal, legacy_rdna() ? 0xffffu : UINT32_MAX);
      expect_result(*instruction, static_cast<uint16_t>(literal));
    }
  }
}

TEST_P(Vector16InlineConstantTest, DelegatesKeepOriginFormatAndStagedLaneValues) {
  for (const auto opcode : {GetParam().add_integer, GetParam().add_float}) {
    auto instruction = decode(opcode, 242);
    ASSERT_NE(instruction, nullptr);
    auto &source = *const_cast<Operand *>(instruction->src_operand(0));
    const bool integer = opcode == GetParam().add_integer;
    // Selector encodings are shared across the target ISAs. A fresh synthetic
    // operand has no construction metadata from the original instruction.
    rdna4::Operand fresh(16, rdna4::OperandType::OPR_SRC, 242);
    EXPECT_EQ(fresh.lane_inline_float_format(), Operand::LaneInlineFloatFormat::F16);
    {
      ScopedOperandDelegate delegate(source, &fresh);
      expect_reads(source, integer ? 0x3F800000u : 0x3C00u);
      expect_result(*instruction, integer ? 0u : 0x3C00u);
    }
    fresh.set_lane_inline_float_format(Operand::LaneInlineFloatFormat::FP32);
    {
      ScopedOperandDelegate delegate(source, &fresh);
      expect_reads(source, integer ? 0x3F800000u : 0x3C00u);
      expect_result(*instruction, integer ? 0u : 0x3C00u);
    }

    std::optional<StagedOperand> staged;
    amdgpu::sdwa::stage_source(source, amdgpu::sdwa::WORD_0, false, false, false,
                               amdgpu::sdwa::SourceModifierFormat::NONE, staged, *wave);
    ASSERT_TRUE(staged.has_value());
    {
      ScopedOperandDelegate delegate(source, &*staged);
      amdgpu::RegisterAccess regs(*wave);
      EXPECT_EQ(regs.read_lane(source, 0), integer ? 0u : 0x3C00u);
      std::array<uint32_t, 2> values{};
      regs.read_chunk(source, 0, values.size(), values.data());
      EXPECT_EQ(values[0], integer ? 0u : 0x3C00u);
      EXPECT_EQ(values[1], 0u);
    }
  }
}

TEST_P(Vector16InlineConstantTest, HighHalfSelectionUsesTheFormattedCarrier) {
  // CDNA1-4 retain the model choice from the manual's generic VOP3A OPSEL
  // description. LLVM rejects OPSEL on these ADD16 forms; the disagreement
  // has no physical qualification here. gfx1100/gfx1201 controls are separate.
  if (legacy_rdna())
    GTEST_SKIP() << "RDNA1/2 numeric ADD has no OPSEL form; integer ADD OPSEL is not modeled.";
  for (const auto &[selector, high] : {std::pair{242u, 0x3F80u}, std::pair{248u, 0x3E22u}}) {
    SCOPED_TRACE(selector);
    auto integer = decode(GetParam().add_integer, selector, 1);
    auto numeric = decode(GetParam().add_float, selector, 1);
    ASSERT_NE(integer, nullptr);
    ASSERT_NE(numeric, nullptr);
    expect_result(*integer, high);
    expect_result(*numeric, 0u);
  }
}

class PackedSatDestinationTest : public Vector16InlineConstantTest {};

TEST_P(PackedSatDestinationTest, Vop1PackedResultClearsTheWholeDestinationWord) {
  const auto opcode = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ? rdna1::kVSatPkU8I16Vop1
                                                                  : rdna2::kVSatPkU8I16Vop1;
  const auto encoded = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1
                           ? rdna1::build_vop1(opcode, {.src0 = 0, .vdst = 4})
                           : rdna2::build_vop1(opcode, {.src0 = 0, .vdst = 4});
  const std::array<uint32_t, 2> words{encoded[0], 0};
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);
  ASSERT_EQ(instruction->size(), 4u);
  ASSERT_NE(instruction->dst_operand(0), nullptr);
  EXPECT_FALSE(instruction->dst_operand(0)->preserve_unwritten_bytes());
  cu->write_sgpr(wave->sgpr_alloc().base, 0xFFFF012Cu);
  expect_word_result(*instruction, 0x000000FFu);
}

TEST_P(PackedSatDestinationTest, Vop3PackedResultClearsTheWholeDestinationWord) {
  const auto opcode = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ? rdna1::kVSatPkU8I16Vop3
                                                                  : rdna2::kVSatPkU8I16Vop3;
  auto instruction = decode(opcode, 0);
  ASSERT_NE(instruction, nullptr);
  ASSERT_EQ(instruction->size(), 8u);
  ASSERT_NE(instruction->dst_operand(0), nullptr);
  EXPECT_FALSE(instruction->dst_operand(0)->preserve_unwritten_bytes());
  cu->write_sgpr(wave->sgpr_alloc().base, 0xFFFF012Cu);
  expect_word_result(*instruction, 0x000000FFu);
}

class Bf16InlineConstantTest : public Vector16InlineConstantTest {
protected:
  bool dot() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4;
  }

  bool fp32_carrier() const { return GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5; }

  std::unique_ptr<Instruction> numeric(uint16_t source,
                                       std::optional<uint32_t> literal = std::nullopt) {
    if (!dot()) {
      const auto opcode = fp32_carrier() ? cdna5::kVCvtF32Bf16Vop3 : cdna4::kVCvtF32Bf16Vop3;
      return decode(opcode, source, fp32_carrier() ? 1 : 0, literal);
    }
    std::array<uint32_t, 2> encoded{};
    if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3) {
      encoded = rdna3::build_vop3(rdna3::kVDot2Bf16Bf16Vop3,
                                  {.vdst = 4, .src0 = 128, .src1 = 128, .src2 = source});
    } else {
      encoded = rdna4::build_vop3(rdna4::kVDot2Bf16Bf16Vop3,
                                  {.vdst = 4, .src0 = 128, .src1 = 128, .src2 = source});
    }
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
  }

  const Operand &numeric_source(const Instruction &instruction) const {
    return *instruction.src_operand(dot() ? 2 : 0);
  }

  void expect_numeric_result(Instruction &instruction, uint32_t fp32, uint16_t bf16) {
    if (dot()) {
      expect_result(instruction, bf16);
    } else {
      expect_word_result(instruction, fp32);
    }
  }
};

TEST_P(Bf16InlineConstantTest, NumericInlineSourceKeepsLaneChunkAndRawScalarContracts) {
  // RDNA3/4 DOT accumulator controls are measured on the local cards. The
  // CDNA4 low and CDNA5 high conversion forms remain source-backed checks.
  struct InlineCase {
    uint16_t selector;
    uint32_t fp32;
    uint16_t bf16;
  };
  constexpr std::array cases{
      InlineCase{240, 0x3F000000u, 0x3F00}, InlineCase{241, 0xBF000000u, 0xBF00},
      InlineCase{242, 0x3F800000u, 0x3F80}, InlineCase{243, 0xBF800000u, 0xBF80},
      InlineCase{244, 0x40000000u, 0x4000}, InlineCase{245, 0xC0000000u, 0xC000},
      InlineCase{246, 0x40800000u, 0x4080}, InlineCase{247, 0xC0800000u, 0xC080},
      InlineCase{248, 0x3E22F983u, 0x3E22}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.selector);
    auto instruction = numeric(test.selector);
    ASSERT_NE(instruction, nullptr);
    const Operand &source = numeric_source(*instruction);
    EXPECT_EQ(source.lane_inline_float_format(), fp32_carrier()
                                                     ? Operand::LaneInlineFloatFormat::FP32
                                                     : Operand::LaneInlineFloatFormat::BF16);
    expect_reads(source, fp32_carrier() ? test.fp32 : test.bf16);
    expect_raw_scalar(source, test.fp32);
    // BF16's 1/(2*pi) entry is 0x3e22, not nearest-rounded 0x3e23.
    expect_numeric_result(*instruction, uint32_t{test.bf16} << 16, test.bf16);
  }
}

TEST_P(Bf16InlineConstantTest, RegisterAndLegalLiteralSourcesRetainTheirCarriers) {
  const uint32_t carrier = fp32_carrier() ? 0x3F800000u : 0x00003F80u;
  cu->write_sgpr(wave->sgpr_alloc().base, carrier);
  auto reg = numeric(0);
  ASSERT_NE(reg, nullptr);
  expect_reads(numeric_source(*reg), carrier);
  expect_numeric_result(*reg, 0x3F800000u, 0x3F80u);
  if (dot()) {
    auto literal = numeric(255, carrier);
    ASSERT_NE(literal, nullptr);
    EXPECT_EQ(literal->size(), 12u);
    expect_reads(numeric_source(*literal), carrier);
    expect_numeric_result(*literal, 0x3F800000u, 0x3F80u);
  }
}

TEST_P(Bf16InlineConstantTest, DelegatesFollowOriginFormatAndKeepStagedValuesRaw) {
  auto instruction = numeric(242);
  ASSERT_NE(instruction, nullptr);
  auto &source = const_cast<Operand &>(numeric_source(*instruction));
  const uint32_t carrier = fp32_carrier() ? 0x3F800000u : 0x00003F80u;
  for (const auto format :
       {Operand::LaneInlineFloatFormat::F16, Operand::LaneInlineFloatFormat::FP32,
        Operand::LaneInlineFloatFormat::BF16}) {
    ScopedIsaExecutionBackend backend_scope{&rdna4::execution_backend()};
    rdna4::Operand fresh(16, rdna4::OperandType::OPR_SRC, 242);
    fresh.set_lane_inline_float_format(format);
    ScopedOperandDelegate delegate(source, &fresh);
    expect_reads(source, carrier);
    expect_raw_scalar(source, 0x3F800000u);
    expect_numeric_result(*instruction, 0x3F800000u, 0x3F80u);
  }
  StagedOperand staged(source, static_cast<int>(wave->wf_size()));
  for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
    staged.set_lane(lane, (kExec & (uint64_t{1} << lane)) ? carrier : 0);
  }
  ScopedOperandDelegate delegate(source, &staged);
  amdgpu::RegisterAccess regs(*wave);
  EXPECT_EQ(regs.read_lane(source, 0), carrier);
  std::array<uint32_t, 3> values{};
  regs.read_chunk(source, 0, values.size(), values.data());
  EXPECT_EQ(values, (std::array<uint32_t, 3>{carrier, 0, carrier}));
  expect_numeric_result(*instruction, 0x3F800000u, 0x3F80u);
}

INSTANTIATE_TEST_SUITE_P(Targets, Vector16InlineConstantTest, ::testing::ValuesIn(kTargets),
                         [](const auto &info) { return std::string(info.param.name); });
INSTANTIATE_TEST_SUITE_P(LegacyRdna, PackedSatDestinationTest,
                         ::testing::Values(kTargets[4], kTargets[5]),
                         [](const auto &info) { return std::string(info.param.name); });
INSTANTIATE_TEST_SUITE_P(Bf16Targets, Bf16InlineConstantTest,
                         ::testing::Values(kTargets[3], kTargets[6], kTargets[8], kTargets[9]),
                         [](const auto &info) { return std::string(info.param.name); });

} // namespace
