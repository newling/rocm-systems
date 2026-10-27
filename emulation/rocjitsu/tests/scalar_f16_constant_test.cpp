// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_f16_constant_test.cpp
/// @brief Decoded scalar F16 source formatting and full-word destination controls.

#include "decode_test_util.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace {

using namespace rocjitsu;

TEST(ScalarF16CompatibilityTest, Rdna35MinFormatsNumericInlineFloats) {
  amdgpu::GpuMemory memory{"rdna35_scalar_f16"};
  amdgpu::L2Cache cache{"rdna35_scalar_f16_cache"};
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA3_5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 16;
  config.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna35_scalar_f16", config, &memory, &cache);
  auto decoder = Decoder::create(config.arch);
  ASSERT_NE(cu, nullptr);
  ASSERT_NE(decoder, nullptr);
  auto *wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0);
  const std::array<uint32_t, 2> words{
      build_sop2_encoding(config.arch, rdna3_5::kSMinF16Sop2, 4, 242, 244), 0};
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);
  cu->write_sgpr(wave->sgpr_alloc().base + 4, 0xDEADBEEFu);
  EXPECT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base + 4), 0x00003C00u);
  wave->halt();
}

struct ScalarF16Target {
  rj_code_arch_t arch;
  const char *name;
};

struct ScalarOperation {
  const char *mnemonic;
  uint16_t rdna4_opcode;
  uint16_t cdna5_opcode;
};

constexpr ScalarOperation kAdd{"s_add_f16", rdna4::kSAddF16Sop2, cdna5::kSAddF16Sop2};
constexpr ScalarOperation kConvertLow{"s_cvt_f32_f16", rdna4::kSCvtF32F16Sop1,
                                      cdna5::kSCvtF32F16Sop1};
constexpr ScalarOperation kConvertHigh{"s_cvt_hi_f32_f16", rdna4::kSCvtHiF32F16Sop1,
                                       cdna5::kSCvtHiF32F16Sop1};

class ScalarF16ConstantTest : public ::testing::TestWithParam<ScalarF16Target> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam().arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("scalar_f16_constants", config, &memory, &cache);
    decoder = Decoder::create(GetParam().arch);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(0);
  }

  void TearDown() override {
    if (wave) {
      wave->halt();
    }
  }

  uint16_t opcode(const ScalarOperation &operation) const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4 ? operation.rdna4_opcode
                                                       : operation.cdna5_opcode;
  }

  void write_sgpr(uint32_t reg, uint32_t value) {
    cu->write_sgpr(wave->sgpr_alloc().base + reg, value);
  }

  void expect_execution(std::span<const uint32_t> words, uint32_t byte_count,
                        std::string_view mnemonic, uint32_t expected) {
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    ASSERT_EQ(std::string_view(instruction->mnemonic()), mnemonic);
    ASSERT_EQ(instruction->size(), byte_count);
    write_sgpr(kDestination, 0xDEADBEEFu);
    write_sgpr(kDestination + 1, 0xA5A5A5A5u);
    ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
    EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base + kDestination), expected);
    EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base + kDestination + 1), 0xA5A5A5A5u);
    EXPECT_EQ(wave->exec(), 0u);
  }

  void expect_sop1(const ScalarOperation &operation, uint16_t source, uint32_t expected,
                   std::optional<uint32_t> literal = std::nullopt) {
    const std::array<uint32_t, 2> words{
        build_sop1_encoding(GetParam().arch, opcode(operation), kDestination, source),
        literal.value_or(0)};
    ASSERT_NO_FATAL_FAILURE(
        expect_execution(words, literal ? 8u : 4u, operation.mnemonic, expected));
  }

  void expect_sop2(const ScalarOperation &operation, uint16_t source0, uint16_t source1,
                   uint32_t expected, std::optional<uint32_t> literal = std::nullopt) {
    const std::array<uint32_t, 2> words{
        build_sop2_encoding(GetParam().arch, opcode(operation), kDestination, source0, source1),
        literal.value_or(0)};
    ASSERT_NO_FATAL_FAILURE(
        expect_execution(words, literal ? 8u : 4u, operation.mnemonic, expected));
  }

  void expect_pseudo(const ScalarOperation &operation, uint16_t source, uint32_t expected,
                     std::optional<uint32_t> literal = std::nullopt) {
    const std::array<uint32_t, 2> encoded =
        GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4
            ? rdna4::build_vop3(opcode(operation), {.vdst = kDestination,
                                                    .src0 = source,
                                                    .src1 = scalar_positive_inline_u32(0),
                                                    .src2 = scalar_positive_inline_u32(0)})
            : cdna5::build_vop3(opcode(operation), {.vdst = kDestination,
                                                    .src0 = source,
                                                    .src1 = scalar_positive_inline_u32(0),
                                                    .src2 = scalar_positive_inline_u32(0)});
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    ASSERT_NO_FATAL_FAILURE(
        expect_execution(words, literal ? 12u : 8u, operation.mnemonic, expected));
  }

  static constexpr uint8_t kDestination = 4;
  amdgpu::GpuMemory memory{"scalar_f16_constant_memory"};
  amdgpu::L2Cache cache{"scalar_f16_constant_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
};

INSTANTIATE_TEST_SUITE_P(Gfx12Targets, ScalarF16ConstantTest,
                         ::testing::Values(ScalarF16Target{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201"},
                                           ScalarF16Target{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250"}),
                         [](const ::testing::TestParamInfo<ScalarF16Target> &info) {
                           return std::string(info.param.name);
                         });

TEST_P(ScalarF16ConstantTest, AddFormatsOnlyNumericF16InlineFloatsAndClearsUpperBits) {
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, 242, 242, 0x00004000u)); // 1.0 + 1.0
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, 244, 243, 0x00003C00u)); // 2.0 + -1.0

  write_sgpr(0, 0xDEAD3C00u);
  write_sgpr(1, 0xBEEF4000u);
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, 0, 1, 0x00004200u));   // low halves: 1.0 + 2.0
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, 240, 0, 0x00003E00u)); // 0.5 + SGPR 1.0
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, 255, 0, 0x00003E00u, 0x00003800u));
  // An integer inline 1 remains the bit pattern 0x0001, not the floating number 1.0.
  // Adding that tiny value to 1.0 rounds to 1.0, with or without input flushing.
  ASSERT_NO_FATAL_FAILURE(expect_sop2(kAdd, scalar_positive_inline_u32(1), 0, 0x00003C00u));
}

TEST_P(ScalarF16ConstantTest, LowAndHighConversionsUseNumericInlineFloats) {
  struct InlineCase {
    uint16_t source;
    uint32_t expected;
  };
  constexpr std::array kInlineCases{
      InlineCase{240, 0x3F000000u}, InlineCase{241, 0xBF000000u}, InlineCase{242, 0x3F800000u},
      InlineCase{243, 0xBF800000u}, InlineCase{244, 0x40000000u}, InlineCase{245, 0xC0000000u},
      InlineCase{246, 0x40800000u}, InlineCase{247, 0xC0800000u},
  };
  for (const InlineCase &test : kInlineCases) {
    SCOPED_TRACE(test.source);
    ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertLow, test.source, test.expected));
    // gfx1201 hardware supplies the numeric float to HI despite the manuals'
    // low-half-only F16 inline wording, which would predict zero. The gfx1250
    // expectation extrapolates that measurement without physical qualification;
    // the shared RDNA3.5 handler follows the same unqualified extrapolation.
    ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, test.source, test.expected));
  }
}

TEST_P(ScalarF16ConstantTest, ConversionsPreserveRegisterAndLiteralHalfSelection) {
  write_sgpr(0, 0x40003C00u); // low 1.0h, high 2.0h
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertLow, 0, 0x3F800000u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, 0, 0x40000000u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertLow, 255, 0x3F800000u, 0x00003C00u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, 255, 0u, 0x00003C00u));
  // Literal data equal to an inline selector is still literal data.
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, 255, 0u, 242u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, 255, 0x40000000u, 0x40003C00u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, scalar_positive_inline_u32(1), 0u));
  write_sgpr(0, 0x3C004000u); // low 2.0h, high 1.0h
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertLow, 0, 0x40000000u));
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kConvertHigh, 0, 0x3F800000u));
}

TEST_P(ScalarF16ConstantTest, DelegationUsesTheEffectiveRegisterSource) {
  ScopedIsaExecutionBackend execution_backend_scope{GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4
                                                        ? &rdna4::execution_backend()
                                                        : &cdna5::execution_backend()};
  const std::array<uint32_t, 2> words{
      build_sop2_encoding(GetParam().arch, opcode(kAdd), kDestination, 242, 242), 0u};
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);
  std::unique_ptr<Operand> replacement;
  if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4) {
    replacement = std::make_unique<rdna4::Operand>(16, rdna4::OperandType::OPR_SSRC, 0);
  } else {
    replacement = std::make_unique<cdna5::Operand>(16, cdna5::OperandType::OPR_SSRC, 0);
  }
  write_sgpr(0, 0xDEAD4000u); // The delegated source is register half 2.0.
  write_sgpr(kDestination, 0xDEADBEEFu);
  const Operand *source = instruction->src_operand(0);
  ASSERT_NE(source, nullptr);
  // The decoded instruction owns mutable operands; its inspection API is const.
  ScopedOperandDelegate binding(*const_cast<Operand *>(source), replacement.get());
  ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base + kDestination), 0x00004200u); // 2.0 + 1.0
  EXPECT_EQ(wave->exec(), 0u);
}

TEST_P(ScalarF16ConstantTest, PseudoScalarF16WritesOnlyOneZeroPaddedScalarWord) {
  struct PseudoCase {
    ScalarOperation operation;
    uint32_t expected;
  };
  constexpr std::array kPseudoCases{
      PseudoCase{{"v_s_exp_f16", rdna4::kVSExpF16Vop3, cdna5::kVSExpF16Vop3}, 0x00004C00u},
      PseudoCase{{"v_s_log_f16", rdna4::kVSLogF16Vop3, cdna5::kVSLogF16Vop3}, 0x00004000u},
      PseudoCase{{"v_s_rcp_f16", rdna4::kVSRcpF16Vop3, cdna5::kVSRcpF16Vop3}, 0x00003400u},
      PseudoCase{{"v_s_rsq_f16", rdna4::kVSRsqF16Vop3, cdna5::kVSRsqF16Vop3}, 0x00003800u},
      PseudoCase{{"v_s_sqrt_f16", rdna4::kVSSqrtF16Vop3, cdna5::kVSSqrtF16Vop3}, 0x00004000u},
  };
  write_sgpr(0, 0x3C004400u); // low 4.0h; high 1.0h must not be consumed.
  for (const PseudoCase &test : kPseudoCases) {
    SCOPED_TRACE(test.operation.mnemonic);
    ASSERT_NO_FATAL_FAILURE(expect_pseudo(test.operation, 0, test.expected));
    ASSERT_NO_FATAL_FAILURE(expect_pseudo(test.operation, 255, test.expected, 0x00004400u));
    // CDNA5 permits numeric float inlines here; RDNA4 excludes these selectors.
    if (GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5) {
      ASSERT_NO_FATAL_FAILURE(expect_pseudo(test.operation, 246, test.expected)); // 4.0h
    }
  }
}

TEST_P(ScalarF16ConstantTest, IntegerBitConsumersRetainTheFp32InlinePattern) {
  constexpr ScalarOperation kSext{"s_sext_i32_i16", rdna4::kSSextI32I16Sop1,
                                  cdna5::kSSextI32I16Sop1};
  ASSERT_NO_FATAL_FAILURE(expect_sop1(kSext, 242, 0u)); // low 16 of FP32 1.0 is zero.
  struct PackCase {
    ScalarOperation operation;
    uint32_t expected;
  };
  constexpr std::array kPackCases{
      PackCase{{"s_pack_ll_b32_b16", rdna4::kSPackLlB32B16Sop2, cdna5::kSPackLlB32B16Sop2}, 0u},
      PackCase{{"s_pack_lh_b32_b16", rdna4::kSPackLhB32B16Sop2, cdna5::kSPackLhB32B16Sop2},
               0x3F800000u},
      PackCase{{"s_pack_hh_b32_b16", rdna4::kSPackHhB32B16Sop2, cdna5::kSPackHhB32B16Sop2},
               0x3F803F80u},
      PackCase{{"s_pack_hl_b32_b16", rdna4::kSPackHlB32B16Sop2, cdna5::kSPackHlB32B16Sop2},
               0x00003F80u},
  };
  for (const PackCase &test : kPackCases) {
    SCOPED_TRACE(test.operation.mnemonic);
    ASSERT_NO_FATAL_FAILURE(expect_sop2(test.operation, 242, 242, test.expected));
  }
}

} // namespace
