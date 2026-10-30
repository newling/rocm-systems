// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>

namespace {
using namespace rocjitsu;

constexpr uint64_t kBase = 0x10000;

class AllImageAddressingTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const { return GetParam(); }
};

TEST_P(AllImageAddressingTest, MisalignedLinearPitchNeedsQualifiedRounding) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const amdgpu::ImageLayout layout{.width = 30, .height = 5, .pitch = 35, .element_bytes = 4};
  const std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  const bool qualified = arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA4;
  ASSERT_EQ(addressing.has_value(), qualified);
  if (addressing) {
    EXPECT_EQ(addressing->byte_offset(0, 1, 0), 256u);
  }
}

TEST_P(AllImageAddressingTest, NativeAddressingMatchesIndependentOffsets) {
  // Captured from an independent address oracle before removing the runtime
  // dependency. The expected offsets are constants, not calculated by RJ.
  // Columns cover GFX9, GFX10, GFX11 with GB_ADDR_CONFIG=0x545 and zero,
  // GFX12, and RDNA2 RB+ (Navi21 revision 0x28). Separate physical raw-backing tests qualify
  // GFX11/12 layouts.
  struct Case {
    uint32_t kind, dimensions, bytes, width, height, depth, levels, mip, x, y, z;
    std::array<uint64_t, 6> expected;
  };
  constexpr Case cases[] = {
      {1,
       2,
       1,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {69633ull, 72961ull, 72961ull, 69633ull, 69633ull, 72961ull}},
      {2,
       2,
       1,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {344064ull, 345088ull, 706048ull, 628736ull, 671744ull, 327936ull}},
      {3,
       3,
       1,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {18688ull, 264320ull, 264960ull, 263680ull, 18688ull, 265984ull}},
      {1,
       2,
       2,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {106498ull, 108290ull, 108290ull, 106498ull, 106498ull, 108290ull}},
      {2,
       2,
       2,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {720896ull, 721920ull, 755200ull, 723968ull, 720896ull, 737536ull}},
      {3,
       3,
       2,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {19456ull, 268288ull, 264960ull, 265216ull, 19456ull, 265984ull}},
      {1,
       2,
       4,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {180228ull, 181764ull, 181764ull, 180228ull, 180228ull, 181764ull}},
      {2,
       2,
       4,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {1114112ull, 1115136ull, 1443328ull, 1442816ull, 1376256ull, 1130752ull}},
      {3,
       3,
       4,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {37888ull, 264192ull, 264960ull, 268288ull, 37888ull, 265984ull}},
      {1,
       2,
       8,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {303112ull, 304136ull, 304136ull, 303112ull, 303112ull, 304136ull}},
      {2,
       2,
       8,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {1703936ull, 1704960ull, 3118080ull, 2885632ull, 2883584ull, 1704192ull}},
      {3,
       3,
       8,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {77824ull, 280576ull, 265984ull, 274432ull, 77824ull, 265984ull}},
      {1,
       2,
       16,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {548880ull, 549904ull, 549904ull, 548880ull, 548880ull, 549904ull}},
      {2,
       2,
       16,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {2883584ull, 2884608ull, 4658176ull, 4457472ull, 4456448ull, 2886912ull}},
      {3,
       3,
       16,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {118784ull, 264192ull, 264448ull, 278528ull, 118784ull, 264448ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       0,
       36,
       18,
       2,
       {37984ull, 46240ull, 46176ull, 46176ull, 46176ull, 46176ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       1,
       17,
       8,
       2,
       {43268ull, 40196ull, 40196ull, 39172ull, 39172ull, 40196ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       2,
       8,
       3,
       2,
       {47696ull, 36400ull, 36392ull, 35368ull, 35368ull, 36400ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       3,
       3,
       1,
       2,
       {46620ull, 33308ull, 33308ull, 34332ull, 34332ull, 33308ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       4,
       1,
       0,
       2,
       {46340ull, 33028ull, 33028ull, 34052ull, 34052ull, 33028ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       5,
       0,
       0,
       2,
       {46080ull, 32768ull, 32768ull, 33792ull, 33792ull, 32768ull}},
      {0,
       1,
       4,
       257,
       1,
       3,
       6,
       0,
       256,
       0,
       2,
       {16384ull, 9728ull, 9728ull, 9728ull, 9728ull, 9728ull}},
      {0, 1, 4, 257, 1, 3, 6, 2, 63, 0, 2, {18172ull, 7676ull, 7676ull, 7676ull, 7676ull, 7676ull}},
      {0, 1, 4, 257, 1, 3, 6, 5, 7, 0, 2, {21788ull, 6684ull, 6684ull, 6684ull, 6684ull, 6684ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       0,
       129,
       128,
       16,
       {1728514ull, 3309570ull, 4444162ull, 4429826ull, 1667074ull, 3279874ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       2,
       31,
       31,
       3,
       {382974ull, 611582ull, 817150ull, 825342ull, 301054ull, 596478ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       4,
       7,
       7,
       0,
       {79486ull, 1150ull, 32894ull, 8318ull, 1662ull, 1150ull}},
  };
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const uint32_t group = arch_is_cdna_4_or_lower(arch()) ? 0
                         : arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2
                             ? 1
                         : arch() == ROCJITSU_CODE_ARCH_RDNA3   ? 2
                         : arch() == ROCJITSU_CODE_ARCH_RDNA3_5 ? 3
                                                                : 4;
  for (const auto &c : cases) {
    SCOPED_TRACE(testing::Message() << c.kind << ':' << c.bytes << ':' << c.mip);
    const uint32_t mode = c.kind == 0    ? 0
                          : c.kind == 1  ? (group == 4 ? 2 : 22)
                          : c.kind == 2  ? (group == 0   ? 26
                                            : group == 1 ? 27
                                            : group == 4 ? 4
                                                         : 31)
                          : group == 0   ? 22
                          : group == 1   ? 27
                          : group == 4   ? 2
                          : c.bytes < 16 ? 24
                                         : 27;
    const amdgpu::ImageLayout layout{.width = c.width,
                                     .height = c.height,
                                     .depth = c.depth,
                                     .element_bytes = c.bytes,
                                     .swizzle_mode = mode,
                                     .dimensions = c.dimensions,
                                     .mip_levels = c.levels};
    const std::optional<amdgpu::ImageAddressing> addressing =
        amdgpu::ImageAddressing::prepare(config, layout, c.mip);
    ASSERT_TRUE(addressing.has_value());
    ASSERT_TRUE(addressing->contains(c.x, c.y, c.z));
    EXPECT_EQ(addressing->byte_offset(c.x, c.y, c.z),
              c.expected[arch() == ROCJITSU_CODE_ARCH_RDNA2 ? 5 : group]);
  }
}

TEST_P(AllImageAddressingTest, NativeAddressingRejectsUnsupportedLayouts) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  amdgpu::ImageLayout layout{.width = 37, .height = 19, .depth = 3, .element_bytes = 4};
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout, 1));
  layout.element_bytes = 3;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.element_bytes = 4;
  layout.mip_levels = 0;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.mip_levels = 18;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.mip_levels = 1;
  layout.swizzle_mode = 8;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
}

TEST_P(AllImageAddressingTest, NativeAddressingSeparatesViewBoundsFromLayoutSupport) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const amdgpu::ImageLayout layout{.width = 37, .height = 19, .depth = 3, .element_bytes = 4};
  const std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->contains(36, 18, 2));
  EXPECT_FALSE(addressing->contains(37, 0, 0));
  EXPECT_FALSE(addressing->contains(0, 19, 0));
  EXPECT_FALSE(addressing->contains(0, 0, 3));
}

TEST_P(AllImageAddressingTest, BaseAlignmentDependsOnTheStorageBlock) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  amdgpu::ImageLayout layout{.element_bytes = 4};
  std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase + 256));
  layout.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  addressing = amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase));
  EXPECT_FALSE(addressing->has_supported_base(kBase + 256));
}

TEST(ImageAddressingTest, Navi10ZModeMipTailMatchesIndependentOffsets) {
  // Navi10 revision 1, GB_ADDR_CONFIG=0x44; the small-format tail differs
  // from later RDNA devices. These offsets come from the independent oracle.
  const amdgpu::ImageAddressConfig config{ROCJITSU_CODE_ARCH_RDNA1, 0x44};
  for (const auto &[bytes, mip, offset] :
       {std::array{1u, 0u, 32768u}, std::array{2u, 1u, 18432u}, std::array{2u, 3u, 1024u}}) {
    const amdgpu::ImageLayout layout{
        .width = 67, .height = 45, .element_bytes = bytes, .swizzle_mode = 24, .mip_levels = 7};
    const std::optional<amdgpu::ImageAddressing> addressing =
        amdgpu::ImageAddressing::prepare(config, layout, mip);
    ASSERT_TRUE(addressing);
    EXPECT_EQ(addressing->byte_offset(0, 0, 0), offset);
  }
}

TEST(ImageAddressingTest, UnqualifiedCdnaTilingRegistersAreRejected) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2}) {
    const amdgpu::ImageAddressConfig config{arch, 0x82004};
    const amdgpu::ImageLayout layout{.element_bytes = 4, .swizzle_mode = 22};
    EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  }
}

TEST(ImageAddressingTest, RejectsArchitecturesWithoutImageMemoryInstructions) {
  const amdgpu::ImageLayout layout{.element_bytes = 4};
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I,
        ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch);
    EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  }
}

INSTANTIATE_TEST_SUITE_P(Architectures, AllImageAddressingTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace
