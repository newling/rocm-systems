// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Array-level SIMD-vs-scalar correctness tests. Each test applies an
// operation kernel to raw uint32_t arrays via util::apply_*_simd (SIMD
// chunked loop) and util::apply_*_scalar (plain lane loop), and asserts
// bit-identical results. No ComputeUnit, Wavefront, Decoder, or
// Instruction — just arrays in, arrays out.

#include "util/simd_apply.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>

namespace {

constexpr uint32_t N = 64;
constexpr uint32_t SENTINEL = 0xDEADBEEFu;

struct Arrays {
  std::array<uint32_t, N> src0{};
  std::array<uint32_t, N> src1{};
  std::array<uint32_t, N> src2{};
  std::array<uint32_t, N> simd_dst{};
  std::array<uint32_t, N> scalar_dst{};

  void fill_sentinel() {
    simd_dst.fill(SENTINEL);
    scalar_dst.fill(SENTINEL);
  }

  void seed_random(uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < N; ++i) {
      src0[i] = static_cast<uint32_t>(rng());
      src1[i] = static_cast<uint32_t>(rng());
      src2[i] = static_cast<uint32_t>(rng());
    }
  }

  static uint32_t finite_normal(uint32_t raw) {
    uint32_t mantissa = raw & 0x007FFFFFu;
    uint32_t exp = 0x40u | ((raw >> 23) & 0x7Eu);
    uint32_t sign = raw & 0x80000000u;
    return sign | (exp << 23) | mantissa;
  }

  void seed_finite_floats(uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < N; ++i) {
      src0[i] = finite_normal(static_cast<uint32_t>(rng()));
      src1[i] = finite_normal(static_cast<uint32_t>(rng()));
      src2[i] = finite_normal(static_cast<uint32_t>(rng()));
    }
  }
};

template <typename T, typename BinOp> void check_binary(Arrays &a, uint64_t exec, BinOp op) {
  a.fill_sentinel();
  util::apply_binary_simd<T>(a.src0.data(), a.src1.data(), a.simd_dst.data(), N, exec, op);
  util::apply_binary_scalar<T>(a.src0.data(), a.src1.data(), a.scalar_dst.data(), N, exec, op);
  EXPECT_EQ(a.simd_dst, a.scalar_dst);
  for (uint32_t i = 0; i < N; ++i) {
    if (!(exec & (1ULL << i)))
      EXPECT_EQ(a.simd_dst[i], SENTINEL) << "inactive lane " << i << " clobbered";
  }
}

template <typename T, typename UnOp> void check_unary(Arrays &a, uint64_t exec, UnOp op) {
  a.fill_sentinel();
  util::apply_unary_simd<T>(a.src0.data(), a.simd_dst.data(), N, exec, op);
  util::apply_unary_scalar<T>(a.src0.data(), a.scalar_dst.data(), N, exec, op);
  EXPECT_EQ(a.simd_dst, a.scalar_dst);
  for (uint32_t i = 0; i < N; ++i) {
    if (!(exec & (1ULL << i)))
      EXPECT_EQ(a.simd_dst[i], SENTINEL) << "inactive lane " << i << " clobbered";
  }
}

template <typename T, typename TernOp> void check_ternary(Arrays &a, uint64_t exec, TernOp op) {
  a.fill_sentinel();
  util::apply_ternary_simd<T>(a.src0.data(), a.src1.data(), a.src2.data(), a.simd_dst.data(), N,
                              exec, op);
  util::apply_ternary_scalar<T>(a.src0.data(), a.src1.data(), a.src2.data(), a.scalar_dst.data(), N,
                                exec, op);
  EXPECT_EQ(a.simd_dst, a.scalar_dst);
  for (uint32_t i = 0; i < N; ++i) {
    if (!(exec & (1ULL << i)))
      EXPECT_EQ(a.simd_dst[i], SENTINEL) << "inactive lane " << i << " clobbered";
  }
}

template <typename T, typename CmpOp> void check_cmp(Arrays &a, uint64_t exec, CmpOp op) {
  uint64_t simd_mask = 0xAAAA'AAAA'AAAA'AAAAULL;
  uint64_t scalar_mask = simd_mask;
  util::apply_cmp_simd<T>(a.src0.data(), a.src1.data(), &simd_mask, N, exec, op);
  util::apply_cmp_scalar<T>(a.src0.data(), a.src1.data(), &scalar_mask, N, exec, op);
  EXPECT_EQ(simd_mask, scalar_mask);
  uint64_t inactive = ~exec;
  EXPECT_EQ(simd_mask & inactive, 0xAAAA'AAAA'AAAA'AAAAULL & inactive) << "inactive bits modified";
}

// ============================================================
// Binary float ops (covers vop2_simd, vop2_minmax, vop3_binary)
// ============================================================

TEST(SimdApply, BinaryFloatAdd_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(42);
  check_binary<float>(a, ~0ULL, std::plus<>{});
}

TEST(SimdApply, BinaryFloatAdd_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(43);
  check_binary<float>(a, 0xA5A5'F0F0'1234'8001ULL, std::plus<>{});
}

TEST(SimdApply, BinaryFloatAdd_EmptyExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(44);
  check_binary<float>(a, 0ULL, std::plus<>{});
}

TEST(SimdApply, BinaryFloatSub) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(45);
  check_binary<float>(a, ~0ULL, std::minus<>{});
}

TEST(SimdApply, BinaryFloatMul) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(46);
  check_binary<float>(a, ~0ULL, std::multiplies<>{});
}

// ============================================================
// Binary integer ops (covers vop2_simd int section)
// ============================================================

TEST(SimdApply, BinaryIntAdd) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(47);
  check_binary<uint32_t>(a, ~0ULL, std::plus<>{});
}

TEST(SimdApply, BinaryIntSub) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(48);
  check_binary<uint32_t>(a, 0xFFFF'0000'FFFF'0000ULL, std::minus<>{});
}

TEST(SimdApply, BinaryBitwiseAnd) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(49);
  check_binary<uint32_t>(a, ~0ULL, std::bit_and<>{});
}

TEST(SimdApply, BinaryBitwiseOr) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(50);
  check_binary<uint32_t>(a, ~0ULL, std::bit_or<>{});
}

TEST(SimdApply, BinaryBitwiseXor) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(51);
  check_binary<uint32_t>(a, ~0ULL, std::bit_xor<>{});
}

TEST(SimdApply, BinaryLshlrev) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(60);
  for (auto &v : a.src0)
    v &= 31u;
  check_binary<uint32_t>(a, ~0ULL, [](auto a, auto b) { return b << a; });
}

TEST(SimdApply, BinaryLshrrev) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(61);
  for (auto &v : a.src0)
    v &= 31u;
  check_binary<uint32_t>(a, ~0ULL, [](auto a, auto b) { return b >> a; });
}

TEST(SimdApply, BinaryMaxU32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(62);
  check_binary<uint32_t>(a, ~0ULL, [](auto a, auto b) -> decltype(a) {
    if constexpr (requires { a.size(); })
      return util::stdx::max(a, b);
    else
      return std::max(a, b);
  });
}

TEST(SimdApply, BinaryMinU32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(63);
  check_binary<uint32_t>(a, ~0ULL, [](auto a, auto b) -> decltype(a) {
    if constexpr (requires { a.size(); })
      return util::stdx::min(a, b);
    else
      return std::min(a, b);
  });
}

// ============================================================
// Unary ops (covers vop1_simd)
// ============================================================

TEST(SimdApply, UnaryNegate) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(52);
  check_unary<float>(a, ~0ULL, std::negate<>{});
}

TEST(SimdApply, UnaryNegate_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(53);
  check_unary<float>(a, 0x0000'FFFF'0000'FFFFULL, std::negate<>{});
}

TEST(SimdApply, UnaryAbs) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(54);
  check_unary<uint32_t>(a, ~0ULL, [](auto v) { return v & 0x7FFFFFFFu; });
}

TEST(SimdApply, UnaryBitwiseNot) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(55);
  check_unary<uint32_t>(a, ~0ULL, std::bit_not<>{});
}

// ============================================================
// Ternary ops (covers vop2_fma, vop3_ternary_fp, vop3_fmac)
// ============================================================

TEST(SimdApply, TernaryFma_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(70);
  check_ternary<float>(a, ~0ULL, [](auto x, auto y, auto z) { return x * y + z; });
}

TEST(SimdApply, TernaryFma_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(71);
  check_ternary<float>(a, 0xF0F0'F0F0'F0F0'F0F0ULL,
                       [](auto x, auto y, auto z) { return x * y + z; });
}

TEST(SimdApply, TernaryMad) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(72);
  check_ternary<float>(a, ~0ULL, [](auto x, auto y, auto z) { return x * y + z; });
}

TEST(SimdApply, TernaryMax3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(73);
  check_ternary<float>(a, ~0ULL, [](auto x, auto y, auto z) -> decltype(x) {
    if constexpr (requires { x.size(); }) {
      auto t = util::stdx::max(x, y);
      return util::stdx::max(t, z);
    } else {
      return std::max({x, y, z});
    }
  });
}

TEST(SimdApply, TernaryMin3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(74);
  check_ternary<float>(a, ~0ULL, [](auto x, auto y, auto z) -> decltype(x) {
    if constexpr (requires { x.size(); }) {
      auto t = util::stdx::min(x, y);
      return util::stdx::min(t, z);
    } else {
      return std::min({x, y, z});
    }
  });
}

TEST(SimdApply, TernaryIntMad) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(75);
  check_ternary<uint32_t>(a, ~0ULL, [](auto x, auto y, auto z) { return x * y + z; });
}

// ============================================================
// Comparison ops (covers vopc_simd, vopc_vop3_*)
// ============================================================

TEST(SimdApply, CmpGtF32_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(80);
  check_cmp<float>(a, ~0ULL, [](auto x, auto y) { return x > y; });
}

TEST(SimdApply, CmpGtF32_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(81);
  check_cmp<float>(a, 0xF0F0'0F0F'F0F0'0F0FULL, [](auto x, auto y) { return x > y; });
}

TEST(SimdApply, CmpEqU32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(82);
  for (uint32_t i = 0; i < N; i += 4)
    a.src1[i] = a.src0[i];
  check_cmp<uint32_t>(a, ~0ULL, [](auto x, auto y) { return x == y; });
}

TEST(SimdApply, CmpLtF32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(83);
  check_cmp<float>(a, ~0ULL, [](auto x, auto y) { return x < y; });
}

TEST(SimdApply, CmpGeU32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(84);
  check_cmp<uint32_t>(a, ~0ULL, [](auto x, auto y) { return x >= y; });
}

TEST(SimdApply, CmpNeU32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_random(85);
  check_cmp<uint32_t>(a, 0x5555'5555'5555'5555ULL, [](auto x, auto y) { return x != y; });
}

// ============================================================
// VOP3-style source modifiers (abs/neg on inputs)
// ============================================================

TEST(SimdApply, BinaryWithAbsSrc0) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(90);
  auto op = [](auto x, auto y) -> decltype(x) {
    if constexpr (requires { x.size(); })
      return std::bit_cast<decltype(x)>(std::bit_cast<util::native<uint32_t>>(x) & 0x7FFFFFFFu) + y;
    else
      return std::fabs(x) + y;
  };
  check_binary<float>(a, ~0ULL, op);
}

TEST(SimdApply, BinaryWithNegSrc1) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(91);
  auto op = [](auto x, auto y) -> decltype(x) {
    if constexpr (requires { x.size(); })
      return x + std::bit_cast<decltype(y)>(std::bit_cast<util::native<uint32_t>>(y) ^ 0x80000000u);
    else
      return x + (-y);
  };
  check_binary<float>(a, ~0ULL, op);
}

// ============================================================
// VOP3-style destination modifiers (omod/clamp)
// ============================================================

TEST(SimdApply, UnaryWithOmodTimes2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(92);
  check_unary<float>(a, ~0ULL, [](auto v) { return v * 2.0f; });
}

TEST(SimdApply, UnaryWithClamp01) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP();
    return;
  }
  Arrays a;
  a.seed_finite_floats(93);
  auto op = [](auto v) -> decltype(v) {
    if constexpr (requires { v.size(); }) {
      util::stdx::where(v < 0.0f, v) = 0.0f;
      util::stdx::where(v > 1.0f, v) = 1.0f;
      return v;
    } else {
      return std::clamp(v, 0.0f, 1.0f);
    }
  };
  check_unary<float>(a, ~0ULL, op);
}

} // namespace
