// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_SIMD_APPLY_H_
#define UTIL_SIMD_APPLY_H_

#include "util/simd.h"

#include <bit>
#include <cstdint>

namespace util {

// --- 32-bit binary ---

template <typename T, typename BinOp>
  requires(has_stdx_simd)
void apply_binary_simd(const uint32_t *src0, const uint32_t *src1, uint32_t *dst, uint32_t n,
                       uint64_t exec, BinOp op) {
  constexpr std::size_t W = native_width_v<T>;
  const uint64_t chunk_full = mask<uint64_t>(static_cast<int>(W));
  for (uint32_t base = 0; base < n; base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = load<T>(src0 + base);
    auto b = load<T>(src1 + base);
    masked_store<T>(dst + base, op(a, b), chunk);
  }
}

template <typename T, typename BinOp>
void apply_binary_scalar(const uint32_t *src0, const uint32_t *src1, uint32_t *dst, uint32_t n,
                         uint64_t exec, BinOp op) {
  for (uint32_t i = 0; i < n; ++i) {
    if (!(exec & (1ULL << i)))
      continue;
    T a = std::bit_cast<T>(src0[i]);
    T b = std::bit_cast<T>(src1[i]);
    dst[i] = std::bit_cast<uint32_t>(op(a, b));
  }
}

// --- 32-bit unary ---

template <typename T, typename UnOp>
  requires(has_stdx_simd)
void apply_unary_simd(const uint32_t *src0, uint32_t *dst, uint32_t n, uint64_t exec, UnOp op) {
  constexpr std::size_t W = native_width_v<T>;
  const uint64_t chunk_full = mask<uint64_t>(static_cast<int>(W));
  for (uint32_t base = 0; base < n; base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = load<T>(src0 + base);
    masked_store<T>(dst + base, op(a), chunk);
  }
}

template <typename T, typename UnOp>
void apply_unary_scalar(const uint32_t *src0, uint32_t *dst, uint32_t n, uint64_t exec, UnOp op) {
  for (uint32_t i = 0; i < n; ++i) {
    if (!(exec & (1ULL << i)))
      continue;
    T a = std::bit_cast<T>(src0[i]);
    dst[i] = std::bit_cast<uint32_t>(op(a));
  }
}

// --- 32-bit ternary (FMA, min3, max3, med3) ---

template <typename T, typename TernOp>
  requires(has_stdx_simd)
void apply_ternary_simd(const uint32_t *src0, const uint32_t *src1, const uint32_t *src2,
                        uint32_t *dst, uint32_t n, uint64_t exec, TernOp op) {
  constexpr std::size_t W = native_width_v<T>;
  const uint64_t chunk_full = mask<uint64_t>(static_cast<int>(W));
  for (uint32_t base = 0; base < n; base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = load<T>(src0 + base);
    auto b = load<T>(src1 + base);
    auto c = load<T>(src2 + base);
    masked_store<T>(dst + base, op(a, b, c), chunk);
  }
}

template <typename T, typename TernOp>
void apply_ternary_scalar(const uint32_t *src0, const uint32_t *src1, const uint32_t *src2,
                          uint32_t *dst, uint32_t n, uint64_t exec, TernOp op) {
  for (uint32_t i = 0; i < n; ++i) {
    if (!(exec & (1ULL << i)))
      continue;
    T a = std::bit_cast<T>(src0[i]);
    T b = std::bit_cast<T>(src1[i]);
    T c = std::bit_cast<T>(src2[i]);
    dst[i] = std::bit_cast<uint32_t>(op(a, b, c));
  }
}

// --- 32-bit comparison (produces a per-lane mask) ---

template <typename T, typename CmpOp>
  requires(has_stdx_simd)
void apply_cmp_simd(const uint32_t *src0, const uint32_t *src1, uint64_t *result_mask, uint32_t n,
                    uint64_t exec, CmpOp op) {
  constexpr std::size_t W = native_width_v<T>;
  const uint64_t chunk_full = mask<uint64_t>(static_cast<int>(W));
  uint64_t out = *result_mask;
  for (uint32_t base = 0; base < n; base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = load<T>(src0 + base);
    auto b = load<T>(src1 + base);
    auto cmp = op(a, b);
    for (std::size_t i = 0; i < W; ++i) {
      if (chunk & (1ULL << i)) {
        if (cmp[i])
          out |= (1ULL << (base + i));
        else
          out &= ~(1ULL << (base + i));
      }
    }
  }
  *result_mask = out;
}

template <typename T, typename CmpOp>
void apply_cmp_scalar(const uint32_t *src0, const uint32_t *src1, uint64_t *result_mask, uint32_t n,
                      uint64_t exec, CmpOp op) {
  uint64_t out = *result_mask;
  for (uint32_t i = 0; i < n; ++i) {
    if (!(exec & (1ULL << i)))
      continue;
    T a = std::bit_cast<T>(src0[i]);
    T b = std::bit_cast<T>(src1[i]);
    if (op(a, b))
      out |= (1ULL << i);
    else
      out &= ~(1ULL << i);
  }
  *result_mask = out;
}

} // namespace util

#endif // UTIL_SIMD_APPLY_H_
