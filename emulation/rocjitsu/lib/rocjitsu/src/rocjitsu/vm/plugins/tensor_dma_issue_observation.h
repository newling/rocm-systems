// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_issue_observation.h
/// @brief LDS bytes of one tensor DMA instruction, reported when it issues.

#pragma once

#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief One contiguous run of LDS bytes touched by a tensor DMA transfer.
struct TensorDmaLdsRange {
  uint32_t address = 0; ///< Absolute CU LDS byte address.
  uint32_t bytes = 0;

  bool operator==(const TensorDmaLdsRange &) const = default;
};

/// @brief A tensor DMA instruction at issue, before any of its data moves.
///
/// @details Every tensor DMA instruction that issues is reported once, including
/// one that moves no data, because each occupies a TENSORcnt slot until it
/// completes. The LDS ranges are the bytes the transfer will write (loads) or
/// read (stores), in element order, with adjacent elements merged. A load
/// writes every element of its tile, zero-filling those outside the tensor; a
/// store reads only in-bounds elements. Padding bytes and bytes outside the
/// wave's LDS allocation are excluded.
///
/// The ranges borrow execution-owned storage and are valid only for the
/// duration of the callback.
struct TensorDmaIssueObservation {
  bool is_load = true; ///< True for global-to-LDS, false for LDS-to-global.
  uint32_t element_size_bytes = 0;
  std::span<const TensorDmaLdsRange> lds_ranges{};
};

} // namespace rocjitsu::amdgpu
