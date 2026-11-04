// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::execution_fuzz {

/// @brief Minimum instruction payload size for a generated kernel.
/// @details Leaves room for the prologue, several operations, and termination.
inline constexpr size_t kMinGeneratedKernelBytes = 128;

/// @brief Emit a bounded kernel into the instruction payload.
/// @details The kernel ends with an explicit s_endpgm.
void write_generated_kernel(std::span<uint8_t> payload, rj_code_arch_t arch, uint64_t seed);

} // namespace rocjitsu::execution_fuzz
