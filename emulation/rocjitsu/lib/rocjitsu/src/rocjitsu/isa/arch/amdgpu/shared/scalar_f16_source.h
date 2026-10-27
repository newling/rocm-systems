// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_f16_source.h
/// @brief Format inline floating constants for numeric scalar F16 sources.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_F16_SOURCE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_F16_SOURCE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

[[nodiscard]] inline uint32_t read_scalar_f16_carrier(const Operand &source, const Wavefront &wf,
                                                      bool inline_high_half = false) {
  const Operand *effective = &source;
  while (effective->delegate())
    effective = effective->delegate();
  if (effective->reads_value() && effective->has_register_selector()) {
    if (auto value = resolve_inline_float16_statically(effective->encoding_value()))
      return uint32_t{*value} << (inline_high_half ? 16 : 0);
  }
  // Registers, integer inline constants and literals retain their full carrier.
  // The RDNA3.5/RDNA4/CDNA5 manuals describe low-half-only F16 inline floats,
  // which would make HI conversion read zero. gfx1201 hardware instead places
  // the numeric float where HI reads it. CDNA5 and RDNA3.5 follow that measured
  // behavior by extrapolation; neither target has physical qualification here.
  return RegisterAccess(wf).read_scalar(source);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_F16_SOURCE_H_
