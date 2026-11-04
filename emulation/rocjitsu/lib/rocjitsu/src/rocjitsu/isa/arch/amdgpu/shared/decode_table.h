// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file decode_table.h
/// @brief Dispatch table storage and emission for generated AMDGPU decoders.

#include "util/host_target.h"

// C++ cannot constant-initialize the difference between a factory and its table.
// On small-model x86-64 ELF the assembler lets the linker resolve that offset,
// so each lookup needs just one indirect call. The portable switch took 4-6% more
// time per valid instruction in x86-64 decoder benchmarks.
#if RJ_HOST_HAS_X86_64_ELF_REL32
#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Link-time displacements from the start of an AMDGPU decoder table.
template <typename Function, std::size_t Size> class DecodeTable {
public:
  Function operator[](std::size_t index) const {
    return reinterpret_cast<Function>(reinterpret_cast<std::uintptr_t>(this) + offsets_[index]);
  }

private:
  std::int32_t offsets_[Size];
};

} // namespace rocjitsu::amdgpu

#define RJ_BEGIN_DECODE_TABLE(name)                                                                \
  {                                                                                                \
    constexpr auto &table = DecoderImpl::name;                                                     \
    asm volatile(                                                                                  \
        ".pushsection .rodata.%c0,\"a\",@progbits\n"                                               \
        ".balign 4\n.hidden %c0\n.globl %c0\n.type %c0,@object\n%c0:\n.size %c0,%c1\n.popsection"  \
        :                                                                                          \
        : "i"(&table), "i"(sizeof(table)));

// Compiler operands keep factories reachable through LTO and archive extraction.
// Numbered subsections put runs in slot order regardless of compiler scheduling.
// The signed PC-relative linker relocation rejects targets outside the 2 GiB range.
#define RJ_DECODE_RUN(first, count, target)                                                        \
  asm volatile(".pushsection .rodata.%c0,\"a\",@progbits\n.subsection %c1\n"                       \
               ".rept %c2\n.long %c3 - %c0\n.endr\n.popsection"                                    \
               :                                                                                   \
               : "i"(&table), "i"(1 + (first)), "i"(count), "i"(static_cast<DecodeFunc>(target)))
#define RJ_END_DECODE_TABLE() }
#endif
