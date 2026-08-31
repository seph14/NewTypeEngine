#pragma once

// CPU-side bulk pixel operations (sRGB decode, float<->half conversion,
// 2x2 box downsample, UNORM8 quantize) used by the texture loading and
// BCn compression paths.
//
// Debug builds compile these files with /Od and no auto-vectorization, and
// luisa's half type never uses F16C on MSVC (half_float only enables it under
// __F16C__, which MSVC doesn't define without /arch:AVX2) — so the scalar
// loops these helpers replace were the dominant texture-load cost in the
// Debug iteration config.
//
// Design contract: every vectorized path is bit-identical to its scalar
// fallback (same rounding semantics, same operation order). Debug builds
// spot-check the first elements of each call against the scalar path.

#include <luisa/core/basic_types.h>
#include "cinder/CinderAssert.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  #define NT_MSVC_X86 1
  #include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
  #include <x86intrin.h>
#endif

#if defined(NT_MSVC_X86) || defined(__SSE2__)
  #define NT_SSE2 1
#endif

namespace newtype::util {

//==============================================================================
// Feature detection
//==============================================================================

namespace detail {

[[nodiscard]] inline bool f16c_available() noexcept {
#if defined(__F16C__)
    return true;  // compiler flag implies the target supports it
#elif defined(NT_MSVC_X86)
    // F16C lives in the AVX instruction group; it needs both CPU support
    // (leaf 1 ECX bit 29) and OS AVX state enablement (OSXSAVE + XCR0 XMM/YMM).
    static const bool ok = []() noexcept {
        int regs[4];
        __cpuid(regs, 1);
        if (!(regs[2] & (1u << 27)) || !(regs[2] & (1u << 29))) return false;
        auto xcr0 = static_cast<unsigned long long>(_xgetbv(0));
        return (xcr0 & 0x6u) == 0x6u;
    }();
    return ok;
#else
    return false;
#endif
}

// Number of leading elements to verify against the scalar path per call.
// Debug-only cost, bounded so per-band invocations don't add up.
inline constexpr size_t kSpotCheckCount = 256u;

} // namespace detail

//==============================================================================
// sRGB decode
//==============================================================================

// Exact sRGB EOTF applied to all 256 possible 8-bit inputs, precomputed once.
// A lookup is bit-identical to running the formula per pixel (same float ops,
// each byte maps to one table slot) and removes a std::pow per channel.
[[nodiscard]] inline const std::array<float, 256>& srgbDecodeLut() noexcept {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> t{};
        for (uint32_t i = 0; i < 256u; i++) {
            float c = static_cast<float>(i) / 255.0f;
            t[i] = (c <= 0.04045f) ? c / 12.92f
                                   : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
        return t;
    }();
    return lut;
}

//==============================================================================
// Bulk float <-> half conversion
//==============================================================================
// F16C path converts 8 components at a time with round-to-nearest-even, which
// is half_float's default rounding mode, so results match static_cast<half>.

inline void f16ToF32(const luisa::half* src, float* dst, size_t n) noexcept {
#if defined(NT_MSVC_X86) || defined(__F16C__)
    if (detail::f16c_available()) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            __m256 f = _mm256_cvtph_ps(h);
            _mm256_storeu_ps(dst + i, f);
        }
        for (; i < n; i++) dst[i] = static_cast<float>(src[i]);
#ifndef NDEBUG
        const size_t check = n < detail::kSpotCheckCount ? n : detail::kSpotCheckCount;
        for (size_t k = 0; k < check; k++) {
            CI_ASSERT(dst[k] == static_cast<float>(src[k]));
        }
#endif
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<float>(src[i]);
}

inline void f32ToF16(const float* src, luisa::half* dst, size_t n) noexcept {
#if defined(NT_MSVC_X86) || defined(__F16C__)
    if (detail::f16c_available()) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 f = _mm256_loadu_ps(src + i);
            __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
        }
        for (; i < n; i++) dst[i] = static_cast<luisa::half>(src[i]);
#ifndef NDEBUG
        const size_t check = n < detail::kSpotCheckCount ? n : detail::kSpotCheckCount;
        for (size_t k = 0; k < check; k++) {
            CI_ASSERT(dst[k] == static_cast<luisa::half>(src[k]));
        }
#endif
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<luisa::half>(src[i]);
}

//==============================================================================
// 2x2 box-filter downsample (exact halving)
//==============================================================================
// dst[y][x] = 0.25 * (((s00 + s01) + s10) + s11), matching the scalar
// expression's association order so SSE2 and scalar paths are bit-identical.
// src must hold 2*rows rows of width sw; dst holds rows rows of width dw.

inline void boxDownsample2x2(const luisa::float4* src, uint sw,
                             luisa::float4* dst, uint dw, uint rows) noexcept {
#if defined(NT_SSE2)
    const __m128 quarter = _mm_set1_ps(0.25f);
#endif
    for (uint y = 0; y < rows; y++) {
        const luisa::float4* r0 = src + 2u * static_cast<size_t>(y) * sw;
        const luisa::float4* r1 = r0 + sw;
        luisa::float4* out = dst + static_cast<size_t>(y) * dw;
        for (uint x = 0; x < dw; x++) {
            uint x0 = 2u * x;
#if defined(NT_SSE2)
            __m128 s = _mm_add_ps(
                _mm_add_ps(
                    _mm_add_ps(_mm_loadu_ps(&r0[x0].x), _mm_loadu_ps(&r0[x0 + 1u].x)),
                    _mm_loadu_ps(&r1[x0].x)),
                _mm_loadu_ps(&r1[x0 + 1u].x));
            _mm_storeu_ps(&out[x].x, _mm_mul_ps(s, quarter));
#else
            out[x] = 0.25f * (((r0[x0] + r0[x0 + 1u]) + r1[x0]) + r1[x0 + 1u]);
#endif
        }
    }
#ifndef NDEBUG
    const uint checkRows = rows < 4u ? rows : 4u;
    for (uint y = 0; y < checkRows; y++) {
        const luisa::float4* r0 = src + 2u * static_cast<size_t>(y) * sw;
        const luisa::float4* r1 = r0 + sw;
        for (uint x = 0; x < dw; x++) {
            uint x0 = 2u * x;
            auto s = 0.25f * (((r0[x0] + r0[x0 + 1u]) + r1[x0]) + r1[x0 + 1u]);
            CI_ASSERT(dst[static_cast<size_t>(y) * dw + x].x == s.x &&
                      dst[static_cast<size_t>(y) * dw + x].y == s.y &&
                      dst[static_cast<size_t>(y) * dw + x].z == s.z &&
                      dst[static_cast<size_t>(y) * dw + x].w == s.w);
        }
    }
#endif
}

//==============================================================================
// UNORM8 quantize
//==============================================================================
// Per texel: byte = uint8(min(255, max(0, v*255 + 0.5))). The SSE2 path uses
// the truncating cvttps_epi32 after +0.5 — same round-half-up semantics as
// the scalar static_cast. dst receives RGBA bytes per texel.

inline void quantizeUnorm8(const luisa::float4* src, uint8_t* dst, size_t n) noexcept {
#if defined(NT_SSE2)
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128 half_ = _mm_set1_ps(0.5f);
    const __m128 hi = _mm_set1_ps(255.0f);
    const __m128 lo = _mm_setzero_ps();
    for (size_t i = 0; i < n; i++) {
        __m128 v = _mm_loadu_ps(&src[i].x);
        v = _mm_add_ps(_mm_mul_ps(v, scale), half_);
        v = _mm_max_ps(_mm_min_ps(v, hi), lo);
        __m128i i32 = _mm_cvttps_epi32(v);
        __m128i u16 = _mm_packs_epi32(i32, i32);
        __m128i u8 = _mm_packus_epi16(u16, u16);
        *reinterpret_cast<uint32_t*>(dst + i * 4u) =
            static_cast<uint32_t>(_mm_cvtsi128_si32(u8));
    }
#else
    for (size_t i = 0; i < n; i++) {
        auto q = [](float f) {
            return static_cast<uint8_t>(std::min(255.f, std::max(0.f, f * 255.f + 0.5f)));
        };
        dst[i * 4u + 0u] = q(src[i].x);
        dst[i * 4u + 1u] = q(src[i].y);
        dst[i * 4u + 2u] = q(src[i].z);
        dst[i * 4u + 3u] = q(src[i].w);
    }
#endif
#ifndef NDEBUG
    const size_t check = n < detail::kSpotCheckCount ? n : detail::kSpotCheckCount;
    for (size_t k = 0; k < check; k++) {
        auto q = [](float f) {
            return static_cast<uint8_t>(std::min(255.f, std::max(0.f, f * 255.f + 0.5f)));
        };
        CI_ASSERT(dst[k * 4u + 0u] == q(src[k].x) &&
                  dst[k * 4u + 1u] == q(src[k].y) &&
                  dst[k * 4u + 2u] == q(src[k].z) &&
                  dst[k * 4u + 3u] == q(src[k].w));
    }
#endif
}

} // namespace newtype::util

#if defined(NT_MSVC_X86)
  #undef NT_MSVC_X86
#endif
#if defined(NT_SSE2)
  #undef NT_SSE2
#endif
