#pragma once

// Bit-pack helpers for reservoir packing (#6 Stage A).
//
// Self-contained DSL utilities for encoding/decoding unit normals via
// octahedral projection (oct32: 16-bit per axis in a single uint32) and
// HDR RGB radiance via LogLuv (16-bit log luminance + 8-bit chroma u + v).
//
// All functions are inline and only callable inside device.compile()
// contexts (they use DSL primitives: cast, ite, clamp, log2, exp2).
//
// Reference: Cigolle et al. 2014, "Survey of Efficient Representations
// for Independent Unit Vectors" for octahedral encoding. LogLuv uses
// the standard Ward formulation.

#include <luisa/luisa-compute.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Internal: octahedral transform between unit sphere and [-1,1]^2 square.
// Inlined (rather than reusing Shading.h's oct_encode/decode) to keep this
// header lightweight — Shading.h pulls in BSDF/MaterialPool/SurfaceResolver
// chains that would propagate to every consumer of these helpers.
//==============================================================================

[[nodiscard]] inline Float2 oct_encode_internal(Float3 n) noexcept {
    Float denom = abs(n.x) + abs(n.y) + abs(n.z);
    Float2 enc = make_float2(n.x, n.y) / max(denom, 1e-10f);
    enc = ite(n.z < 0.0f,
        (make_float2(1.0f) - abs(enc)) * make_float2(
            ite(n.x >= 0.0f, 1.0f, -1.0f),
            ite(n.y >= 0.0f, 1.0f, -1.0f)),
        enc);
    return enc;
}

[[nodiscard]] inline Float3 oct_decode_internal(Float2 enc) noexcept {
    Float3 n = make_float3(enc.x, enc.y, 1.0f - abs(enc.x) - abs(enc.y));
    $if(n.z < 0.0f) {
        n = make_float3(
            (1.0f - abs(enc.y)) * ite(enc.x >= 0.0f, 1.0f, -1.0f),
            (1.0f - abs(enc.x)) * ite(enc.y >= 0.0f, 1.0f, -1.0f),
            n.z);
    };
    return normalize(n);
}

//==============================================================================
// Octahedral unit-vector encoding packed into a single uint32.
// 16-bit signed per axis. Precision: ~0.0003 RMS for unit normals.
//==============================================================================

[[nodiscard]] inline UInt oct32_encode(Float3 n) noexcept {
    Float2 enc = oct_encode_internal(n);                       // [-1,1]^2
    UInt   xq  = cast<UInt>(cast<Int>(enc.x * 32767.0f)) & 0xFFFFu;
    UInt   yq  = cast<UInt>(cast<Int>(enc.y * 32767.0f)) & 0xFFFFu;
    return xq | (yq << 16u);
}

[[nodiscard]] inline Float3 oct32_decode(UInt packed) noexcept {
    UInt   xq = packed & 0xFFFFu;
    UInt   yq = (packed >> 16u) & 0xFFFFu;
    // Sign-extend each 16-bit slot back to signed int before normalizing.
    Int    xs = cast<Int>(xq) - ite(xq >= 32768u, 65536, 0);
    Int    ys = cast<Int>(yq) - ite(yq >= 32768u, 65536, 0);
    Float2 enc = make_float2(cast<Float>(xs), cast<Float>(ys)) * (1.0f / 32767.0f);
    return oct_decode_internal(enc);
}

//==============================================================================
// LogLuv HDR RGB encoding packed into a single uint32.
// Layout: 8-bit v chroma (bits 0-7) + 8-bit u chroma (8-15) + 16-bit log luminance (16-31).
// Covers luminance 2^-64..2^+191; loses some chroma detail (8-bit u'v').
// The codec is idempotent: encode(decode(p)) == p for any packed p, so
// repeated decode→re-encode roundtrips (reservoir reuse) cannot drift.
// Regression-tested by tests/test_logluv_packing.py — keep in sync.
//==============================================================================

[[nodiscard]] inline UInt logluv_encode(Float3 rgb) noexcept {
    // Sanitize: NaN/Inf propagate through the matrix/log2 chain and the
    // truncating cast of a NaN is undefined — encode black instead.
    // (dsl-qualified: unqualified isnan/isinf collide with UCRT overloads)
    rgb = ite(any(luisa::compute::dsl::isnan(rgb)) |
                  any(luisa::compute::dsl::isinf(rgb)),
              make_float3(0.f), rgb);

    // 1. Convert to XYZ
    auto X = rgb.x * 0.4124f + rgb.y * 0.3576f + rgb.z * 0.1805f;
    auto Y = rgb.x * 0.2126f + rgb.y * 0.7152f + rgb.z * 0.0722f;
    auto Z = rgb.x * 0.0193f + rgb.y * 0.1192f + rgb.z * 0.9505f;

    // 2. CIE 1976 u'v' chromaticity
    auto denom = max(1e-10f, X + 15.0f * Y + 3.0f * Z);
    auto u_prime = (4.0f * X) / denom;
    auto v_prime = (9.0f * Y) / denom;
    
    // Quantize with round-to-nearest (+0.5 before the truncating cast).
    // Plain truncation is not idempotent: a decoded value re-multiplies just
    // below its quantization boundary in float32 (k/410*410 == k - eps), so
    // every re-encode walks the code down 1 LSB and cached radiance ratchets
    // off-neutral across temporal/spatial reservoir reuse.
    auto Ue = cast<UInt>(clamp(u_prime * 410.0f + 0.5f, 0.0f, 255.0f));
    auto Ve = cast<UInt>(clamp(v_prime * 410.0f + 0.5f, 0.0f, 255.0f));

    // 3. Log2 luminance
    auto Le = cast<UInt>(clamp(256.0f * (log2(max(Y, 1e-6f)) + 64.0f) + 0.5f, 0.0f, 65535.0f));
    
    // 4. Pack into uint
    return ((Le & 0xFFFFu) << 16) | ((Ue & 0xFFu) << 8) | (Ve & 0xFFu);
}

[[nodiscard]] inline Float3 logluv_decode(UInt packed) noexcept {
    // 1. Unpack the components
    auto Le = (packed >> 16) & 0xFFFFu;
    auto Ue = (packed >> 8) & 0xFFu;
    auto Ve = packed & 0xFFu;

    auto v_prime = cast<Float>(Ve) / 410.0f;
    Float3 rgb = def(make_float3(0.f));
    // early exist on black
    $if(v_prime > 1e-6f) {
        // 2. Decode Luminance (Y)
        auto Y = pow(2.0f, cast<Float>(Le) / 256.0f - 64.0f);

        // 3. Decode Chromaticity (u', v')
        auto u_prime = cast<Float>(Ue) / 410.0f;

        // 4. u'v'Y → XYZ
        auto denom = 4.0f * v_prime;
        auto X = Y * 9.0f * u_prime / denom;
        auto Z = Y * (12.0f - 3.0f * u_prime - 20.0f * v_prime) / denom;

        // 5. Convert XYZ back to RGB (Inverse Matrix)
        rgb.x = 3.2406f * X - 1.5372f * Y - 0.4986f * Z;
        rgb.y = -0.9689f * X + 1.8758f * Y + 0.0415f * Z;
        rgb.z = 0.0557f * X - 0.2040f * Y + 1.0570f * Z;

        // Clamp negative channels (out-of-gamut chroma), then rescale to
        // restore the target luminance Y. Clamping alone lowers/raises the
        // reconstructed luminance for saturated colors, so re-encoding a
        // decoded value would ratchet Le every roundtrip.
        rgb = max(0.f, rgb);
        auto lum = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
        rgb = rgb * (Y / max(lum, 1e-20f));
    };
    
    return rgb;
}

} // namespace newtype::render
