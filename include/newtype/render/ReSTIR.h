#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// ReSTIR DI Reservoir
//==============================================================================

/**
 * @brief Reservoir for ReSTIR DI (Direct Illumination)
 *
 * Stores a single selected light sample along with the accumulated
 * weight information needed for unbiased RIS (Resampled Importance Sampling).
 *
 * 24 bytes — light sample (20B) + packed metadata (4B).
 * Metadata bit layout in packed_meta (LSB → MSB):
 *   bits 0-15   : M           (unsigned, 0..65535)
 *   bits 16-19  : vis_age     (unsigned, 0..15)
 *   bit  20     : visibility  (0 or 1)
 *   bits 21-25  : spatial_dx  (signed 5-bit two's complement, -16..15)
 *   bits 26-30  : spatial_dy  (signed 5-bit two's complement, -16..15)
 *   bit  31     : spare
 */
struct alignas(8) Reservoir {
    uint  light_idx;       // Selected emissive triangle index (~0u if invalid)
    float w_sum;           // Sum of RIS weights
    float target_pdf;      // p_hat of selected sample
    float light_bary_u;    // Barycentric u on light triangle (for shade pass)
    float light_bary_v;    // Barycentric v on light triangle (for shade pass)
    uint  packed_meta;     // M / vis_age / visibility / spatial_dist_x / y
};
static_assert(sizeof(Reservoir) == 24u);

//==============================================================================
// Presampled Light Candidate (for light presampling tiles)
//==============================================================================

struct PresampledCandidate {
    uint  light_idx;       // Emissive triangle index (~0u if invalid, kEnvLightSentinel for env)
    float bary_u;          // Barycentric u on light triangle (or envmap u)
    float bary_v;          // Barycentric v on light triangle (or envmap v)
    float inv_source_pdf;  // 1.0 / source_pdf
};
static_assert(sizeof(PresampledCandidate) == 16u);

} // namespace newtype::render

//==============================================================================
// DSL Struct Registration
//==============================================================================

LUISA_STRUCT(newtype::render::Reservoir, light_idx, w_sum, target_pdf, light_bary_u, light_bary_v, packed_meta) {

    //-----------------------------------------------------------------------
    // Read accessors (const)
    //-----------------------------------------------------------------------

    /// Number of candidates considered.
    [[nodiscard]] auto M() const noexcept {
        return packed_meta & 0xFFFFu;
    }

    /// Frames since visibility was last confirmed (0..15).
    [[nodiscard]] auto vis_age() const noexcept {
        return (packed_meta >> 16u) & 0xFu;
    }

    /// 0 = untested/invisible (need shadow ray), 1 = confirmed visible.
    [[nodiscard]] auto visibility() const noexcept {
        return (packed_meta >> 20u) & 1u;
    }

    /// Accumulated screen-space X displacement since last visibility confirmation.
    /// Signed 5-bit two's complement, range -16..15.
    [[nodiscard]] auto spatial_dist_x() const noexcept {
        auto v = (packed_meta >> 21u) & 0x1Fu;
        return cast<int>(v) - ite(v >= 16u, 32, 0);
    }

    /// Accumulated screen-space Y displacement (signed 5-bit, range -16..15).
    [[nodiscard]] auto spatial_dist_y() const noexcept {
        auto v = (packed_meta >> 26u) & 0x1Fu;
        return cast<int>(v) - ite(v >= 16u, 32, 0);
    }

    //-----------------------------------------------------------------------
    // Write accessors (read-modify-write on packed_meta)
    //-----------------------------------------------------------------------

    void set_M(auto m) noexcept {
        packed_meta = (packed_meta & ~0xFFFFu) | (m & 0xFFFFu);
    }

    // vis_age is 4 bits (0-15); values > 15 wrap via this mask — keep visMaxAge <= 15
    void set_vis_age(auto v) noexcept {
        packed_meta = (packed_meta & ~(0xFu << 16u)) | ((v & 0xFu) << 16u);
    }

    void set_visibility(auto v) noexcept {
        packed_meta = (packed_meta & ~(1u << 20u)) | ((v & 1u) << 20u);
    }

    void set_spatial_dist_x(::luisa::compute::Int v) noexcept {
        // Two's complement 5-bit: -16..15 maps to 16..31 / 0..15.
        // cast<int>→cast<uint> wrap-around gives the correct bit pattern, then mask.
        auto clamped = clamp(v, -16, 15);
        auto packed5 = cast<::luisa::compute::UInt>(clamped) & 0x1Fu;
        packed_meta = (packed_meta & ~(0x1Fu << 21u)) | (packed5 << 21u);
    }

    void set_spatial_dist_y(::luisa::compute::Int v) noexcept {
        auto clamped = clamp(v, -16, 15);
        auto packed5 = cast<::luisa::compute::UInt>(clamped) & 0x1Fu;
        packed_meta = (packed_meta & ~(0x1Fu << 26u)) | (packed5 << 26u);
    }

    //-----------------------------------------------------------------------
    // Derived helpers
    //-----------------------------------------------------------------------

    /// Reservoir weight W = w_sum / (M * target_pdf)
    [[nodiscard]] auto weight() const noexcept {
        auto Mv = M();
        return ite(Mv > 0u & target_pdf > 1e-8f,
                   w_sum / (cast<float>(Mv) * target_pdf),
                   0.0f);
    }

    /// Valid if at least one candidate was considered
    [[nodiscard]] auto is_valid() const noexcept {
        return M() > 0u;
    }
};

LUISA_STRUCT(newtype::render::PresampledCandidate, light_idx, bary_u, bary_v, inv_source_pdf) {

    [[nodiscard]] auto is_valid() const noexcept {
        return light_idx != ~0u;
    }
};
