#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/render/Packing.h"

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// ReSTIR GI Reservoir
//==============================================================================

/**
 * @brief Reservoir for ReSTIR GI (1-bounce indirect illumination)
 *
 * Stores a surface sample point (x2) from a BRDF bounce with cached radiance.
 * 32 bytes — matches RTXDI's GIReservoir size for bandwidth efficiency.
 *
 * Layout:
 *   px, py, pz      (12B)  — x2 world-space position (no good encoding)
 *   packed_normal   (4B)   — oct32-encoded x2 shading normal (16b/axis signed)
 *   packed_radiance (4B)   — LogLuv-encoded cached radiance at x2
 *                            (16b log luminance + 8b chroma u + 8b chroma v)
 *   weight_sum      (4B)   — RIS weight sum
 *   target_pdf      (4B)   — p_hat of selected sample
 *   packed_meta     (4B)   — M(16) | age(6) | vis_age(3) | visibility(1) | roughness_q(6)
 */
struct alignas(8) GIReservoir {
    float px, py, pz;
    uint  packed_normal;

    uint  packed_radiance;
    //float rx, ry, rz;
    float weight_sum;
    float target_pdf;
    uint  packed_meta;
};
static_assert(sizeof(GIReservoir) == 32u);

} // namespace newtype::render

//==============================================================================
// CPU-side Helpers (not in LUISA_STRUCT block)
//==============================================================================

namespace newtype::render {

[[nodiscard]] inline luisa::float3 gi_pos(const GIReservoir& r) noexcept {
    return {r.px, r.py, r.pz};
}

} // namespace newtype::render

//==============================================================================
// DSL Struct Registration
//==============================================================================

LUISA_STRUCT(newtype::render::GIReservoir,
    px, py, pz, packed_normal, packed_radiance, weight_sum, target_pdf, packed_meta) {
    //px, py, pz, packed_normal, rx,ry,rz, weight_sum, target_pdf, packed_meta) {

    //-----------------------------------------------------------------------
    // Read accessors (const)
    //-----------------------------------------------------------------------

    /// Number of candidates considered (0..65535).
    [[nodiscard]] auto M() const noexcept {
        return packed_meta & 0xFFFFu;
    }

    /// Frames survived (0..63).
    [[nodiscard]] auto age() const noexcept {
        return (packed_meta >> 16u) & 0x3Fu;
    }

    /// Frames since visibility was last confirmed (0..7).
    [[nodiscard]] auto vis_age() const noexcept {
        return (packed_meta >> 22u) & 0x7u;
    }

    /// 0 = untested/blocked (need shadow ray), 1 = confirmed visible.
    [[nodiscard]] auto visibility() const noexcept {
        return (packed_meta >> 25u) & 1u;
    }

    /// Receiver roughness at sample-selection time (quantized to 6 bits / 63 levels).
    [[nodiscard]] auto roughness() const noexcept {
        return (packed_meta >> 26u) & 0x3Fu;
    }

    /// Valid if at least one candidate was considered.
    [[nodiscard]] auto is_valid() const noexcept {
        return M() > 0u;
    }

    /// x2 world-space position.
    [[nodiscard]] auto pos() const noexcept {
        return luisa::compute::make_float3(px, py, pz);
    }

    /// x2 shading normal — decodes from oct32.
    [[nodiscard]] auto nrm() const noexcept {
        return newtype::render::oct32_decode(packed_normal);
    }

    /// Cached radiance at x2 — decodes from LogLuv.
    [[nodiscard]] auto rad() const noexcept {
        return newtype::render::logluv_decode(packed_radiance);
        //return make_float3(rx,ry,rz);
    }

    /// Reservoir weight (post-finalize RTXDI form). weight_sum is stored after
    /// finalize divides by (M * target_pdf) at write time, so weight() is a direct
    /// multiplier on radiance at shade. Cap at 20 to suppress fireflies.
    [[nodiscard]] auto weight() const noexcept {
        auto W = ite(M() > 0u, weight_sum, 0.0f);
        return luisa::compute::min(W, 20.0f);
    }

    //-----------------------------------------------------------------------
    // Write accessors (read-modify-write on packed_meta / packed_normal / packed_radiance)
    //-----------------------------------------------------------------------

    void set_M(::luisa::compute::UInt v) noexcept {
        packed_meta = (packed_meta & ~0xFFFFu) | (v & 0xFFFFu);
    }

    void set_age(::luisa::compute::UInt v) noexcept {
        packed_meta = (packed_meta & ~(0x3Fu << 16u)) | ((v & 0x3Fu) << 16u);
    }

    void set_vis_age(::luisa::compute::UInt v) noexcept {
        packed_meta = (packed_meta & ~(0x7u << 22u)) | ((v & 0x7u) << 22u);
    }

    void set_visibility(::luisa::compute::UInt v) noexcept {
        packed_meta = (packed_meta & ~(1u << 25u)) | ((v & 1u) << 25u);
    }

    void set_roughness(::luisa::compute::UInt q) noexcept {
        packed_meta = (packed_meta & ~(0x3Fu << 26u)) | ((q & 0x3Fu) << 26u);
    }

    void set_nrm(::luisa::compute::Float3 n) noexcept {
        packed_normal = newtype::render::oct32_encode(n);
    }

    void set_rad(::luisa::compute::Float3 rgb) noexcept {
        packed_radiance = newtype::render::logluv_encode(rgb);
        //rx = rgb.x;
        //ry = rgb.y;
        //rz = rgb.z;
    }
};

//==============================================================================
// DSL-side Helpers (free functions, callable inside LuisaCompute kernels)
//==============================================================================

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

// Mirrors RTXDI_FinalizeGIResampling (GI/Reservoir.hlsli:237-243):
//   reservoir.weightSum = (denom == 0) ? 0 : (weightSum * num) / denom;
// Call at write time to convert streaming-form w_acc back to post-finalize.
inline void finalize_gi(Var<GIReservoir>& r, Float num, Float denom) noexcept {
    r.weight_sum = ite(denom <= 0.0f, 0.0f,
                       r.weight_sum * num / max(denom, 1e-10f));
}

} // namespace newtype::render
