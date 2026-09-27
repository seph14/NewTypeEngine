#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

/**
 * @brief Result of a single GI bounce (BRDF sample -> trace -> reconstruct -> NEE)
 *
 * Decomposed floats to avoid alignment padding in LUISA_STRUCT.
 * 100 bytes total.
 */
struct GIBounceResult {
    // Hit position (12 bytes)
    float px, py, pz;
    // Hit shading normal (12 bytes)
    float nx, ny, nz;
    // Hit tangent (12 bytes)
    float tx, ty, tz;
    float tw;              // tangent handedness
    // NEE radiance at hit point (12 bytes)
    float rad_x, rad_y, rad_z;
    // BRDF sample direction (needed for next bounce wo = -wi)
    float wi_x, wi_y, wi_z;
    // Metadata
    uint  hit_inst;        // TLAS slot (both mesh and proc) — used for self-rejection
    uint  hit_inst_data_y; // material_layers (per-branch source: mesh=inst_data.y, proc=proc.material_layers)
    uint  hit_prim;        // mesh: triangle idx; proc: AABB idx (= inst_id for resolver)
    float hit_bary_u;      // mesh: hit.bary.x; proc: hit.local_bary.x
    float hit_bary_v;      // mesh: hit.bary.y; proc: hit.local_bary.y
    float hit_t;           // committed ray t (for next bounce offset)
    uint  seed;            // consumed RNG state
    // SHARC query tri-state (plan §7 Phase 2):
    //   0 = no query attempted (query off / gates rejected / invalid hit)
    //   1 = cache HIT  — rad_* is the cached radiance; the caller skips the
    //       x2->x3 trace (the value already carries the converged tail)
    //   2 = query attempted, MISSED — NEE ran; counters tag it for the
    //       SHARC panel hit-rate readout
    uint  cache_status;
    // Bit-packed: bit 0 = hit valid, bit 1 = is_procedural,
    // bits 2-31 = local_tri (proc only; 0 for mesh). Always test with
    // `(valid & 1u)`, never `valid == 1u` — proc hits set valid=3.
    uint  valid;
};
static_assert(sizeof(GIBounceResult) == 100u);

} // namespace newtype::render

LUISA_STRUCT(newtype::render::GIBounceResult,
    px, py, pz, nx, ny, nz, tx, ty, tz, tw,
    rad_x, rad_y, rad_z, wi_x, wi_y, wi_z,
    hit_inst, hit_inst_data_y, hit_prim,
    hit_bary_u, hit_bary_v, hit_t, seed, cache_status, valid) {

    [[nodiscard]] auto pos() const noexcept {
        return luisa::compute::make_float3(px, py, pz);
    }
    [[nodiscard]] auto nrm() const noexcept {
        return luisa::compute::make_float3(nx, ny, nz);
    }
    [[nodiscard]] auto tangent() const noexcept {
        return luisa::compute::make_float3(tx, ty, tz);
    }
    [[nodiscard]] auto rad() const noexcept {
        return luisa::compute::make_float3(rad_x, rad_y, rad_z);
    }
    [[nodiscard]] auto wi() const noexcept {
        return luisa::compute::make_float3(wi_x, wi_y, wi_z);
    }
    [[nodiscard]] auto is_procedural() const noexcept {
        return ((valid >> 1u) & 1u) != 0u;
    }
    [[nodiscard]] auto local_tri() const noexcept {
        return valid >> 2u;
    }
};
