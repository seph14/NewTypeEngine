#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <newtype/scene/VoxelGrid.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

/// DDA march with expanded params (for raster shaders that can't use binding groups).
[[nodiscard]] inline Float dda_march_flat(
    const BufferVar<luisa::half>& occupancy,
    const BufferVar<luisa::uint>& summary,
    const BufferVar<luisa::uint>& occupied_count,
    const Var<scene::VoxelGridParams>& params,
    const Float3& origin,
    const Float3& direction,
    const Float& tmax,
    Float density_scale = 1.0f) noexcept {

    Float3  bounds_min   = params.bounds_min;
    Float3  bounds_max   = params.bounds_max;
    UInt3   res          = params.resolution;
    Float   cell_size    = params.cell_size;
    UInt3   summary_res  = params.summary_resolution;
    constexpr uint block_size = scene::VoxelGrid::kBlockSize;

    Float attenuation = 1.0f;

    $if(Expr{ occupied_count.read(0u) > 0u }) {
        Float3 grid_extent = make_float3(
            cast<Float>(res.x),
            cast<Float>(res.y),
            cast<Float>(res.z)) * cell_size;
        Float3 grid_max = bounds_min + grid_extent;

        Float3 inv_dir = 1.0f / max(abs(direction), make_float3(1e-10f));
        Float3 t0 = (bounds_min - origin) * inv_dir;
        Float3 t1 = (grid_max - origin) * inv_dir;
        Float ray_tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
        Float ray_tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
        ray_tmin = max(ray_tmin, 0.0f);
        ray_tmax = min(ray_tmax, tmax);

        $if(Expr{ ray_tmin < ray_tmax }) {

        Float3 entry = origin + direction * ray_tmin;

        Int3 cell = make_int3(
            cast<int>(clamp((entry.x - bounds_min.x) / cell_size, 0.0f, cast<Float>(res.x - 1u))),
            cast<int>(clamp((entry.y - bounds_min.y) / cell_size, 0.0f, cast<Float>(res.y - 1u))),
            cast<int>(clamp((entry.z - bounds_min.z) / cell_size, 0.0f, cast<Float>(res.z - 1u)))
        );

        Int3 step = make_int3(
            ite(direction.x > 0.0f, 1, -1),
            ite(direction.y > 0.0f, 1, -1),
            ite(direction.z > 0.0f, 1, -1)
        );

        auto cellFlt = make_float3(cast<Float>(cell.x), cast<Float>(cell.y), cast<Float>(cell.z));
        Float3 next_boundary = make_float3(
            ite(direction > 0.0f,
                (cellFlt + 1.0f) * cell_size + bounds_min,
                cellFlt * cell_size + bounds_min)
        );

        Float3 t_max_local = abs((next_boundary - entry) * inv_dir);
        Float3 t_delta = cell_size * inv_dir;
        Float march_tmax = ray_tmax - ray_tmin;

        Bool starts_inside = ray_tmin < 0.001f;
        UInt steps_to_skip = ite(starts_inside, 1u, 0u);

        UInt max_steps = (res.x + res.y + res.z) * 2u;

        $for(i, max_steps) {
            $if(any(make_uint3(cell) > res) | any(cell < 0)) {
                $break;
            };

            UInt3 block_coord = make_uint3(
                cast<uint>(cell.x) / block_size,
                cast<uint>(cell.y) / block_size,
                cast<uint>(cell.z) / block_size
            );
            UInt summary_flat = block_coord.x
                + block_coord.y * summary_res.x
                + block_coord.z * summary_res.x * summary_res.y;
            UInt block_flag = summary.read(summary_flat);

            $if(block_flag == 0u) {
                Float3 block_lo_f = make_float3(
                    cast<Float>(block_coord.x * block_size),
                    cast<Float>(block_coord.y * block_size),
                    cast<Float>(block_coord.z * block_size)
                ) * cell_size + bounds_min;
                Float3 block_hi_f = make_float3(
                    cast<Float>((block_coord.x + 1u) * block_size),
                    cast<Float>((block_coord.y + 1u) * block_size),
                    cast<Float>((block_coord.z + 1u) * block_size)
                ) * cell_size + bounds_min;
                Float3 block_exit_t = abs((ite(direction > 0.0f, block_hi_f, block_lo_f) - entry) * inv_dir);

                Float t_block_exit = min(min(block_exit_t.x, block_exit_t.y), block_exit_t.z);
                $if(t_block_exit > march_tmax) { $break; };

                $if(block_exit_t.x <= block_exit_t.y & block_exit_t.x <= block_exit_t.z) {
                    UInt cells_in_block = cast<uint>(cell.x) % block_size;
                    UInt remaining = ite(direction.x > 0.0f, block_size - 1u - cells_in_block, cells_in_block);
                    Float advance = cast<Float>(remaining + 1u) * t_delta.x;
                    t_max_local.x += advance;
                    cell.x += step.x * cast<int>(remaining + 1u);
                }
                $elif(block_exit_t.y <= block_exit_t.z) {
                    UInt cells_in_block = cast<uint>(cell.y) % block_size;
                    UInt remaining = ite(direction.y > 0.0f, block_size - 1u - cells_in_block, cells_in_block);
                    Float advance = cast<Float>(remaining + 1u) * t_delta.y;
                    t_max_local.y += advance;
                    cell.y += step.y * cast<int>(remaining + 1u);
                }
                $else{
                    UInt cells_in_block = cast<uint>(cell.z) % block_size;
                    UInt remaining = ite(direction.z > 0.0f, block_size - 1u - cells_in_block, cells_in_block);
                    Float advance = cast<Float>(remaining + 1u) * t_delta.z;
                    t_max_local.z += advance;
                    cell.z += step.z * cast<int>(remaining + 1u);
                };
            }
            $else{
                $if(i >= steps_to_skip) {
                    UInt flat = cast<uint>(cell.x)
                              + cast<uint>(cell.y) * res.x
                              + cast<uint>(cell.z) * res.x * res.y;
                    Float density = cast<Float>(occupancy.read(flat));
                    $if(density > 0.0f) {
                        attenuation = attenuation * exp(-density * density_scale);
                        $if(attenuation < 0.001f) { $break; };
                    };
                };

                $if(t_max_local.x < t_max_local.y & t_max_local.x < t_max_local.z) {
                    $if(t_max_local.x > march_tmax) { $break; };
                    cell.x += step.x;
                    t_max_local.x += t_delta.x;
                }
                $elif(t_max_local.y < t_max_local.z) {
                    $if(t_max_local.y > march_tmax) { $break; };
                    cell.y += step.y;
                    t_max_local.y += t_delta.y;
                }
                $else {
                    $if(t_max_local.z > march_tmax) { $break; };
                    cell.z += step.z;
                    t_max_local.z += t_delta.z;
                };
            };
        };
    };
    };

    return attenuation;
}

/// DDA ray march through a voxel density grid with hierarchical block-level skip.
/// Returns attenuation: 1.0 = fully visible, 0.0 = fully occluded.
///
/// Takes a VoxelGrid::Resources binding group. Grid params are scalar constants
/// in the binding group (not buffer reads). block_size is compile-time (8).
/// density_scale controls how quickly shadow attenuates per cell.
/// Old linear model: 1 cell at density=1 -> full occlusion.
/// Exponential: ~3 cells at density=1 -> 0.05 attenuation with scale=1.0.
[[nodiscard]] inline Float dda_march(
    const Var<scene::VoxelGrid::Resources>& voxel,
    const Float3& origin,
    const Float3& direction,
    const Float& tmax,
    Float density_scale = 1.0f) noexcept {
    return dda_march_flat(
        voxel.occupancy, voxel.summary, voxel.occupied_count,
        Var<scene::VoxelGridParams>{
            voxel.bounds_min, voxel.bounds_max,
            voxel.resolution, voxel.cell_size,
            voxel.summary_resolution},
        origin, direction, tmax, density_scale);
}

} // namespace newtype::render
