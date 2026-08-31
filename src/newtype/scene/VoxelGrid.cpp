#include "newtype/scene/VoxelGrid.h"
#include <luisa/dsl/sugar.h>

namespace newtype::scene {

using namespace luisa;
using namespace luisa::compute;

VoxelGrid::VoxelGrid(Device& device) noexcept
    : _device(device) {}

void VoxelGrid::create(uint base_resolution, float3 bounds_min, float3 bounds_max) {
    float3 extent = bounds_max - bounds_min;
    // Uniform cell size from largest axis so no axis gets fewer than base_resolution cells
    float max_extent = max(max(extent.x, extent.y), extent.z);
    float cell_size = max_extent / static_cast<float>(base_resolution);

    // Per-axis resolution — each axis gets extent/cell_size cells
    uint3 res = make_uint3(
        max(1u, static_cast<uint>(ceil(extent.x / cell_size))),
        max(1u, static_cast<uint>(ceil(extent.y / cell_size))),
        max(1u, static_cast<uint>(ceil(extent.z / cell_size))));

    _resolution = res;
    _totalVoxels = res.x * res.y * res.z;

    _params.bounds_min = bounds_min;
    _params.bounds_max = bounds_max;
    _params.resolution = res;
    _params.cell_size = cell_size;

    // Summary grid: one coarse cell per kBlockSize³ fine cells
    _summaryRes = make_uint3(
        (res.x + kBlockSize - 1u) / kBlockSize,
        (res.y + kBlockSize - 1u) / kBlockSize,
        (res.z + kBlockSize - 1u) / kBlockSize);
    _totalSummary = _summaryRes.x * _summaryRes.y * _summaryRes.z;
    _params.summary_resolution = _summaryRes;

    // Allocate occupancy buffers (zeroed by clear() before first use)
    _occupancy     = _device.create_buffer<half>(_totalVoxels);
    _occupancyTemp = _device.create_buffer<half>(_totalVoxels);
    _summary       = _device.create_buffer<uint>(_totalSummary);
    _occupiedCount = _device.create_buffer<uint>(1u);

    _paramsBuffer = _device.create_buffer<VoxelGridParams>(1u);
    _paramsBuffer.view(0u, 1u).copy_from(&_params);

    _compileShaders();
    _created = true;
}

void VoxelGrid::_compileShaders() {
    _clearShader = _device.compile<1>(
        [&](BufferVar<half> occupancy, UInt total) noexcept {
        set_block_size(512u, 1u, 1u);
        set_name("voxel_clear");
        UInt idx = dispatch_id().x;
        $if(idx < total) {
            occupancy.write(idx, cast<half>(0.0f));
        };
    });

    // Voxelize: per-axis resolution with uniform cell_size
    _voxelizeShader = _device.compile<1>(
        [&](BufferVar<half> occupancy,
            BufferVar<float3> positions,
            UInt particle_count,
            Float3 bounds_min,
            Float3 bounds_max,
            UInt3 grid_res) noexcept {
        set_block_size(512u, 1u, 1u);
        set_name("voxel_splat");

        UInt idx = dispatch_id().x;
        $if(idx >= particle_count) { $return(); };

        Float3 pos = positions.read(idx);
        Float3 extent = bounds_max - bounds_min;

        // Per-axis cell coordinate
        Float3 cell_f = (pos - bounds_min) * make_float3(cast<Float>(grid_res.x),
                                                           cast<Float>(grid_res.y),
                                                           cast<Float>(grid_res.z)) / extent;
        Int3 cell = make_int3(cast<int>(cell_f.x),
                              cast<int>(cell_f.y),
                              cast<int>(cell_f.z));

        $if(cell.x >= 0 & cast<uint>(cell.x) < grid_res.x &
             cell.y >= 0 & cast<uint>(cell.y) < grid_res.y &
             cell.z >= 0 & cast<uint>(cell.z) < grid_res.z) {
            UInt flat = cast<uint>(cell.x)
                      + cast<uint>(cell.y) * grid_res.x
                      + cast<uint>(cell.z) * grid_res.x * grid_res.y;
            occupancy.write(flat, cast<half>(1.0f));
        };
    });

    _voxelizeF4Shader = _device.compile<1>(
        [&](BufferVar<half> occupancy,
            BufferVar<luisa::float4> positions,
            UInt particle_count,
            Float3 bounds_min,
            Float3 bounds_max,
            UInt3 grid_res,
            Float radius_scale) noexcept {
        set_block_size(512u, 1u, 1u);
        set_name("voxel_splat_f4");

        UInt idx = dispatch_id().x;
        $if(idx >= particle_count) { $return(); };

        Float4 particle = positions.read(idx);
        Float3 pos = particle.xyz();
        Float  radius = particle.w * radius_scale;
        Float3 extent = bounds_max - bounds_min;

        // Cell size (uniform across all axes)
        Float cell_size = max(max(extent.x, extent.y), extent.z)
                        / cast<Float>(max(max(grid_res.x, grid_res.y), grid_res.z));

        // Cell radius: how many cells the sphere spans in each direction
        Float cell_radius_f = max(radius / cell_size, 0.0f);
        Int cell_radius = cast<int>(ceil(cell_radius_f));

        // Center cell
        Float3 cell_f = (pos - bounds_min) * make_float3(cast<Float>(grid_res.x),
                                                           cast<Float>(grid_res.y),
                                                           cast<Float>(grid_res.z)) / extent;
        Int3 center = make_int3(cast<int>(cell_f.x),
                                cast<int>(cell_f.y),
                                cast<int>(cell_f.z));

        // Loop over AABB of cells overlapped by the sphere
        Int3 lo = max(center - cell_radius, make_int3(0));
        Int3 hi = min(center + cell_radius,
                      make_int3(cast<int>(grid_res.x) - 1,
                                cast<int>(grid_res.y) - 1,
                                cast<int>(grid_res.z) - 1));

        Float radius_sq = radius * radius;

        $for(dz, hi.z - lo.z + 1) {
            Int cz = lo.z + dz;
            $for(dy, hi.y - lo.y + 1) {
                Int cy = lo.y + dy;
                $for(dx, hi.x - lo.x + 1) {
                    Int cx = lo.x + dx;
                    // Cell center in world space
                    Float3 cell_center = bounds_min + cell_size * (make_float3(
                        cast<Float>(cx) + 0.5f,
                        cast<Float>(cy) + 0.5f,
                        cast<Float>(cz) + 0.5f));
                    Float dist_sq = length_squared(cell_center - pos);
                    $if(dist_sq <= radius_sq) {
                        UInt flat = cast<uint>(cx)
                                  + cast<uint>(cy) * grid_res.x
                                  + cast<uint>(cz) * grid_res.x * grid_res.y;
                        occupancy.write(flat, cast<half>(1.0f));
                    };
                };
            };
        };
    });

    // Dilation: 3x3x3 max-filter with shared memory tile (3D dispatch)
    // Each 8x8x8 block loads a 10x10x10 shared tile (1-cell border).
    // 512 threads cooperatively load 1000 cells (2 passes), then read from SM.
    _dilateShader = _device.compile<3>(
        [&](BufferVar<half> src, BufferVar<half> dst, UInt3 res) noexcept {
        set_block_size(8u, 8u, 8u);
        set_name("voxel_dilate");

        UInt3 coord = dispatch_id().xyz();
        $if(any(coord >= res)) { $return(); };

        constexpr uint B = 8u;
        constexpr uint T = B + 2u;
        constexpr uint TILE_TOTAL = T * T * T;   // 1000
        constexpr uint BLOCK_TOTAL = B * B * B;  // 512

        Shared<float> tile(TILE_TOTAL);

        UInt3 lid = thread_id();
        UInt3 gid = block_id();

        UInt flat_tid = lid.x + lid.y * B + lid.z * B * B;

        // Pass 0: each thread loads tile cell #flat_tid
        {
            UInt si = flat_tid;
            UInt tz = si / (T * T);
            UInt ry = si - tz * T * T;
            UInt ty = ry / T;
            UInt tx = ry - ty * T;
            Int gxx = cast<Int>(gid.x * B + tx) - 1;
            Int gyy = cast<Int>(gid.y * B + ty) - 1;
            Int gzz = cast<Int>(gid.z * B + tz) - 1;
            $if(gxx >= 0 & cast<UInt>(gxx) < res.x &
                 gyy >= 0 & cast<UInt>(gyy) < res.y &
                 gzz >= 0 & cast<UInt>(gzz) < res.z) {
                UInt flat = cast<UInt>(gxx) + cast<UInt>(gyy) * res.x
                          + cast<UInt>(gzz) * res.x * res.y;
                tile.write(si, cast<Float>(src.read(flat)));
            } $else {
                tile.write(si, 0.0f);
            };
        }

        // Pass 1: each thread loads tile cell #(flat_tid + 512)
        $if(flat_tid + BLOCK_TOTAL < TILE_TOTAL) {
            UInt si = flat_tid + BLOCK_TOTAL;
            UInt tz = si / (T * T);
            UInt ry = si - tz * T * T;
            UInt ty = ry / T;
            UInt tx = ry - ty * T;
            Int gxx = cast<Int>(gid.x * B + tx) - 1;
            Int gyy = cast<Int>(gid.y * B + ty) - 1;
            Int gzz = cast<Int>(gid.z * B + tz) - 1;
            $if(gxx >= 0 & cast<UInt>(gxx) < res.x &
                 gyy >= 0 & cast<UInt>(gyy) < res.y &
                 gzz >= 0 & cast<UInt>(gzz) < res.z) {
                UInt flat = cast<UInt>(gxx) + cast<UInt>(gyy) * res.x
                          + cast<UInt>(gzz) * res.x * res.y;
                tile.write(si, cast<Float>(src.read(flat)));
            } $else {
                tile.write(si, 0.0f);
            };
        };

        sync_block();

        // Shared memory coords for this thread's cell (1-cell border offset)
        UInt sx = lid.x + 1u;
        UInt sy = lid.y + 1u;
        UInt sz = lid.z + 1u;

        // 3x3x3 max-filter from shared memory
        Float max_density = 0.f;
        $for(dz, 3u) {
            $for(dy, 3u) {
                $for(dx, 3u) {
                    UInt ri = (sx + dx - 1u) + (sy + dy - 1u) * T + (sz + dz - 1u) * T * T;
                    max_density = max(max_density, tile.read(ri));
                };
            };
        };

        // Write result for valid voxels
        dst.write(coord.x + coord.y * res.x + coord.z * res.x * res.y,
                      cast<half>(max_density));
    });

    // Summary builder: each thread handles one coarse cell (kBlockSize³ fine cells)
    // Writes 1 to summary if any fine cell is occupied, and atomically counts occupied fine cells
    // Early-exit: stops reading once any occupied cell is found (count is only checked for >0)
    _buildSummaryShader = _device.compile<1>(
        [&](BufferVar<half> fine,
            BufferVar<uint> summary_out,
            BufferVar<uint> count_out,
            UInt3 fine_res,
            UInt3 summary_res) noexcept {
        set_block_size(256u);
        set_name("voxel_build_summary");
        UInt idx = dispatch_id().x;
        UInt total_summary = summary_res.x * summary_res.y * summary_res.z;
        $if(idx >= total_summary) { $return(); };

        UInt sz = idx / (summary_res.x * summary_res.y);
        UInt rem = idx - sz * summary_res.x * summary_res.y;
        UInt sy = rem / summary_res.x;
        UInt sx = rem - sy * summary_res.x;

        UInt any_occupied = 0u;

        UInt fine_x0 = sx * kBlockSize;
        UInt fine_y0 = sy * kBlockSize;
        UInt fine_z0 = sz * kBlockSize;

        Bool found = false;
        $for(dz, kBlockSize) {
            $if(found) { $break; };
            UInt fz = fine_z0 + dz;
            $if(fz >= fine_res.z) { $break; };
            $for(dy, kBlockSize) {
                $if(found) { $break; };
                UInt fy = fine_y0 + dy;
                $if(fy >= fine_res.y) { $break; };
                $for(dx, kBlockSize) {
                    UInt fx = fine_x0 + dx;
                    $if(fx >= fine_res.x) { $break; };
                    UInt flat = fx + fy * fine_res.x + fz * fine_res.x * fine_res.y;
                    Float d = fine.read(flat);
                    $if(d > 0.0f) {
                        any_occupied = 1u;
                        found = true;
                        $break;
                    };
                };
            };
        };

        summary_out.write(idx, any_occupied);
        $if(any_occupied > 0u) {
            count_out->atomic(0u).fetch_add(1u);
        };
    });
}

void VoxelGrid::clear(Stream& stream) {
    if (!_created || _totalVoxels == 0u) return;
    stream << _clearShader(_occupancy, _totalVoxels).dispatch(_totalVoxels);
}

void VoxelGrid::buildSummary(Stream& stream) {
    if (!_created || _totalSummary == 0u) return;
    // Reset count, then build summary + count in one dispatch
    static const uint zero = 0u;
    _occupiedCount.view(0u, 1u).copy_from(&zero);
    stream << _buildSummaryShader(
        _occupancy, _summary, _occupiedCount,
        _resolution, _summaryRes
    ).dispatch(_totalSummary);
}

void VoxelGrid::voxelize(Stream& stream, const Buffer<float3>& positions, uint particle_count) {
    if (!_created || particle_count == 0u) return;
    stream << _voxelizeShader(
        _occupancy,
        positions,
        particle_count,
        _params.bounds_min,
        _params.bounds_max,
        _resolution
    ).dispatch(particle_count);
}

void VoxelGrid::voxelize(Stream& stream, const Buffer<luisa::float4>& positions, uint particle_count, float radius_scale) {
    if (!_created || particle_count == 0u) return;
    stream << _voxelizeF4Shader(
        _occupancy,
        positions,
        particle_count,
        _params.bounds_min,
        _params.bounds_max,
        _resolution,
        radius_scale
    ).dispatch(particle_count);
}

void VoxelGrid::dilate(Stream& stream, uint iterations) {
    if (!_created || iterations == 0u || _totalVoxels == 0u) return;
    for (uint i = 0u; i < iterations; ++i) {
        stream << _dilateShader(_occupancy, _occupancyTemp, _resolution)
                      .dispatch(_resolution.x, _resolution.y, _resolution.z);
        stream << _occupancy.copy_from(_occupancyTemp.view(0u, _totalVoxels));
    }
}

void VoxelGrid::update(Stream& stream) {
    if (!_created) return;
    clear(stream);
}

void VoxelGrid::register_cloud(void* cloud) {
    for (auto& c : _registeredClouds) {
        if (c == cloud) return;
    }
    _registeredClouds.push_back(cloud);
}

void VoxelGrid::unregister_cloud(void* cloud) {
    for (auto it = _registeredClouds.begin(); it != _registeredClouds.end(); ++it) {
        if (*it == cloud) {
            _registeredClouds.erase(it);
            return;
        }
    }
}

} // namespace newtype::scene
