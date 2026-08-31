#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::scene {

struct VoxelGridParams {
    luisa::float3 bounds_min;
    luisa::float3 bounds_max;
    luisa::uint3  resolution;          // per-axis resolution (res_x, res_y, res_z)
    float         cell_size;            // uniform cell size (same on all axes)
    luisa::uint3  summary_resolution;   // coarse grid: (resolution + kBlockSize - 1) / kBlockSize
};

class VoxelGrid {
public:
    explicit VoxelGrid(luisa::compute::Device& device) noexcept;

    /// Allocate grid resources. Computes per-axis resolution from base_resolution
    /// so that all cells are uniformly sized (cell_size = max_extent / base_resolution).
    void create(luisa::uint base_resolution,
                luisa::float3 bounds_min,
                luisa::float3 bounds_max);

    /// Clear occupancy to zero
    void clear(luisa::compute::Stream& stream);

    /// Voxelize particles into the grid.
    void voxelize(luisa::compute::Stream& stream,
                  const luisa::compute::Buffer<luisa::float3>& positions,
                  luisa::uint particle_count);

    /// Float4 positions variant (reads .xyz() for position, .w for radius)
    /// radius_scale: multiplier applied to particle radius during voxelization (for shadow inflation)
    void voxelize(luisa::compute::Stream& stream,
                  const luisa::compute::Buffer<luisa::float4>& positions,
                  luisa::uint particle_count,
                  float radius_scale = 1.0f);

    void update(luisa::compute::Stream& stream);

    /// Morphological dilation: expand occupancy by N cells (3x3x3 max-filter per iteration)
    void dilate(luisa::compute::Stream& stream, luisa::uint iterations);

    /// Build block-level summary + global occupied count. Call after voxelize+dilate.
    void buildSummary(luisa::compute::Stream& stream);

    void register_cloud(void* cloud);
    void unregister_cloud(void* cloud);

    // Accessors
    [[nodiscard]] const luisa::compute::Buffer<luisa::half>& occupancy() const noexcept { return _occupancy; }
    [[nodiscard]] const luisa::compute::Buffer<uint>&   summary() const noexcept { return _summary; }
    [[nodiscard]] const luisa::compute::Buffer<uint>&   occupiedCount() const noexcept { return _occupiedCount; }
    [[nodiscard]] const luisa::compute::Buffer<VoxelGridParams>& paramsBuffer() const noexcept { return _paramsBuffer; }
    [[nodiscard]] const VoxelGridParams& params() const noexcept { return _params; }
    [[nodiscard]] luisa::uint3 resolution() const noexcept { return _resolution; }
    [[nodiscard]] luisa::uint3 summaryResolution() const noexcept { return _summaryRes; }
    [[nodiscard]] luisa::uint total_voxels() const noexcept { return _totalVoxels; }
    [[nodiscard]] bool created() const noexcept { return _created; }

    // Bundle all GPU resources for passing as a single shader argument.
    // Params passed as scalar constants (not buffer read) to avoid LUISA_STRUCT layout issues.
    struct Resources {
        const luisa::compute::Buffer<luisa::half>& occupancy;
        const luisa::compute::Buffer<uint>& summary;
        const luisa::compute::Buffer<uint>& occupied_count;
        luisa::float3 bounds_min;
        luisa::float3 bounds_max;
        luisa::uint3  resolution;
        float         cell_size;
        luisa::uint3  summary_resolution;
    };
    [[nodiscard]] Resources gpu_resources() const noexcept {
        return Resources{
            _occupancy, _summary, _occupiedCount,
            _params.bounds_min, _params.bounds_max,
            _params.resolution, _params.cell_size,
            _params.summary_resolution
        };
    }

    static constexpr luisa::uint kBlockSize = 8u;

private:
    void _compileShaders();

    luisa::compute::Device& _device;

    luisa::compute::Buffer<luisa::half>    _occupancy;
    luisa::compute::Buffer<luisa::half>    _occupancyTemp;  // ping-pong for dilation
    luisa::compute::Buffer<uint>           _summary;        // coarse grid: 1 if any fine cell in block is occupied
    luisa::compute::Buffer<uint>           _occupiedCount;  // single uint: total occupied fine cells (0 = grid empty)
    luisa::compute::Buffer<VoxelGridParams>       _paramsBuffer;

    VoxelGridParams _params{};
    luisa::uint3 _resolution = luisa::make_uint3(0u);
    luisa::uint3 _summaryRes = luisa::make_uint3(0u);
    luisa::uint  _totalVoxels = 0u;
    luisa::uint  _totalSummary = 0u;
    bool _created = false;

    luisa::vector<void*> _registeredClouds;

    // Compiled kernels
    luisa::compute::Shader<1,
        luisa::compute::Buffer<luisa::half>,
        luisa::uint
    > _clearShader;

    luisa::compute::Shader<1,
        luisa::compute::Buffer<luisa::half>,
        luisa::compute::Buffer<luisa::float3>,
        luisa::uint,
        luisa::float3,
        luisa::float3,
        luisa::uint3
    > _voxelizeShader;

    luisa::compute::Shader<1,
        luisa::compute::Buffer<luisa::half>,
        luisa::compute::Buffer<luisa::float4>,
        luisa::uint,
        luisa::float3,
        luisa::float3,
        luisa::uint3,
        float
    > _voxelizeF4Shader;

    luisa::compute::Shader<3,
        luisa::compute::Buffer<luisa::half>,
        luisa::compute::Buffer<luisa::half>,
        luisa::uint3
    > _dilateShader;

    luisa::compute::Shader<1,
        luisa::compute::Buffer<luisa::half>, // fine occupancy
        luisa::compute::Buffer<uint>,    // summary out
        luisa::compute::Buffer<uint>,    // occupied count out
        luisa::uint3,                    // fine resolution
        luisa::uint3                     // summary resolution
    > _buildSummaryShader;
};

} // namespace newtype::scene

// LUISA_STRUCT registration
LUISA_STRUCT(newtype::scene::VoxelGridParams,
             bounds_min, bounds_max, resolution, cell_size, summary_resolution) {};

// Binding group: bundles VoxelGrid GPU resources + scalar params into a single shader parameter
LUISA_BINDING_GROUP(newtype::scene::VoxelGrid::Resources,
                    occupancy, summary, occupied_count,
                    bounds_min, bounds_max, resolution, cell_size, summary_resolution) {};
