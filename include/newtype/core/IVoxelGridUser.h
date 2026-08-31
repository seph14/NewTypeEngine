#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::core {

/// Configuration for the shared VoxelGrid, provided by features that voxelize particles.
struct VoxelGridConfig {
    luisa::float3 bounds_min;
    luisa::float3 bounds_max;
    luisa::uint   resolution = 256u;
};

/// Interface for features that need a voxel grid.
/// Pipeline queries all IVoxelGridUser features in buildScene() to determine
/// the VoxelGrid bounds and resolution.
class IVoxelGridUser {
public:
    virtual ~IVoxelGridUser() = default;
    [[nodiscard]] virtual VoxelGridConfig voxelGridConfig() const = 0;
    [[nodiscard]] virtual luisa::uint shadowDilation() const { return 0u; }
};

} // namespace newtype::core
