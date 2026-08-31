#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <cinder/TriMesh.h>
#include "newtype/feature/RasterBase.h"
#include "newtype/core/IVoxelGridUser.h"
#include "newtype/scene/VoxelGrid.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/core/BindingGroups.h"

namespace newtype::feature {

/// Particle data — position (xyz) + size (w), velocity (xyz) + alpha (w)
struct ParticleBase {
    luisa::float4 position;  // xyz = world position, w = particle size
    luisa::float4 velocity;  // xyz = velocity, w = per-particle alpha [0..1]
};

/// Minimal mesh-to-vertex struct for instanced point billboards.
/// Only provides position (from vertex buffer) and instance_id (SV_InstanceID, GPU-generated).
struct PointAppData {
    luisa::float4 position;     // corner offset from unit quad (RGBA32F)
    luisa::uint    instance_id; // particle index — SV_InstanceID
};

/// Vertex data for custom mesh GPU buffer (40 bytes — matches MeshFormat stride).
/// No LUISA_STRUCT needed — only used for buffer stride, not DSL access.
struct CustomMeshVertex {
    luisa::float4 position;
    luisa::float4 normal;
    luisa::float2 uv;
};

/// App data for custom mesh particles — carries Position + Normal + UV + SV_InstanceID.
struct CustomMeshAppData {
    luisa::float4 position;     // Position (RGBA32F) — mesh vertex position
    luisa::float4 normal;       // Normal (RGBA32F) — mesh vertex normal
    luisa::float2 uv;           // UV0 (RG32F) — mesh vertex UV
    luisa::uint   instance_id;  // SV_InstanceID
};

} // namespace newtype::feature

// LUISA_STRUCT registration (global scope)
LUISA_STRUCT(newtype::feature::ParticleBase, position, velocity) {};
LUISA_STRUCT(newtype::feature::PointAppData, position, instance_id) {};
LUISA_STRUCT(newtype::feature::CustomMeshAppData, position, normal, uv, instance_id) {};

namespace newtype::feature {

// Vertex → Pixel varyings (first member MUST be float4 position for rasterizer)
struct PointV2P {
    luisa::float4 position;      // clip-space position (MUST be first)
    luisa::float4 packed_data;   // x=linear_depth, y=particle_id, z=motion_ndc.x, w=motion_ndc.y
    luisa::float2 corner_uv;     // billboard corner offset (-1..+1)
    luisa::float3 cam_right;     // camera right for sphere normal transform
    luisa::float3 cam_up;        // camera up for sphere normal transform
    luisa::float3 cam_forward;   // billboard forward for sphere normal transform
};

// Vertex → Pixel varyings for custom mesh particles
struct CustomMeshV2P {
    luisa::float4 position;      // clip-space position (MUST be first)
    luisa::float4 packed_data;   // x=linear_depth, y=particle_id, z=barycentric index, w=unused
    luisa::float3 world_normal;  // mesh normal in world space
    luisa::float2 uv;            // mesh texture coordinates
    luisa::float2 motion;        // screen-space motion NDC (xy)
};

// Vertex → Pixel varyings for transparent billboard particles
struct PointTransparentV2P {
    luisa::float4 position;      // clip-space position (MUST be first)
    luisa::float4 packed_data;   // x=linear_depth, y=particle_id, z=motion_ndc.x, w=motion_ndc.y
    luisa::float2 corner_uv;     // billboard corner offset (-1..+1)
    luisa::float3 cam_right;
    luisa::float3 cam_up;
    luisa::float3 cam_forward;
    luisa::float3 world_pos;     // world-space position for forward lighting
};

} // namespace newtype::feature

LUISA_STRUCT(newtype::feature::PointV2P, position, packed_data, corner_uv, cam_right, cam_up, cam_forward) {};
LUISA_STRUCT(newtype::feature::CustomMeshV2P, position, packed_data, world_normal, uv, motion) {};
LUISA_STRUCT(newtype::feature::PointTransparentV2P, position, packed_data, corner_uv, cam_right, cam_up, cam_forward, world_pos) {};

namespace newtype::feature {

//==============================================================================
// Custom Shader Hooks
//==============================================================================

/// Custom shade function for point cloud forward lighting.
/// Only valid inside device.compile() / Callable contexts.
/// Parameters:
///   normal       — sphere face normal (world space)
///   mat          — material data from GPU buffer
///   env_radiance — pre-computed environment radiance along normal
/// Returns: Float3 lit color.
using PointCloudShadeFn = std::function<luisa::compute::Float3(
    luisa::compute::Float3,
    luisa::compute::Var<render::MaterialData>,
    luisa::compute::Float3)>;

namespace detail {

inline luisa::compute::Float3 default_pointcloud_shade(
    luisa::compute::Float3 normal,
    luisa::compute::Var<render::MaterialData> mat,
    luisa::compute::Float3 env_radiance) noexcept {
    return env_radiance * mat.albedo.xyz();
}

} // namespace detail

/// PointCloud feature: renders dynamic particles via hardware rasterization
/// into the G-buffer, with self-shadowing via voxel grid DDA ray march.
///
/// Uses instanced billboard quads — a 6-vertex unit quad drawn N times.
/// The vertex shader uses instance_id to look up particle data and computes
/// camera-facing billboard corners. Intermediate RTVs are merged into the
/// G-buffer by RasterBase.
///
/// Supports transparent mode (BlendMode::Transparent) with McGuire weighted
/// blended OIT. Per-particle alpha comes from velocity.w.
///
/// Usage:
///   auto cloud = std::make_unique<PointCloud>(device, voxelGrid, 1'000'000u, matId);
///   cloud->upload_positions(stream, pos_data, vel_data, count);
///   pipeline.addFeature(std::move(cloud));
class PointCloud : public RasterBase, public core::IVoxelGridUser {
public:
    PointCloud(luisa::compute::Device& device,
               luisa::uint max_particles,
               luisa::uint material_id,
               luisa::float3 bounds_min = {-50.f, -50.f, -50.f},
               luisa::float3 bounds_max = { 50.f,  50.f,  50.f},
               luisa::uint voxel_resolution = 256u);

    ~PointCloud() override;

    // --- IVoxelGridUser interface ---
    [[nodiscard]] core::VoxelGridConfig voxelGridConfig() const override {
        return {_boundsMin, _boundsMax, _voxelResolution};
    }
    [[nodiscard]] luisa::uint shadowDilation() const override { return _shadowDilation; }

    // --- RasterBase hooks ---
    void onRasterInit(luisa::compute::Device& device) override;
    bool onRasterExecute(luisa::compute::Stream& rasterStream,
                          RasterContext& rc,
                          const core::FeatureContext& ctx) override;
    bool onRasterExecuteTransparent(luisa::compute::Stream& rasterStream,
                                     RasterContext& rc,
                                     const core::FeatureContext& ctx) override;
    void onVoxelize(luisa::compute::Stream& computeStream,
                    const core::FeatureContext& ctx) override;
    [[nodiscard]] bool shouldExecute() const override;

    // --- Data upload ---
    /// Upload positions (xyz+size) and velocities (xyz+alpha).
    /// velocity.w = per-particle alpha (0.0 = invisible, 1.0 = opaque).
    void upload_positions(luisa::compute::Stream& stream,
                          const luisa::vector<luisa::float4>& positions,
                          const luisa::vector<luisa::float4>& velocities,
                          luisa::uint count);

    void upload_positions(luisa::compute::Stream& stream,
                          luisa::compute::BufferView<luisa::float4> positions,
                          luisa::compute::BufferView<luisa::float4> velocities,
                          luisa::uint count);

    // --- Custom mesh ---
    /// Load a custom mesh to use instead of the billboard quad.
    /// Must be called BEFORE buildScene() / onInit().
    void setMesh(const ci::geom::Source& geom);

    // --- Custom shader hooks ---
    /// Set a custom shade function. Call recompile() afterward to rebuild shaders.
    /// Pass {} to revert to default (env_radiance * albedo).
    void setCustomShadeFn(PointCloudShadeFn fn) { _customShadeFn = std::move(fn); }

    /// Recompile raster shaders with current custom hooks.
    /// Must be called between frames (before any draw commands).
    void recompile(luisa::compute::Device& device);

    // --- Parameters ---
    luisa::uint material_id;

    // --- Accessors ---
    [[nodiscard]] luisa::uint particle_count() const noexcept { return _particleCount; }
    [[nodiscard]] luisa::uint max_particles() const noexcept { return _maxParticles; }

    [[nodiscard]] luisa::compute::Buffer<luisa::float4>& pos_buffer() { return _positionsBuffer; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float4>& pos_buffer() const { return _positionsBuffer; }
    [[nodiscard]] luisa::compute::Buffer<luisa::float4>& vel_buffer() { return _velocitiesBuffer; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float4>& vel_buffer() const { return _velocitiesBuffer; }

    void load(const nlohmann::json& file) override;
    [[nodiscard]] nlohmann::json toJson() const override;

    void drawUi() override;

private:
    void _compileRasterShader(luisa::compute::Device& device);
    void _initRasterState(luisa::compute::Device& device);

    scene::VoxelGrid* _voxelGrid = nullptr;  // lazy-bound from Pipeline via ctx.pipeline.voxelGrid()
    luisa::uint _maxParticles;
    luisa::uint _particleCount = 0u;

    // Voxel grid config (provided to Pipeline via IVoxelGridUser)
    luisa::float3 _boundsMin;
    luisa::float3 _boundsMax;
    luisa::uint   _voxelResolution;

    // --- GPU resources ---
    luisa::compute::Buffer<luisa::float4> _positionsBuffer;   // xyz=pos, w=size
    luisa::compute::Buffer<luisa::float4> _velocitiesBuffer;  // xyz=vel, w=alpha
    luisa::compute::Buffer<luisa::float4> _quadVB;            // 6-vertex unit billboard quad (float4 per vertex)
    luisa::compute::Buffer<CustomMeshVertex> _customMeshVB;   // interleaved vertex buffer for custom mesh
    luisa::uint _customMeshVertexCount = 0u;                  // number of vertices in custom mesh
    bool _useCustomMesh = false;                               // set by setMesh(), read at compile time

    // Rasterizer resources (subclass-specific: shader + format + state)
    luisa::compute::MeshFormat    _meshFormat;
    luisa::compute::RasterState   _rasterState;
    luisa::compute::RasterState   _transparentRasterState;

    // Opaque G-buffer raster shader
    using PointRasterShader = luisa::compute::RasterShader<
        luisa::compute::Buffer<luisa::float4>,
        luisa::compute::Buffer<luisa::float4>,
        newtype::util::CameraData,
        luisa::uint,
        float,
        luisa::uint
    >;
    PointRasterShader _rasterShader;

    // Transparent forward-lit + OIT raster shader (binding groups expanded for raster invoke)
    using TransparentRasterShader = luisa::compute::RasterShader<
        luisa::compute::Buffer<luisa::float4>,        // positions
        luisa::compute::Buffer<luisa::float4>,        // velocities (xyz=vel, w=alpha)
        newtype::util::CameraData,                    // camera
        luisa::uint,                                  // count
        float,                                        // dt
        luisa::uint,                                  // material_id
        luisa::uint,                                  // receive_shadow
        luisa::compute::Buffer<luisa::half>,          // voxel occupancy
        luisa::compute::Buffer<luisa::uint>,          // voxel summary
        luisa::compute::Buffer<luisa::uint>,          // voxel occupied_count
        scene::VoxelGridParams,                       // voxel scalars
        luisa::compute::Buffer<render::MaterialData>, // material pool
        luisa::compute::Image<float>,                 // envmap
        luisa::compute::Buffer<float>,                // env marginal cdf
        luisa::compute::Buffer<float>,                // env conditional cdf
        float,                                        // env integral
        luisa::uint,                                  // env width
        luisa::uint,                                  // env height
        luisa::float3x3,                              // env rotation
        float,                                        // env exposure
        luisa::compute::Image<float>                  // gbuf_depth (software depth test)
    >;
    TransparentRasterShader _transparentShader;

    // Debug voxel grid visualizer
    luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // output (RGBA16F)
        luisa::compute::Buffer<luisa::half>, // occupancy (half precision)
        luisa::float3,                  // bounds_min
        luisa::float3,                  // bounds_max
        luisa::uint3,                   // resolution (per-axis)
        float,                          // cell_size
        newtype::util::CameraData       // camera
    > _voxelDebugShader;
    bool _voxelDebug = false;

    luisa::uint _shadowDilation = 2u;
    float _shadowRadiusScale = 5.0f;

    // Custom shader hooks
    PointCloudShadeFn _customShadeFn;
};

} // namespace newtype::feature
