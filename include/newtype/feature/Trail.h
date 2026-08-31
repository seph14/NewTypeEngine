#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include "newtype/feature/RasterBase.h"
#include "newtype/core/IVoxelGridUser.h"
#include "newtype/scene/VoxelGrid.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/core/BindingGroups.h"

namespace newtype::feature {

/// A single trail point: position + width (adjacent format).
/// N points define N-1 segments; segment i spans points[i]→points[i+1].
struct TrailPoint {
    luisa::float4 pos_width;  // xyz = position, w = width
};

/// Vertex → Pixel varyings for transparent trail ribbon
struct TrailTransparentV2P {
    luisa::float4 position;      // clip-space (MUST be first)
    luisa::float4 packed_data;   // x=linear_depth, y=segment_id(bitcast), z=lerp_t, w=unused
    luisa::float3 ribbon_normal; // face normal for shading
    luisa::float3 world_pos;     // world-space position for forward lighting
};

} // namespace newtype::feature

LUISA_STRUCT(newtype::feature::TrailPoint, pos_width) {};
LUISA_STRUCT(newtype::feature::TrailTransparentV2P, position, packed_data, ribbon_normal, world_pos) {};

namespace newtype::feature {

/// App data for trail ribbon — same layout as PointAppData:
/// position.xy = (side, t) corner pair, instance_id = segment index
struct TrailAppData {
    luisa::float4 position;     // corner: x=side(-1/+1), y=t(0/1), zw=unused
    luisa::uint   instance_id;  // segment index — SV_InstanceID
};

/// Vertex → Pixel varyings for trail ribbon
struct TrailV2P {
    luisa::float4 position;      // clip-space (MUST be first)
    luisa::float4 packed_data;   // x=linear_depth, y=segment_id(bitcast), z=lerp_t, w=unused
    luisa::float3 ribbon_normal; // face normal for shading
};

} // namespace newtype::feature

LUISA_STRUCT(newtype::feature::TrailAppData, position, instance_id) {};
LUISA_STRUCT(newtype::feature::TrailV2P, position, packed_data, ribbon_normal) {};

namespace newtype::feature {

//==============================================================================
// Custom Shader Hooks
//==============================================================================

/// Custom width function for trail ribbon segments.
/// Only valid inside device.compile() / Callable contexts.
/// Parameters:
///   t     — interpolation along segment [0, 1]
///   seg_id — segment index
///   points — trail point buffer (read pt0=points[seg_id], pt1=points[seg_id+1])
/// Returns: Float width at the given interpolation point.
using TrailWidthFn = std::function<luisa::compute::Float(
    luisa::compute::Float, luisa::compute::UInt,
    const luisa::compute::BufferVar<TrailPoint>&)>;

/// Custom shade function for trail forward lighting.
/// Only valid inside device.compile() / Callable contexts.
/// Parameters:
///   normal       — ribbon face normal (world space)
///   mat          — material data from GPU buffer
///   env_radiance — pre-computed environment radiance along normal
/// Returns: Float3 lit color.
using TrailShadeFn = std::function<luisa::compute::Float3(
    luisa::compute::Float3,
    luisa::compute::Var<render::MaterialData>,
    luisa::compute::Float3)>;

namespace detail {

inline luisa::compute::Float default_trail_width(
    luisa::compute::Float t, luisa::compute::UInt seg_id,
    const luisa::compute::BufferVar<TrailPoint>& points) noexcept {
    using namespace luisa::compute;
    Var<TrailPoint> pt0 = points.read(seg_id);
    Var<TrailPoint> pt1 = points.read(seg_id + 1u);
    return lerp(pt0.pos_width.w, pt1.pos_width.w, t);
}

inline luisa::compute::Float3 default_trail_shade(
    luisa::compute::Float3 normal,
    luisa::compute::Var<render::MaterialData> mat,
    luisa::compute::Float3 env_radiance) noexcept {
    return env_radiance * mat.albedo.xyz();
}

} // namespace detail

/// Trail feature: renders ribbon trails via hardware rasterization into the G-buffer.
/// Each segment is a camera-facing quad (billboarded ribbon) with width interpolation.
/// Self-shadowing via voxel grid DDA ray march (same path as PointCloud).
///
/// Supports transparent mode (BlendMode::Transparent) with McGuire weighted blended OIT.
/// Uses uniform alpha per trail feature.
///
/// Data model: N TrailPoint entries → N-1 segments. Segment i spans points[i]→points[i+1].
///
/// Usage:
///   auto trail = std::make_unique<Trail>(device, maxSegments, matId);
///   trail->uploadPoints(stream, points, pointCount);
///   pipeline.addFeature(std::move(trail));
class Trail : public RasterBase, public core::IVoxelGridUser {
public:
    Trail(luisa::compute::Device& device,
          luisa::uint max_segments,
          luisa::uint material_id,
          luisa::float3 bounds_min = {-50.f, -50.f, -50.f},
          luisa::float3 bounds_max = { 50.f,  50.f,  50.f},
          luisa::uint voxel_resolution = 256u);

    ~Trail() override;

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
    void uploadPoints(luisa::compute::Stream& stream,
                      const luisa::vector<TrailPoint>& points,
                      luisa::uint point_count);

    void uploadPoints(luisa::compute::Stream& stream,
                      const luisa::compute::BufferView<TrailPoint>& points,
                      luisa::uint point_count);

    // --- Accessors ---
    [[nodiscard]] auto& pointBuffer() { return _pointBuffer; }
    [[nodiscard]] auto  pointCount() const { return _pointCount; }
    [[nodiscard]] uint  segmentCount() const { return _pointCount > 0u ? _pointCount - 1u : 0u; }

    // --- Custom shader hooks ---
    /// Set a custom width function. Call recompile() afterward to rebuild shaders.
    /// Pass {} to revert to default (linear lerp).
    void setCustomWidthFn(TrailWidthFn fn) { _customWidthFn = std::move(fn); }
    /// Set a custom shade function. Call recompile() afterward to rebuild shaders.
    /// Pass {} to revert to default (env_radiance * albedo).
    void setCustomShadeFn(TrailShadeFn fn) { _customShadeFn = std::move(fn); }

    /// Recompile raster shaders with current custom hooks.
    /// Must be called between frames (before any draw commands).
    void recompile(luisa::compute::Device& device);

    // --- Parameters ---
    luisa::uint material_id;

    void load(const nlohmann::json& file) override;
    [[nodiscard]] nlohmann::json toJson() const override;

    void drawUi() override;

private:
    void _compileShaders(luisa::compute::Device& device);
    scene::VoxelGrid* _voxelGrid = nullptr;
    luisa::uint _maxPoints;      // max points the buffer can hold (max_segments + 1)
    luisa::uint _pointCount = 0u;

    // Voxel grid config
    luisa::float3 _boundsMin;
    luisa::float3 _boundsMax;
    luisa::uint   _voxelResolution;

    // GPU resources
    luisa::compute::Buffer<TrailPoint>  _pointBuffer;
    luisa::compute::Buffer<luisa::float4> _quadVB;        // 6-vertex ribbon quad
    luisa::compute::Buffer<luisa::float4> _positionBuffer; // flat positions for voxelize (1 per point)

    // Rasterizer resources
    luisa::compute::MeshFormat  _meshFormat;
    luisa::compute::RasterState _rasterState;
    luisa::compute::RasterState _transparentRasterState;

    // Opaque G-buffer raster shader
    using TrailRasterShader = luisa::compute::RasterShader<
        luisa::compute::Buffer<TrailPoint>,
        newtype::util::CameraData,
        luisa::uint,
        float,
        luisa::uint
    >;
    TrailRasterShader _rasterShader;

    // Transparent forward-lit + OIT raster shader (binding groups expanded for raster invoke)
    using TransparentTrailRasterShader = luisa::compute::RasterShader<
        luisa::compute::Buffer<TrailPoint>,
        newtype::util::CameraData,
        luisa::uint,
        float,                                          // miter_limit
        luisa::uint,
        float,                                          // alpha
        luisa::uint,                                    // receive_shadow
        luisa::compute::Buffer<luisa::half>,            // voxel occupancy
        luisa::compute::Buffer<luisa::uint>,            // voxel summary
        luisa::compute::Buffer<luisa::uint>,            // voxel occupied_count
        scene::VoxelGridParams,                         // voxel scalars
        luisa::compute::Buffer<render::MaterialData>,   // material pool
        luisa::compute::Image<float>,                   // envmap
        luisa::compute::Buffer<float>,                  // env marginal cdf
        luisa::compute::Buffer<float>,                  // env conditional cdf
        float,                                          // env integral
        luisa::uint,                                    // env width
        luisa::uint,                                    // env height
        luisa::float3x3,                                // env rotation
        float,                                          // env exposure
        luisa::compute::Image<float>                    // gbuf depth
    >;
    TransparentTrailRasterShader _transparentShader;

    // Unpack shader: points → flat position buffer for voxelize (applies radius_scale)
    luisa::compute::Shader<1,
        luisa::compute::Buffer<TrailPoint>,
        luisa::compute::Buffer<luisa::float4>,
        luisa::uint,
        float                          // radius_scale (runtime)
    > _unpackShader;

    luisa::uint _shadowDilation = 2u;
    float _shadowRadiusScale = 3.0f;
    float _miterLimit = 4.0f;
    float _alpha = 1.0f;  // uniform alpha per trail feature

    // Custom shader hooks
    TrailWidthFn _customWidthFn;
    TrailShadeFn _customShadeFn;
};

} // namespace newtype::feature
