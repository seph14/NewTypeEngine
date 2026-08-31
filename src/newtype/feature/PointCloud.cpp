#include "newtype/feature/PointCloud.h"
#include "newtype/feature/RasterContext.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Camera.h"
#include "newtype/util/Profiler.h"
#include "newtype/render/Shading.h"
#include "newtype/render/DDAMarch.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/core/BindingGroups.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "newtype/core/Config.h"
#include "cinder/TriMesh.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {

using namespace luisa;
using namespace luisa::compute;

PointCloud::PointCloud(Device& device, uint max_particles, uint material_id,
                       luisa::float3 bounds_min, luisa::float3 bounds_max, uint voxel_resolution)
    : RasterBase(device)
    , _maxParticles(max_particles)
    , material_id(material_id)
    , _boundsMin(bounds_min)
    , _boundsMax(bounds_max)
    , _voxelResolution(voxel_resolution)
{
    _positionsBuffer  = _device.create_buffer<luisa::float4>(max_particles);
    _velocitiesBuffer = _device.create_buffer<luisa::float4>(max_particles);
}

PointCloud::~PointCloud() = default;

void PointCloud::setMesh(const ci::geom::Source& geom) {
    ci::TriMesh triMesh(geom);
    if (!triMesh.hasNormals()) triMesh.recalculateNormals();

    const auto& indices = triMesh.getIndices();
    if (indices.empty()) return;

    const ci::vec3* positions = triMesh.getPositions<3>();
    const std::vector<ci::vec3>& normals = triMesh.getNormals();
    const ci::vec2* texCoords = triMesh.hasTexCoords0() ? triMesh.getTexCoords0<2>() : nullptr;

    size_t triCount = indices.size() / 3u;
    _customMeshVertexCount = static_cast<uint>(triCount * 3u);

    luisa::vector<CustomMeshVertex> vertexData(_customMeshVertexCount);

    for (size_t t = 0; t < triCount; ++t) {
        for (int v = 0; v < 3; ++v) {
            uint idx = static_cast<uint>(indices[t * 3 + v]);
            auto& vtx = vertexData[t * 3 + v];

            auto& p = positions[idx];
            vtx.position = {p.x, p.y, p.z, 0.f};

            auto& n = normals[idx];
            vtx.normal = {n.x, n.y, n.z, static_cast<float>(v)};

            if (texCoords) {
                auto& uv = texCoords[idx];
                vtx.uv = {uv.x, uv.y};
            } else {
                vtx.uv = {0.f, 0.f};
            }
        }
    }

    _customMeshVB = _device.create_buffer<CustomMeshVertex>(_customMeshVertexCount);
    auto stream = _device.create_stream();
    stream << _customMeshVB.copy_from(luisa::span{vertexData.data(), vertexData.size()});

    _useCustomMesh = true;
}

void PointCloud::_compileRasterShader(Device& device) {
    auto& shadeFn = _customShadeFn;

    Callable litFunc = [&](
        Float3 world_n,
        Var<render::MaterialData> mat,
        ImageFloat envmap,
        BufferVar<float> env_marginal_cdf,
        BufferVar<float> env_conditional_cdf,
        Float env_integral,
        UInt env_width,
        UInt env_height,
        Float3x3 env_rotation,
        Float env_exposure
        ) noexcept {
        Float3 env_radiance = render::eval_envmap_radiance(
            world_n, envmap, env_width, env_height,
            env_rotation, env_exposure);
        return shadeFn
            ? shadeFn(world_n, mat, env_radiance)
            : detail::default_pointcloud_shade(world_n, mat, env_radiance);
    };

    if (_useCustomMesh) {
        auto vert = RasterStageKernel<CustomMeshV2P(CustomMeshAppData,
            luisa::compute::Buffer<luisa::float4>,
            luisa::compute::Buffer<luisa::float4>,
            util::CameraData, luisa::uint, float)>{
            [&](Var<CustomMeshAppData> app,
                BufferVar<luisa::float4> particles,
                BufferVar<luisa::float4> velocities,
                Var<util::CameraData> camera,
                UInt count,
                Float dt) noexcept {
                set_name("custom_mesh_vert");

                UInt pid = app.instance_id;

                Float4 particle = particles.read(pid);
                Float3 center = particle.xyz();
                Float  size   = particle.w;

                Float3 mesh_pos = app.position.xyz();
                Float3 world_pos = center + mesh_pos * size;

                Float4 clip = camera.view_proj * make_float4(world_pos, 1.0f);
                clip = make_float4(clip.x, -clip.y, clip.z, clip.w);

                Float3 vel = velocities.read(pid).xyz();
                Float3 prev_center = center - vel * dt;
                Float2 motion_ndc = camera->project_prev(prev_center) - camera->project(center);

                Float4 packed_data = make_float4(clip.w, as<float>(pid), app.normal.w, 0.0f);
                Float3 world_nrm = normalize(app.normal.xyz());
                Var<CustomMeshV2P> vary{clip, packed_data, world_nrm, app.uv, motion_ndc};
                return vary.expression();
            }
        };

        auto frag = RasterStageKernel<PointPixelOut(CustomMeshV2P, luisa::uint)>{
            [&](Expr<CustomMeshV2P> vary, UInt mat_id) noexcept {
                set_name("custom_mesh_frag");

                Float linear_depth = vary.packed_data.x;
                UInt  particle_id  = as<uint>(vary.packed_data.y);
                Float2 motion_ndc  = vary.motion;
                Float mx = clamp(motion_ndc.x * 0.5f + 0.5f, 0.0f, 1.0f);
                Float my = clamp(motion_ndc.y * 0.5f + 0.5f, 0.0f, 1.0f);
                UInt packed = (cast<uint>(mx * 65535.0f) << 16u) | cast<uint>(my * 65535.0f);
                Float packed_motion = as<float>(packed);

                Float2 mesh_uv = vary.uv;
                Float4 depth_out = make_float4(linear_depth, packed_motion, 0.f, 0.f);

                UInt vis_y = (1u << 30u) | particle_id;
                Float2 vis_xy = make_float2(as<float>(mat_id), as<float>(vis_y));
                Float4 vis_out = make_float4(vis_xy, 0.0f, 0.0f);

                Float2 oct_n = render::oct_encode(normalize(vary.world_normal));
                Float4 bary_out = make_float4(mesh_uv.x, mesh_uv.y, oct_n);

                Var<PointPixelOut> result{depth_out, vis_out, bary_out};
                return result.expression();
            }
        };

        auto kernel = RasterKernel<decltype(vert), decltype(frag)>{vert, frag};

        ShaderOption opt;
        opt.name = "custom_mesh_raster";
        _rasterShader = device.compile(kernel, _meshFormat, opt);
    } else {
        // ====== Opaque billboard shader ======
        auto vert = RasterStageKernel<PointV2P(PointAppData,
            luisa::compute::Buffer<luisa::float4>,
            luisa::compute::Buffer<luisa::float4>,
            util::CameraData, luisa::uint, float)>{
            [&](Var<PointAppData> app,
                BufferVar<luisa::float4> particles,
                BufferVar<luisa::float4> velocities,
                Var<util::CameraData> camera,
                UInt count,
                Float dt) noexcept {
                set_name("point_vert");

                UInt pid = app.instance_id;

                Float4 particle = particles.read(pid);
                Float3 pos  = particle.xyz();
                Float  size = particle.w;

                Float3 cam_right = camera.right;
                Float3 cam_up    = camera.up;

                Float2 corner = make_float2(app.position.x, app.position.y);
                Float3 world_pos = pos + (cam_right * corner.x + cam_up * corner.y) * size;

                Float4 clip = camera.view_proj * make_float4(world_pos, 1.0f);
                clip = make_float4(clip.x, -clip.y, clip.z, clip.w);

                Float3 billboard_forward = normalize(camera.position - pos);

                Float3 vel = velocities.read(pid).xyz();
                Float3 prev_pos = pos - vel * dt;
                Float2 motion_ndc = camera->project_prev(prev_pos) - camera->project(pos);

                Float4 packed_data = make_float4(clip.w, as<float>(pid), motion_ndc.x, motion_ndc.y);
                Var<PointV2P> vary{clip, packed_data, corner, camera.right, camera.up, billboard_forward};
                return vary.expression();
            }
        };

        auto frag = RasterStageKernel<PointPixelOut(PointV2P, luisa::uint)>{
            [&](Expr<PointV2P> vary, UInt mat_id) noexcept {
                set_name("point_frag");

                Float  linear_depth = vary.packed_data.x;
                UInt   particle_id  = as<uint>(vary.packed_data.y);
                Float2 motion_ndc   = make_float2(vary.packed_data.z, vary.packed_data.w);
                Float mx = clamp(motion_ndc.x * 0.5f + 0.5f, 0.0f, 1.0f);
                Float my = clamp(motion_ndc.y * 0.5f + 0.5f, 0.0f, 1.0f);
                UInt packed = (cast<uint>(mx * 65535.0f) << 16u) | cast<uint>(my * 65535.0f);
                Float  packed_motion = as<float>(packed);
                Float2 corner_uv    = vary.corner_uv;

                $if(Expr{ length_squared(corner_uv)} > 1.0f) {
                    luisa::compute::detail::FunctionBuilder::current()->call(
                        luisa::compute::CallOp::RASTER_DISCARD, {});
                };

                Float r2 = corner_uv.x * corner_uv.x + corner_uv.y * corner_uv.y;
                Float3 local_n = luisa::compute::normalize(
                    make_float3(-corner_uv.x, -corner_uv.y, sqrt(max(0.0f, 1.0f - r2))));
                Float3 world_n = luisa::compute::normalize(
                    vary.cam_right * local_n.x + vary.cam_up * local_n.y + vary.cam_forward * local_n.z);
                Float2 oct_n = render::oct_encode(world_n);

                Float4 depth_out = make_float4(linear_depth, packed_motion, 0.f, 0.f);

                UInt vis_y = (1u << 30u) | particle_id;
                Float4 vis_out = make_float4(as<float>(mat_id), as<float>(vis_y), 0.0f, 0.0f);

                Float4 bary_out = make_float4(Expr{ corner_uv * 0.5f + 0.5f }, oct_n);

                Var<PointPixelOut> result{depth_out, vis_out, bary_out};
                return result.expression();
            }
        };

        auto kernel = RasterKernel<decltype(vert), decltype(frag)>{vert, frag};

        ShaderOption opt;
        opt.name = "point_cloud_raster";
        _rasterShader = device.compile(kernel, _meshFormat, opt);

        // ====== Transparent forward-lit + OIT shader ======
        auto trans_vert = RasterStageKernel<PointTransparentV2P(PointAppData,
            luisa::compute::Buffer<luisa::float4>,
            luisa::compute::Buffer<luisa::float4>,
            util::CameraData, luisa::uint, float)>{
            [&](Var<PointAppData> app,
                BufferVar<luisa::float4> particles,
                BufferVar<luisa::float4> velocities,
                Var<util::CameraData> camera,
                UInt count,
                Float dt) noexcept {
                set_name("point_transparent_vert");

                UInt pid = app.instance_id;

                Float4 particle = particles.read(pid);
                Float3 pos  = particle.xyz();

                Float3 cam_right = camera.right;
                Float3 cam_up    = camera.up;

                Float2 corner = make_float2(app.position.x, app.position.y);
                Float3 world_pos = pos + (cam_right * corner.x + cam_up * corner.y) * particle.w;

                Float  particle_alpha = velocities.read(pid).w;
                Float4 clip = camera.view_proj * make_float4(world_pos, ite(particle_alpha>.00001f, 1.0f, 0.f));
                clip = make_float4(clip.x, -clip.y, clip.z, clip.w);

                Float3 billboard_forward = normalize(camera.position - pos);

                Float4 packed_data = make_float4(clip.w, as<float>(pid), camera.far_clip, particle_alpha);
                Var<PointTransparentV2P> vary{clip, packed_data, corner, camera.right, camera.up, billboard_forward, world_pos};
                return vary.expression();
            }
        };

        auto trans_frag = RasterStageKernel<TransparentPixelOut(PointTransparentV2P,
            luisa::uint,
            luisa::uint,                                  // receive_shadow
            // VoxelGrid expanded
            luisa::compute::Buffer<luisa::half>,
            luisa::compute::Buffer<luisa::uint>,
            luisa::compute::Buffer<luisa::uint>,
            scene::VoxelGridParams,
            // MaterialData
            luisa::compute::Buffer<render::MaterialData>,
            // EnvLight expanded
            luisa::compute::Image<float>,
            luisa::compute::Buffer<float>,
            luisa::compute::Buffer<float>,
            float,
            luisa::uint,
            luisa::uint,
            luisa::float3x3,
            float,
            luisa::compute::Image<float>)>{
            [&](Expr<PointTransparentV2P> vary,
                UInt mat_id,
                UInt receive_shadow,
                // VoxelGrid
                BufferVar<luisa::half> voxel_occupancy,
                BufferVar<luisa::uint> voxel_summary,
                BufferVar<luisa::uint> voxel_occupied_count,
                Var<scene::VoxelGridParams> voxel_params,
                // MaterialData
                BufferVar<render::MaterialData> material_buffer,
                // EnvLight
                ImageFloat envmap,
                BufferVar<float> env_marginal_cdf,
                BufferVar<float> env_conditional_cdf,
                Float env_integral,
                UInt env_width,
                UInt env_height,
                Float3x3 env_rotation,
                Float env_exposure,
                ImageFloat gbuf_depth) noexcept {
                set_name("point_transparent_frag");

                Float  linear_depth = vary.packed_data.x;
                UInt   particle_id  = as<uint>(vary.packed_data.y);
                Float  far_plane    = vary.packed_data.z;
                Float2 corner_uv    = vary.corner_uv;

                // Discard outside unit sphere
                $if(Expr{ length_squared(corner_uv) } > 1.0f) {
                    luisa::compute::detail::FunctionBuilder::current()->call(
                        luisa::compute::CallOp::RASTER_DISCARD, {});
                };

                // Software depth test against opaque G-buffer
                UInt2 pixel = make_uint2(cast<uint>(vary.position.x), cast<uint>(vary.position.y));
                Float opaque_depth = gbuf_depth.read(pixel).x;
                $if(Expr{ opaque_depth > 0.0f & linear_depth > opaque_depth }) {
                    luisa::compute::detail::FunctionBuilder::current()->call(
                        luisa::compute::CallOp::RASTER_DISCARD, {});
                };

                // Sphere normal
                Float r2 = corner_uv.x * corner_uv.x + corner_uv.y * corner_uv.y;
                Float3 local_n = luisa::compute::normalize(
                    make_float3(-corner_uv.x, -corner_uv.y, sqrt(max(0.0f, 1.0f - r2))));
                Float3 world_n = luisa::compute::normalize(
                    vary.cam_right * local_n.x + vary.cam_up * local_n.y + vary.cam_forward * local_n.z);

                Float alpha = vary.packed_data.w;

                // Material albedo
                Var<render::MaterialData> mat = material_buffer.read(mat_id);
                Float3 lit = litFunc(world_n, mat, 
                    envmap, env_marginal_cdf, env_conditional_cdf,
                    env_integral,  env_width, env_height, 
                    env_rotation, env_exposure);

                $if(Expr{ receive_shadow > 0u }) {
                    // Voxel grid shadow (DDA march, expanded params)
                    lit *= render::dda_march_flat(
                        voxel_occupancy, voxel_summary, voxel_occupied_count, voxel_params,
                        Expr{ vary.world_pos + world_n * max(0.001f * linear_depth, 1e-4f) }, world_n, 1e10f);
                };
                
                // McGuire weight: w(z, α) = clamp((1-z/far)^3 * 10^5 * sqrt(α), 1e-5, 1e5)
                Float depth_norm = 1.0f - min(linear_depth / far_plane, 1.0f);
                Float w = clamp(depth_norm * depth_norm * depth_norm * 1e5f * sqrt(max(alpha, 0.0f)), 1e-5f, 1e5f);

                Float4 accum_out = make_float4(alpha * lit * w, alpha * w);
                Float4 reveal_out = make_float4(log(max(1.0f - alpha, 1e-5f)), 0.0f, 0.0f, 0.0f);

                Var<TransparentPixelOut> result{accum_out, reveal_out};
                return result.expression();
            }
        };

        auto trans_kernel = RasterKernel<decltype(trans_vert), decltype(trans_frag)>{trans_vert, trans_frag};

        ShaderOption trans_opt;
        trans_opt.name = "point_cloud_transparent";
        _transparentShader = device.compile(trans_kernel, _meshFormat, trans_opt);
    }
}

void PointCloud::recompile(Device& device) {
    CI_LOG_I("PointCloud: recompiling shaders...");
    _compileRasterShader(device);
}

void PointCloud::_initRasterState(Device& device) {
    _meshFormat = {};

    if (_useCustomMesh) {
        luisa::compute::VertexAttribute attrs[] = {
            {VertexAttributeType::Position, PixelFormat::RGBA32F},
            {VertexAttributeType::Normal,   PixelFormat::RGBA32F},
            {VertexAttributeType::Tangent,  PixelFormat::RG32F},
        };
        _meshFormat.emplace_vertex_stream(attrs);

        _rasterState = RasterState{
            .cull_mode = CullMode::Back,
            .depth_state = DepthState{
                .enable_depth = true,
                .comparison = Comparison::LessEqual,
                .write = true,
            },
            .topology = TopologyType::Triangle,
        };
    } else {
        luisa::compute::VertexAttribute attrs[] = {
            {VertexAttributeType::Position, PixelFormat::RGBA32F},
        };
        _meshFormat.emplace_vertex_stream(attrs);

        _rasterState = RasterState{
            .cull_mode = CullMode::Back,
            .depth_state = DepthState{
                .enable_depth = true,
                .comparison = Comparison::LessEqual,
                .write = true,
            },
            .topology = TopologyType::Triangle,
        };

        // Transparent state: additive blend, no depth write/test
        _transparentRasterState = RasterState{
            .cull_mode = CullMode::Back,
            .blend_state = BlendState{
                .enable_blend = true,
                .op = BlendOp::Add,
                .prim_op = BlendWeight::One,
                .img_op  = BlendWeight::One,
            },
            .depth_state = DepthState{
                .enable_depth = false,
                //.comparison = Comparison::Always,
                .write = false,
            },
            .topology = TopologyType::Triangle,
        };

        luisa::float4 quadVerts[6] = {
            {-1.f, -1.f, 0.f, 0.f},
            {+1.f, -1.f, 0.f, 0.f},
            {-1.f, +1.f, 0.f, 0.f},
            {-1.f, +1.f, 0.f, 0.f},
            {+1.f, -1.f, 0.f, 0.f},
            {+1.f, +1.f, 0.f, 0.f},
        };

        _quadVB = device.create_buffer<luisa::float4>(6u);
        auto stream = device.create_stream();
        stream << _quadVB.copy_from(luisa::span{quadVerts, 6u});
    }
}

// --- RasterBase hooks ---

void PointCloud::onRasterInit(Device& device) {
    _initRasterState(device);
    _compileRasterShader(device);

#if NT_DEBUG_VIZ
    // Debug voxel grid visualizer
    _voxelDebugShader = device.compile<2>(
        [&](ImageFloat output,
            BufferVar<half> occupancy,
            Float3 bounds_min,
            Float3 bounds_max,
            UInt3 res,
            Float cell_size,
            Var<util::CameraData> camera) noexcept {
        set_name("voxel_debug");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();

        Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
        auto ray = camera->generate_ray(ndc);
        Float3 origin = ray->origin();
        Float3 dir = luisa::compute::normalize(ray->direction());

        Float3 inv_dir = 1.0f / luisa::compute::max(luisa::compute::abs(dir), luisa::make_float3(1e-10f));
        Float3 t0 = (bounds_min - origin) * inv_dir;
        Float3 t1 = (bounds_max - origin) * inv_dir;
        Float tmin = luisa::compute::max(luisa::compute::max(luisa::compute::min(t0.x, t1.x), luisa::compute::min(t0.y, t1.y)), luisa::compute::min(t0.z, t1.z));
        Float tmax = luisa::compute::max(luisa::compute::max(luisa::compute::max(t0.x, t1.x), luisa::compute::max(t0.y, t1.y)), luisa::compute::max(t0.z, t1.z));
        tmin = max(tmin, 0.0f);

        Float3 color = luisa::make_float3(1.f, 1.f, 1.f);
        $if(tmin < tmax) {
            Float3 entry = origin + dir * tmin;
            Float3 grid_pos = (entry - bounds_min) / cell_size;
            Int3 cell = make_int3(
                cast<int>(clamp(grid_pos.x, 0.0f, cast<Float>(res.x - 1u))),
                cast<int>(clamp(grid_pos.y, 0.0f, cast<Float>(res.y - 1u))),
                cast<int>(clamp(grid_pos.z, 0.0f, cast<Float>(res.z - 1u))));

            Int3 step = make_int3(
                ite(dir.x > 0.0f, 1, -1),
                ite(dir.y > 0.0f, 1, -1),
                ite(dir.z > 0.0f, 1, -1));

            Float3 next_boundary = make_float3(
                ite(dir.x > 0.0f, (cast<Float>(cell.x) + 1.0f) * cell_size + bounds_min.x,
                                     cast<Float>(cell.x) * cell_size + bounds_min.x),
                ite(dir.y > 0.0f, (cast<Float>(cell.y) + 1.0f) * cell_size + bounds_min.y,
                                     cast<Float>(cell.y) * cell_size + bounds_min.y),
                ite(dir.z > 0.0f, (cast<Float>(cell.z) + 1.0f) * cell_size + bounds_min.z,
                                     cast<Float>(cell.z) * cell_size + bounds_min.z));

            Float3 t_max_local = abs((next_boundary - origin) * inv_dir);
            Float3 t_delta = cell_size * inv_dir;

            UInt max_steps = (res.x + res.y + res.z) * 2u;
            $for(i, max_steps) {
                $if(cast<uint>(cell.x) >= res.x |
                     cast<uint>(cell.y) >= res.y |
                     cast<uint>(cell.z) >= res.z) { $break; };
                $if(cell.x < 0 | cell.y < 0 | cell.z < 0) { $break; };
                $if(t_max_local.x > tmax & t_max_local.y > tmax & t_max_local.z > tmax) { $break; };

                UInt flat = cast<uint>(cell.x)
                          + cast<uint>(cell.y) * res.x
                          + cast<uint>(cell.z) * res.x * res.y;
                UInt occ = occupancy.read(flat);
                $if(occ > 0.0f) {
                    color = luisa::make_float3(1.0f, 0.2f, 0.1f);
                    $break;
                };

                $if(t_max_local.x < t_max_local.y & t_max_local.x < t_max_local.z) {
                    cell.x += step.x;
                    t_max_local.x += t_delta.x;
                } $elif(t_max_local.y < t_max_local.z) {
                    cell.y += step.y;
                    t_max_local.y += t_delta.y;
                } $else {
                    cell.z += step.z;
                    t_max_local.z += t_delta.z;
                };
            };
        };
        output.write(coord, make_float4(color, 1.0f));
    });
#endif
}

bool PointCloud::onRasterExecute(Stream& rasterStream, RasterContext& rc,
                                  const core::FeatureContext& ctx) {
    auto& profiler = util::Profiler::instance();
    uint w = ctx.width(), h = ctx.height();

    // Lazy-bind VoxelGrid
    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return false;
    }

#if NT_DEBUG_VIZ
    // Debug: visualize voxel grid occupancy (red = occupied)
    if (_voxelDebug) {
        profiler.set_pass("Voxel/VoxelDebug");
        rasterStream << _voxelDebugShader(
            ctx.renderTarget,
            _voxelGrid->occupancy(),
            _voxelGrid->params().bounds_min,
            _voxelGrid->params().bounds_max,
            _voxelGrid->resolution(),
            _voxelGrid->params().cell_size,
            ctx.frame.camera
        ).dispatch(w, h);
        return false;  // skip merge
    }
#endif

    // Draw instanced particles
    luisa::vector<RasterMesh> scene;
    if (_useCustomMesh) {
        VertexBufferView vbv{_customMeshVB};
        RasterMesh mesh(
            luisa::span<const VertexBufferView, 1>{&vbv, 1},
            _customMeshVertexCount,
            _particleCount,
            0u
        );
        scene.push_back(std::move(mesh));
    } else {
        VertexBufferView vbv{_quadVB};
        RasterMesh mesh(
            luisa::span<const VertexBufferView, 1>{&vbv, 1},
            6u,
            _particleCount,
            0u
        );
        scene.push_back(std::move(mesh));
    }

    auto drawCmd = std::move(
        _rasterShader(
            _positionsBuffer,
            _velocitiesBuffer,
            ctx.frame.camera,
            _particleCount,
            ctx.frame.deltaTime,
            material_id
        )
    ).draw(
        std::move(scene),
        _meshFormat,
        Viewport{0u, 0u, w, h},
        _rasterState,
        &rc.depthBuffer(),
        rc.rasterDepth(),
        rc.rasterVis(),
        rc.rasterBary()
    );
    profiler.set_pass("PC/render");
    rasterStream << std::move(drawCmd);

    return true;
}

bool PointCloud::onRasterExecuteTransparent(Stream& rasterStream, RasterContext& rc,
                                             const core::FeatureContext& ctx) {
    if (_useCustomMesh) return false;  // Custom mesh transparent not yet supported
    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return false;
    }

    uint w = ctx.width(), h = ctx.height();

    // Build instanced scene (same as opaque)
    luisa::vector<RasterMesh> scene;
    VertexBufferView vbv{_quadVB};
    RasterMesh mesh(
        luisa::span<const VertexBufferView, 1>{&vbv, 1},
        6u,
        _particleCount,
        0u
    );
    scene.push_back(std::move(mesh));

    auto voxel_res = _voxelGrid->gpu_resources();
    auto env_res = ctx.pipeline.env_gpu_resources();
    scene::VoxelGridParams voxel_params{
        voxel_res.bounds_min, voxel_res.bounds_max,
        voxel_res.resolution, voxel_res.cell_size,
        voxel_res.summary_resolution
    };

    auto drawCmd = std::move(
        _transparentShader(
            _positionsBuffer,
            _velocitiesBuffer,
            ctx.frame.camera,
            _particleCount,
            ctx.frame.deltaTime,
            material_id,
            _receiveShadow ? 1u : 0u,
            // VoxelGrid expanded
            voxel_res.occupancy,
            voxel_res.summary,
            voxel_res.occupied_count,
            voxel_params,
            // MaterialData
            ctx.pipeline.material()->buffer(),
            // EnvLight expanded
            env_res.envmap,
            env_res.env_marginal_cdf,
            env_res.env_conditional_cdf,
            env_res.env_integral,
            env_res.env_width,
            env_res.env_height,
            env_res.env_rotation,
            ctx.pipeline.lightsampler()->env_exposure(),
            ctx.frame.gbufDepth
        )
    ).draw(
        std::move(scene),
        _meshFormat,
        Viewport{0u, 0u, w, h},
        _transparentRasterState,
        nullptr,               // no depth buffer (software depth test)
        rc.oitAccum(),         // MRT 0
        rc.oitLogReveal()      // MRT 1
    );

    auto& profiler = util::Profiler::instance();
    profiler.set_pass("PC/transparent");
    rasterStream << std::move(drawCmd);

    return true;
}

void PointCloud::onVoxelize(Stream& computeStream, const core::FeatureContext& ctx) {
    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return;
    }

    auto& profiler = util::Profiler::instance();
    if (_castShadow) {
        profiler.set_pass("PC/voxelize");
        _voxelGrid->voxelize(computeStream, _positionsBuffer, _particleCount, _shadowRadiusScale);
    }
}

bool PointCloud::shouldExecute() const {
    return RasterBase::shouldExecute() && _particleCount > 0u;
}

// --- Data upload ---

void PointCloud::upload_positions(Stream& stream,
                                   const luisa::vector<luisa::float4>& positions,
                                   const luisa::vector<luisa::float4>& velocities,
                                   uint count) {
    _particleCount = min(count, _maxParticles);
    stream << _positionsBuffer.view(0u, _particleCount).copy_from(positions.data())
           << _velocitiesBuffer.view(0u, _particleCount).copy_from(velocities.data());
}

void PointCloud::upload_positions(Stream& stream,
                                   BufferView<luisa::float4> positions,
                                   BufferView<luisa::float4> velocities,
                                   uint count) {
    _particleCount = min(count, _maxParticles);
    stream << _positionsBuffer.view(0u, _particleCount).copy_from(positions)
           << _velocitiesBuffer.view(0u, _particleCount).copy_from(velocities);
}

// --- Serialization ---

void PointCloud::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool en = j.value("enabled", false);
    setEnabled(en);
    _particleCount = j.value("pntcount", _particleCount);
    _shadowDilation = j.value("shadowdilate", _shadowDilation);
    _shadowRadiusScale = j.value("shadowradius", _shadowRadiusScale);
    if (j.value("transparent", false))
        setBlendMode(BlendMode::Transparent);
    _receiveShadow = j.value("receiveShadow", _receiveShadow);
    _castShadow = j.value("castShadow", _castShadow);
}

nlohmann::json PointCloud::toJson() const {
    nlohmann::json file;
    file["enabled"] = enabled();
    file["shutter"] = _particleCount;
    file["shadowdilate"] = _shadowDilation;
    file["shadowradius"] = _shadowRadiusScale;
    file["transparent"] = isTransparent();
    file["receiveShadow"] = _receiveShadow;
    file["castShadow"] = _castShadow;
    return file;
}

// --- UI ---

void PointCloud::drawUi() {
    if (ImGui::CollapsingHeader("Point Cloud")) {
        ImGui::ScopedId scpId("pointcloud");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::Text("Particles: %u / %u", _particleCount, _maxParticles);
            bool trans = isTransparent();
            if (ImGui::Checkbox("Transparent", &trans))
                setBlendMode(trans ? BlendMode::Transparent : BlendMode::Opaque);
            ImGui::Checkbox("Receive Shadow", &_receiveShadow);
            ImGui::Checkbox("Cast Shadow", &_castShadow);
            ImGui::Checkbox("Voxel Debug", &_voxelDebug);
            int dilate = static_cast<int>(_shadowDilation);
            if (ImGui::SliderInt("Shadow Dilate", &dilate, 0, 8))
                _shadowDilation = static_cast<luisa::uint>(dilate);
            ImGui::SliderFloat("Shadow Radius", &_shadowRadiusScale, 1.0f, 20.0f, "%.1fx");
        }
    }
}

} // namespace newtype::feature
