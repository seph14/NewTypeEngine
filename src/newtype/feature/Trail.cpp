#include "newtype/feature/Trail.h"
#include "newtype/feature/RasterContext.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/util/Camera.h"
#include "newtype/util/Profiler.h"
#include "newtype/render/Shading.h"
#include "newtype/render/DDAMarch.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/core/BindingGroups.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {

using namespace luisa;
using namespace luisa::compute;

Trail::Trail(Device& device, uint max_segments, uint material_id,
             luisa::float3 bounds_min, luisa::float3 bounds_max, uint voxel_resolution)
    : RasterBase(device)
    , _maxPoints(max_segments + 1u)
    , material_id(material_id)
    , _boundsMin(bounds_min)
    , _boundsMax(bounds_max)
    , _voxelResolution(voxel_resolution)
{
    _pointBuffer    = _device.create_buffer<TrailPoint>(_maxPoints);
    _positionBuffer = _device.create_buffer<luisa::float4>(_maxPoints);
}

Trail::~Trail() = default;

void Trail::onRasterInit(Device& device) {
    // Mesh format: position only (corner offset + t parameter)
    _meshFormat = {};
    luisa::compute::VertexAttribute attrs[] = {
        {VertexAttributeType::Position, PixelFormat::RGBA32F},
    };
    _meshFormat.emplace_vertex_stream(attrs);

    // No culling — ribbons visible from both sides
    _rasterState = RasterState{
        .cull_mode = CullMode::None,
        .depth_state = DepthState{
            .enable_depth = true,
            .comparison = Comparison::LessEqual,
            .write = true,
        },
        .topology = TopologyType::Triangle,
    };

    // Transparent state: additive blend, no depth write/test
    _transparentRasterState = RasterState{
        .cull_mode = CullMode::None,
        .blend_state = BlendState{
            .enable_blend = true,
            .op = BlendOp::Add,
            .prim_op = BlendWeight::One,
            .img_op  = BlendWeight::One,
        },
        .depth_state = DepthState{
            .enable_depth = false,
            .write = false,
        },
        .topology = TopologyType::Triangle,
    };

    // Ribbon quad: 6 vertices encoding (side, t) pairs
    // Each segment is a quad with 4 corners: (-1,0), (1,0), (-1,1), (1,1)
    luisa::float4 quadVerts[6] = {
        {-1.f,  0.f, 0.f, 0.f},  // BL
        {+1.f,  0.f, 0.f, 0.f},  // BR
        {-1.f,  1.f, 0.f, 0.f},  // TL
        {-1.f,  1.f, 0.f, 0.f},  // TL
        {+1.f,  0.f, 0.f, 0.f},  // BR
        {+1.f,  1.f, 0.f, 0.f},  // TR
    };

    _quadVB = device.create_buffer<luisa::float4>(6u);
    auto stream = device.create_stream();
    stream << _quadVB.copy_from(luisa::span{quadVerts, 6u});

    // Compile shaders with current hooks
    _compileShaders(device);

    // Unpack shader: points → flat position buffer for voxelize (applies radius_scale)
    _unpackShader = device.compile<1>(
        [&](BufferVar<TrailPoint> points,
            BufferVar<luisa::float4> positions,
            UInt point_count,
            Float radius_scale) noexcept {
            set_name("trail_unpack");
            UInt idx = dispatch_id().x;
            $if(idx < point_count) {
                Var<TrailPoint> pt = points.read(idx);
                positions.write(idx, make_float4(pt.pos_width.xyz(), pt.pos_width.w * radius_scale));
            };
        }
    );
}

void Trail::_compileShaders(Device& device) {
    // References to hook members for Callable lambda capture
    auto& widthFn = _customWidthFn;
    auto& shadeFn = _customShadeFn;

    Callable vertFunc = [&](UInt seg_id, UInt seg_count,
        Bool is_end, Float side, Float t, Float miter_limit,
        Float3 campos, BufferVar<TrailPoint> points) noexcept {
        // Read the two points forming this segment
        Var<TrailPoint> pt0 = points.read(seg_id);
        Var<TrailPoint> pt1 = points.read(seg_id + 1u);

        // Current segment direction
        Float3 dir_curr = luisa::compute::normalize(
            pt1.pos_width.xyz() - pt0.pos_width.xyz());

        // Interpolate position along segment
        Float3 pos = luisa::compute::lerp(pt0.pos_width.xyz(), pt1.pos_width.xyz(), t);

        // Width: custom or default
        Float width = widthFn
            ? widthFn(t, seg_id, points)
            : detail::default_trail_width(t, seg_id, points);

        // Camera-facing ribbon normal (used for shading)
        Float3 to_cam = luisa::compute::normalize(campos - pos);
        Float3 ribbon_n = luisa::compute::normalize(luisa::compute::cross(dir_curr, to_cam));

        // --- Miter joint computation (adjacent points) ---
        Bool has_prev = seg_id > 0u;
        Bool has_next = seg_id < seg_count - 1u;
        UInt pt_prev = luisa::compute::max(seg_id, 1u) - 1u;

        UInt pt_next = luisa::compute::min(seg_id + 2u, seg_count);

        Float3 dir_prev = luisa::compute::normalize(
            pt0.pos_width.xyz() - points.read(pt_prev).pos_width.xyz());
        Float3 dir_next = luisa::compute::normalize(
            points.read(pt_next).pos_width.xyz() - pt1.pos_width.xyz());

        Float3 dir_in = luisa::compute::ite(is_end, dir_curr, dir_prev);
        Float3 dir_out = luisa::compute::ite(is_end, dir_next, dir_curr);
        Bool   has_adj = luisa::compute::ite(is_end, has_next, has_prev);

        Float3 n_in = luisa::compute::normalize(luisa::compute::cross(dir_in, to_cam));
        Float3 n_out = luisa::compute::normalize(luisa::compute::cross(dir_out, to_cam));
        Float3 miter = luisa::compute::normalize(n_in + n_out);

        Float miter_scale = 1.0f / luisa::compute::max(luisa::compute::dot(miter, n_out), 1e-4f);
        miter_scale = luisa::compute::min(miter_scale, miter_limit);

        Float3 world_pos = pos
            + luisa::compute::ite(has_adj, miter, ribbon_n)
            * width * luisa::compute::ite(has_adj, miter_scale, 1.0f) * side;
        return compose(world_pos, ribbon_n);
    };

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
        // Pre-compute env radiance — user can use or ignore
        Float3 env_radiance = render::eval_envmap_radiance(
            world_n, envmap, env_width, env_height,
            env_rotation, env_exposure);

        // Custom or default shade
        return shadeFn
            ? shadeFn(world_n, mat, env_radiance)
            : detail::default_trail_shade(world_n, mat, env_radiance);
    };

    // ====== Opaque vertex shader: ribbon billboarding with miter joints ======
    auto vert = RasterStageKernel<TrailV2P(TrailAppData,
        luisa::compute::Buffer<TrailPoint>,
        util::CameraData,
        luisa::uint,
        float)>{
        [&](Var<TrailAppData> app,
            BufferVar<TrailPoint> points,
            Var<util::CameraData> camera,
            UInt point_count,
            Float miter_limit) noexcept {
            set_name("trail_vert");

            UInt seg_id = app.instance_id;
            UInt seg_count = point_count - 1u;
            Float t = app.position.y;  // 0 or 1
            Var data = vertFunc(seg_id, seg_count,
                Expr{ t > .5f}, app.position.x, t, miter_limit,
                camera.position, points);

            Float4 clip = camera.view_proj * make_float4(data.get<0>(), 1.0f);
            clip = make_float4(clip.x, -clip.y, clip.z, clip.w);

            Float4 packed = make_float4(clip.w, as<float>(seg_id), t, 0.0f);
            Var<TrailV2P> vary{ clip, packed, data.get<1>() };
            return vary.expression();
        }
    };

    // ====== Opaque pixel shader: G-buffer output ======
    auto frag = RasterStageKernel<PointPixelOut(TrailV2P, luisa::uint)>{
        [&](Expr<TrailV2P> vary, UInt mat_id) noexcept {
            set_name("trail_frag");

            Float linear_depth = vary.packed_data.x;
            UInt  seg_id       = as<uint>(vary.packed_data.y);
            Float lerp_t       = vary.packed_data.z;

            Float4 depth_out = make_float4(linear_depth, 0.f, 0.f, 0.f);

            UInt vis_y = (1u << 30u) | seg_id;
            Float2 vis_xy = make_float2(as<float>(mat_id), as<float>(vis_y));
            Float4 vis_out = make_float4(vis_xy, 0.0f, 0.0f);

            Float2 bary = make_float2(lerp_t, 0.5f);
            Float2 oct_n = render::oct_encode(luisa::compute::normalize(vary.ribbon_normal));
            Float4 bary_out = make_float4(bary, oct_n);

            Var<PointPixelOut> result{depth_out, vis_out, bary_out};
            return result.expression();
        }
    };

    auto kernel = RasterKernel<decltype(vert), decltype(frag)>{vert, frag};
    ShaderOption opt;
    opt.name = "trail_raster";
    _rasterShader = device.compile(kernel, _meshFormat, opt);

    // ====== Transparent vertex shader (same billboarding, passes world_pos) ======
    auto trans_vert = RasterStageKernel<TrailTransparentV2P(TrailAppData,
        luisa::compute::Buffer<TrailPoint>,
        util::CameraData, luisa::uint, float)>{
        [&](Var<TrailAppData> app,
            BufferVar<TrailPoint> points,
            Var<util::CameraData> camera,
            UInt point_count,
            Float miter_limit) noexcept {
            set_name("trail_transparent_vert");

            UInt seg_id = app.instance_id;
            UInt seg_count = point_count - 1u;
            Float t = app.position.y;  // 0 or 1
            Var data = vertFunc(seg_id, seg_count,
                Expr{ t > .5f }, app.position.x, t, miter_limit,
                camera.position, points);

            Float4 clip = camera.view_proj * make_float4(data.get<0>(), 1.0f);
            clip = make_float4(clip.x, -clip.y, clip.z, clip.w);

            Float4 packed = make_float4(clip.w, as<float>(seg_id), t, camera.far_clip);
            Var<TrailTransparentV2P> vary{ clip, packed, data.get<1>(), data.get<0>() };
            return vary.expression();
        }
    };

    // ====== Transparent pixel shader: forward lighting + OIT output ======
    auto trans_frag = RasterStageKernel<TransparentPixelOut(TrailTransparentV2P,
        luisa::uint,
        float,
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
        [&](Expr<TrailTransparentV2P> vary,
            UInt mat_id,
            Float alpha,
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
            set_name("trail_transparent_frag");

            Float linear_depth = vary.packed_data.x;
            Float far_plane    = vary.packed_data.w;

            // Software depth test against opaque G-buffer
            UInt2 pixel = make_uint2(cast<uint>(vary.position.x), cast<uint>(vary.position.y));
            Float opaque_depth = gbuf_depth.read(pixel).x;
            $if(Expr{ opaque_depth > 0.0f & linear_depth > opaque_depth }) {
                luisa::compute::detail::FunctionBuilder::current()->call(
                    luisa::compute::CallOp::RASTER_DISCARD, {});
            };

            Float3 world_n = luisa::compute::normalize(vary.ribbon_normal);

            // Material albedo
            Var<render::MaterialData> mat = material_buffer.read(mat_id);
            Float3 lit = litFunc(world_n, mat,
                envmap, env_marginal_cdf, env_conditional_cdf,
                env_integral, env_width, env_height,
                env_rotation, env_exposure);

            // Voxel grid shadow (expanded params)
            $if(Expr{ receive_shadow > 0u }) {
                lit *= render::dda_march_flat(
                    voxel_occupancy, voxel_summary, voxel_occupied_count, voxel_params,
                    Expr{ vary.world_pos + world_n * max(0.001f * linear_depth, 1e-4f) }, world_n, 1e10f);
            };

            // McGuire weight
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
    trans_opt.name = "trail_transparent";
    _transparentShader = device.compile(trans_kernel, _meshFormat, trans_opt);
}

void Trail::recompile(Device& device) {
    CI_LOG_I("Trail: recompiling shaders...");
    _compileShaders(device);
}

bool Trail::onRasterExecute(Stream& rasterStream, RasterContext& rc,
                             const core::FeatureContext& ctx) {
    uint w = ctx.width(), h = ctx.height();

    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return false;
    }

    uint seg_count = segmentCount();
    luisa::vector<RasterMesh> scene;
    VertexBufferView vbv{_quadVB};
    RasterMesh mesh(
        luisa::span<const VertexBufferView, 1>{&vbv, 1},
        6u,
        seg_count,
        0u
    );
    scene.push_back(std::move(mesh));

    auto drawCmd = std::move(
        _rasterShader(
            _pointBuffer,
            ctx.frame.camera,
            _pointCount,
            _miterLimit,
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

    auto& profiler = util::Profiler::instance();
    profiler.set_pass("Trail/render");
    rasterStream << std::move(drawCmd);

    return true;
}

bool Trail::onRasterExecuteTransparent(Stream& rasterStream, RasterContext& rc,
                                         const core::FeatureContext& ctx) {
    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return false;
    }

    uint w = ctx.width(), h = ctx.height();
    uint seg_count = segmentCount();

    luisa::vector<RasterMesh> scene;
    VertexBufferView vbv{_quadVB};
    RasterMesh mesh(
        luisa::span<const VertexBufferView, 1>{&vbv, 1},
        6u,
        seg_count,
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
            _pointBuffer,
            ctx.frame.camera,
            _pointCount,
            _miterLimit,
            material_id,
            _alpha,
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
        nullptr,
        rc.oitAccum(),
        rc.oitLogReveal()
    );

    auto& profiler = util::Profiler::instance();
    profiler.set_pass("Trail/transparent");
    rasterStream << std::move(drawCmd);

    return true;
}

void Trail::onVoxelize(Stream& computeStream, const core::FeatureContext& ctx) {
    if (!_voxelGrid) {
        _voxelGrid = ctx.pipeline.voxelGrid();
        if (!_voxelGrid || !_voxelGrid->created()) return;
    }

    auto& profiler = util::Profiler::instance();

    if (!_castShadow) return;

    profiler.set_pass("Trail/unpack");
    computeStream << _unpackShader(
        _pointBuffer,
        _positionBuffer,
        _pointCount,
        _shadowRadiusScale
    ).dispatch(_pointCount);

    profiler.set_pass("Trail/voxelize");
    _voxelGrid->voxelize(computeStream, _positionBuffer, _pointCount, 1.0f);
}

bool Trail::shouldExecute() const {
    return RasterBase::shouldExecute() && segmentCount() > 0u;
}

// --- Data upload ---
void Trail::uploadPoints(Stream& stream,
    const vector<TrailPoint>& points,
    uint point_count) {
    _pointCount = min(point_count, _maxPoints);
    stream << _pointBuffer.view(0u, _pointCount).copy_from(points.data());
}

void Trail::uploadPoints(Stream& stream,
                            const BufferView<TrailPoint>& points,
                            uint point_count) {
    _pointCount = min(point_count, _maxPoints);
    stream << _pointBuffer.view(0u, _pointCount).copy_from(points);
}

// --- Serialization ---

void Trail::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    setEnabled(j.value("enabled", false));
    _pointCount = j.value("pointCount", _pointCount);
    _shadowDilation = j.value("shadowdilate", _shadowDilation);
    _shadowRadiusScale = j.value("shadowradius", _shadowRadiusScale);
    _miterLimit = j.value("miterLimit", _miterLimit);
    _alpha = j.value("alpha", _alpha);
    if (j.value("transparent", false) || _alpha < 1.0f)
        setBlendMode(BlendMode::Transparent);
    _receiveShadow = j.value("receiveShadow", _receiveShadow);
    _castShadow = j.value("castShadow", _castShadow);
}

nlohmann::json Trail::toJson() const {
    nlohmann::json file;
    file["enabled"] = enabled();
    file["pointCount"] = _pointCount;
    file["shadowdilate"] = _shadowDilation;
    file["shadowradius"] = _shadowRadiusScale;
    file["miterLimit"] = _miterLimit;
    file["alpha"] = _alpha;
    file["transparent"] = isTransparent();
    file["receiveShadow"] = _receiveShadow;
    file["castShadow"] = _castShadow;
    return file;
}

// --- UI ---

void Trail::drawUi() {
    if (ImGui::CollapsingHeader("Trail")) {
        ImGui::ScopedId scpId("trail");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::Text("Points: %u / %u (%u segments)", _pointCount, _maxPoints, segmentCount());
            bool trans = isTransparent();
            if (ImGui::Checkbox("Transparent", &trans))
                setBlendMode(trans ? BlendMode::Transparent : BlendMode::Opaque);
            if (isTransparent()) {
                if (ImGui::SliderFloat("Alpha", &_alpha, 0.0f, 1.0f, "%.2f"))
                    setBlendMode(BlendMode::Transparent);
            }
            ImGui::Checkbox("Receive Shadow", &_receiveShadow);
            ImGui::Checkbox("Cast Shadow", &_castShadow);
            int dilate = static_cast<int>(_shadowDilation);
            if (ImGui::SliderInt("Shadow Dilate", &dilate, 0, 8))
                _shadowDilation = static_cast<luisa::uint>(dilate);
            ImGui::SliderFloat("Shadow Radius", &_shadowRadiusScale, 1.0f, 20.0f, "%.1fx");
            ImGui::SliderFloat("Miter Limit", &_miterLimit, 1.0f, 10.0f, "%.1f");
        }
    }
}

} // namespace newtype::feature
