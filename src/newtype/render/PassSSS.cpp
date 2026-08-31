#include "newtype/render/PassSSS.h"
#include "newtype/render/BSSRDF.h"
#include "newtype/render/Shading.h"
#include "newtype/render/ProceduralTrace.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>

namespace newtype {
namespace core {

using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

//==============================================================================
// Release
//==============================================================================
void PassSSS::release() {
    _geom = nullptr;
    (void)_sssRadiance.release();
    (void)_sssParamsBuf.release();
}

//==============================================================================
// createImages — HALF4 simultaneous-access image + params buffer
//==============================================================================
void PassSSS::createImages(luisa::compute::Device& device, uint width, uint height) {
    // HALF4: 3 channels of demodulated radiance (per-channel strobing) + alpha.
    // simultaneous_access=true matches _accumBuffer/_specularBuffer so it's
    // safe to read in the shade shader on Renderer::stream().
    _sssRadiance = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _sssRadiance.set_name("sss_radiance");

    _sssParamsBuf = device.create_buffer<SSSParams>(1u);

    // Default params — match plan §1.2 / §7 defaults
    // radiusQuantile=0.5 (median) instead of plan's 0.999: the 99.9th percentile
    // Burley radius is ~14x larger than the median, which exceeds typical scene
    // object sizes (1-unit sphere etc.) and causes the probe ray to miss the
    // actual surface. Lower quantile = smaller probe disk = more hits, at the
    // cost of truncating the long tail (~50% of energy at q=0.5). The x3
    // channel-strobe scale below does NOT compensate for this; SSS regions
    // will appear dimmer than physically correct until a future Phase adds
    // a 1/radiusQuantile normalization term.
    _sssParamsCpu.radiusQuantile       = 0.5f;
    _sssParamsCpu.scatterDistanceScale = 1.0f;
    _sssParamsCpu.nLocalFloor          = 1e-3f;
}

//==============================================================================
// _populateParams — copy CPU staging struct to GPU buffer
//==============================================================================
void PassSSS::_populateParams(luisa::compute::CommandList& cmdlist) noexcept {
    // _sssParamsCpu is a class member because cmdlist << copy_from(&local)
    // is deferred; a stack-local struct would be freed before the GPU reads it.
    cmdlist << _sssParamsBuf.copy_from(&_sssParamsCpu);
}

//==============================================================================
// compile
//==============================================================================
void PassSSS::compile(luisa::compute::Device& device,
                      scene::Geometry& geom,
                      const SurfaceResolverPoly& resolver) {
    _geom = &geom;

    using Vertex   = MeshShape::Vertex;
    using Triangle = luisa::compute::Triangle;

    _probeShader = device.compile<2>([&](
        BufferVar<SSSParams> sss_params,
        ImageFloat            sss_radiance,
        ImageFloat            gbuf_depth,
        ImageUInt             gbuf_vis,
        ImageFloat            gbuf_bary_motion,
        UInt                  frame_count,
        AccelVar              accel,
        Var<util::CameraData> camera,
        Var<SceneGeometryResources> scene,
        BindlessVar           vertex_bindless,
        BindlessVar           tex_bindless,
        Var<LightSamplingResources> lights
#if NT_ENABLE_PROCEDURAL
        , BindlessVar proc_bindless
#endif
    ) noexcept {
        set_name("SSS_Probe");
        set_block_size(16u, 16u, 1u);

        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(any(coord >= resolution)) { $return(); };

        // Clear to zero by default. Write only happens inside the gated SSS path.
        sss_radiance.write(coord, make_float4(0.0f));

        auto p = sss_params.read(0u);

        Float depth = gbuf_depth.read(coord).x;
        UInt4 vis   = gbuf_vis.read(coord);
        UInt  inst_id = vis.x;

        // Sky / no hit — nothing to do (already zeroed).
        $if(inst_id == ~0u) { $return(); };

        UInt  prim_id       = vis.y & 0x3FFFFFFFu;
        Bool  is_glass      = (vis.y >> 31u) > 0u;
        Bool  is_point      = ((vis.y >> 30u) & 1u) > 0u;
        Bool  is_procedural = ((vis.y >> 29u) & 1u) > 0u;

        // SSS probe only handles opaque mesh Subsurface pixels. Glass has its
        // own PSR; point primitives have no SSS; procedural geometry is out of
        // Phase 1 scope.
        $if(is_glass | is_point | is_procedural) { $return(); };

        UInt4 inst_data = scene.instance_buffer.read(inst_id);
        Var<MaterialData> material = scene.material_buffer.read(Expr{ inst_data.y & 0xFFu });

        // Only run on Subsurface materials with flatness > 0 (HK gate) that
        // are NOT thin-wall (diffuse_trans > 0 keeps the HK/transmission
        // BSDF lobes as the sole model — a volumetric probe here would
        // double-count with the thin transmission lobe).
        $if(material.type != 6u) { $return(); };
        $if(material.flatness <= 0.0f) { $return(); };
        $if(material.diffuse_trans > 0.0f) { $return(); };

        Float4 barymotion = gbuf_bary_motion.read(coord);
        Float2 bary = barymotion.xy();

        Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
        auto ray = camera->generate_ray(ndc);
        Float3 wo = -normalize(ray->direction());
        Float3 world_pos = ray->origin() + ray->direction() * depth;

        // Resolve surface at x0 (camera pixel) — needed for albedo, geo_ns.
        // Pass zeros for screen-uv/wh since this is full-res and textures are
        // sampled inside resolve_surface via the per-pixel uv.
        auto instance_xform = scene.instance_transform_buffer.read(inst_id);
        SurfaceData surface = resolve_surface_from_instance(
            resolver, vertex_bindless, tex_bindless,
            inst_data, prim_id, bary,
            scene.material_buffer, wo,
            instance_xform,
            cast<Float>(frame_count),
            (make_float2(coord) + 0.5f) / make_float2(resolution),
            resolution.x, resolution.y);

        Float3 geo_ns   = surface.geo_ns;
        Float3 facing_ns = ite(dot(wo, geo_ns) < 0.0f, -geo_ns, geo_ns);

        // === Burley setup ===
        // Use tinted albedo (albedo * attenuation) — matches HK's effective BRDF
        // albedo so the SSS profile picks up the absorption tint. Without this,
        // a white albedo + red attenuation would produce untinted SSS (red tint
        // appears only in HK reflection, not in the probe contribution).
        Float3 tinted_albedo = surface.albedo * surface.attenuation;
        Float3 s_rgb = bssrdf_scatter_distance(tinted_albedo)
                     * material.attenuation_distance
                     * p.scatterDistanceScale;

        // Per-frame random channel strobing — 1 of 3 per frame, x3 scale.
        // Frame-coherent so accum fills in over 3+ frames.
        UInt seed = util::xxhash32(make_uint3(
            coord.x * 1973u + coord.y * 9277u,
            frame_count,
            0x9E3779B9u));
        UInt channel = seed % 3u;
        Float s_ch = ite(channel == 0u, s_rgb.x,
                      ite(channel == 1u, s_rgb.y, s_rgb.z));

        // === Sample radius, angle, axis ===
        Float radius_max = bssrdf_sample_radius(p.radiusQuantile, s_ch);

        // Advance RNG for radius / phi / axis
        seed = util::lcg_ui(seed);
        Float u_radius = util::uniform_uint_to_float(seed);
        seed = util::lcg_ui(seed);
        Float u_phi    = util::uniform_uint_to_float(seed);
        seed = util::lcg_ui(seed);
        Float u_axis   = util::uniform_uint_to_float(seed);

        Float radius = bssrdf_sample_radius(u_radius, s_ch);
        // Reject samples outside the radius_max quantile — keeps tMin/tMax
        // bounded and matches the reference GLSL's clamp behavior.
        $if(radius > radius_max) { $return(); };

        Float phi = u_phi * 2.0f * pi;

        // Build orthonormal basis (columns: t, b, n) — make_orthonormal_basis
        // returns a 3x3 whose columns are (b1, b2, normal).
        Float3x3 tnb = make_orthonormal_basis(facing_ns);
        // NaN guard: initialize to defaults before branching (Risk #18).
        Float3 axis_n = facing_ns;
        Float3 axis_t = tnb[0];
        Float3 axis_b = tnb[1];
        Float  axis_weight = 0.5f;
        $if(u_axis < 0.5f) {
            axis_n = facing_ns; axis_t = tnb[0]; axis_b = tnb[1]; axis_weight = 0.5f;
        } $elif(u_axis < 0.75f) {
            axis_n = tnb[0];    axis_t = facing_ns; axis_b = tnb[1]; axis_weight = 0.25f;
        } $else {
            axis_n = tnb[1];    axis_t = facing_ns; axis_b = tnb[0]; axis_weight = 0.25f;
        };

        // === Probe ray (polar disk per ReSTIR-SSS reference) ===
        // Origin sits on a virtual disk of radius `radius_max` perpendicular
        // to `axis_n` and offset by `radius_max` along `axis_n`. The probe
        // ray travels along -axis_n. Each point on the source disk maps to
        // a unique exit point on the surface at chord distance up to 2*radius_max.
        Float3 origin = world_pos
                      + radius_max * axis_n
                      + (cos(phi) * radius) * axis_t
                      + (sin(phi) * radius) * axis_b;
        Float3 dir = -axis_n;
        Float sphere_frac = sqrt(radius_max * radius_max - radius * radius);
        Float tMin = max(radius_max - sphere_frac, 1e-4f);  // Risk #6: floor at 1e-4
        Float tMax = radius_max + sphere_frac;
        auto probe_ray = make_ray(origin, dir, tMin, tMax);
        auto probe_hit = render::trace_closest(accel, probe_ray
#if NT_ENABLE_PROCEDURAL
            , proc_bindless
#endif
        );

        $if(probe_hit->miss()) { $return(); };

        // Same-instance heuristic — Phase 1 limitation (Risk #12). Adjacent
        // instances sharing the SSS material are rejected; Phase 2 will relax.
        $if(probe_hit.inst != inst_id) { $return(); };

        // === Resolve geometry at exit point x2 ===
        // probe_hit.inst == inst_id verified above — reuse x0's inst_data +
        // instance_xform (saves 2 buffer reads per SSS pixel).
        UInt   x2_prim  = probe_hit.prim;
        Float2 x2_bary  = probe_hit.bary;
        auto   x2_tri   = vertex_bindless.buffer<Triangle>(inst_data.w).read(x2_prim);
        auto   x2_v0    = vertex_bindless.buffer<Vertex>(inst_data.z).read(x2_tri.i0);
        auto   x2_v1    = vertex_bindless.buffer<Vertex>(inst_data.z).read(x2_tri.i1);
        auto   x2_v2    = vertex_bindless.buffer<Vertex>(inst_data.z).read(x2_tri.i2);
        Float  x2_b2    = 1.0f - x2_bary.x - x2_bary.y;
        Float3 x2_obj_p = x2_v0->position() * x2_b2
                        + x2_v1->position() * x2_bary.x
                        + x2_v2->position() * x2_bary.y;
        Float3 x2_obj_n = normalize(
                            x2_v0->normal() * x2_b2
                          + x2_v1->normal() * x2_bary.x
                          + x2_v2->normal() * x2_bary.y);
        Float3 x2_pos = make_float3(instance_xform * make_float4(x2_obj_p, 1.0f));
        Float3 x2_ns  = transform_normal(instance_xform, x2_obj_n);

        // Surface-to-disk Jacobian (Bug 4): tilted exit surface needs |dot(axis_n, x2_ns)|.
        Float n_local = abs(dot(axis_n, x2_ns));
        $if(n_local < p.nLocalFloor) { $return(); };

        // === Sample one light (single-technique NEE, no MIS Phase 1) ===
        $if(lights.emissive_count > 0u) {
            // Alias-table light selection (power-weighted).
            UInt lseed = util::xxhash32(make_uint4(
                coord.x * 1973u + coord.y * 9277u,
                frame_count,
                x2_prim,
                0x85EBCA6Bu));
            lseed = util::lcg_ui(lseed);
            Float u_select = util::uniform_uint_to_float(lseed);
            lseed = util::lcg_ui(lseed);
            Float u_tri_x  = util::uniform_uint_to_float(lseed);
            lseed = util::lcg_ui(lseed);
            Float u_tri_y  = util::uniform_uint_to_float(lseed);

            Float u_scaled = u_select * cast<Float>(lights.emissive_count);
            UInt  aidx = cast<uint>(u_scaled);
            aidx = min(aidx, lights.emissive_count - 1u);
            Var<AliasEntry> entry = lights.alias_table.read(aidx);
            UInt light_idx = ite(u_scaled - cast<Float>(aidx) < entry.pdf,
                                 entry.triangle_index, entry.alias_index);
            light_idx = min(light_idx, lights.emissive_count - 1u);

            auto tri_light = lights.triangle_lights.read(light_idx);
            auto verts     = lights.triangle_vertices.read(light_idx);

            // Uniform area-weighted triangle sample (matches the GI/shade pattern).
            Float su = sqrt(u_tri_x);
            Float2 bary_light = make_float2(1.0f - su, u_tri_y * su);
            Float3 light_point = bary_light.x * verts.v0
                               + bary_light.y * verts.v1
                               + (1.0f - bary_light.x - bary_light.y) * verts.v2;

            Float3 light_n_unaligned = normalize(
                cross(verts.v1 - verts.v0, verts.v2 - verts.v0));
            // Align light normal to face x2.
            Float3 to_light_x2 = light_point - x2_pos;
            Float3 light_n = ite(dot(light_n_unaligned, to_light_x2) >= 0.0f,
                                 light_n_unaligned, -light_n_unaligned);

            Float3 to_light  = light_point - x2_pos;
            Float  dist_sq   = max(dot(to_light, to_light), 1e-6f);
            Float3 light_dir = to_light * rsqrt(dist_sq);
            Float  cos_x2    = max(0.0f, dot(x2_ns, light_dir));

            $if(cos_x2 > 0.0f) {
                // Self-intersection offset at x2 (Risk #9).
                Float offset = max(0.001f * sqrt(dist_sq), 1e-4f);
                Float3 origin2 = x2_pos + x2_ns * offset;
                auto s_ray = make_ray(origin2, light_dir, 0.0f, sqrt(dist_sq) - offset);
                Bool occluded = render::trace_occluded(accel, s_ray
#if NT_ENABLE_PROCEDURAL
                    , proc_bindless
#endif
                );

                $if(!occluded) {
                    Float cos_light = max(0.0f, dot(light_n, -light_dir));

                    Float3 Le_rgb = tri_light->emission();
                    // Extract chosen channel — per-channel strobing (Phase 1).
                    Float Le = ite(channel == 0u, Le_rgb.x,
                                ite(channel == 1u, Le_rgb.y, Le_rgb.z));

                    // Fresnel at x0 (outgoing) and x2 (incoming) — Bug 3.
                    // Schlick R0 from material IOR (dielectric form).
                    Float ior_ratio = (material.ior - 1.0f) / (material.ior + 1.0f);
                    Float R0 = sqr(ior_ratio);
                    Float F_x0 = 1.0f - fresnel_schlick(R0, abs(dot(facing_ns, wo)));
                    Float F_x2 = 1.0f - fresnel_schlick(R0, cos_x2);

                    // Geometry term (light → x2). The probe ray already accounts
                    // for the x0→x2 transport via the BSSRDF disk sample.
                    Float geom_visible = Le * cos_x2 * cos_light / dist_sq;

                    // Source pdf in solid angle ≈ pdf_select * area_to_solid_angle.
                    // For an area-sampled light: p_area = tri_light.pdf / tri_light.area
                    // (kUniformLightSampling path uses emissive_count_inv / area).
                    Float source_pdf = tri_light.pdf / tri_light.area;
                    source_pdf = max(source_pdf, 1e-10f);

                    // === Corrected single-sample estimator (§3) ===
                    // R(r_p) cancels between S and p_x2; remaining denom = n_local.
                    // The x3 scale compensates for the 1/3 channel strobe.
                    // Per-channel scalar — write only the chosen slot this frame.
                    Float sss_c = (3.0f / axis_weight)
                                * F_x0 * F_x2 * (1.0f / pi)
                                * geom_visible
                                / (n_local * source_pdf);

                    // Write only to the chosen channel slot; others get 0 this
                    // frame. ReLAX temporal accum fills them in.
                    Float3 out_rgb = make_float3(
                        ite(channel == 0u, sss_c, 0.0f),
                        ite(channel == 1u, sss_c, 0.0f),
                        ite(channel == 2u, sss_c, 0.0f));

                    sss_radiance.write(coord, make_float4(out_rgb, 1.0f));
                };
            };
        };
    });
}

//==============================================================================
// renderProbe
//==============================================================================
void PassSSS::renderProbe(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
    if (_paramDirty) {
        _populateParams(cmdlist);
        _paramDirty = false;
    }

    auto& ls = ctx.lightSampler;
    LightSamplingResources light_resources {
        ls.triangle_buffer(), ls.vertex_buffer(),
        ls.alias_table(), ls.emissive_triangle_count(),
        ls.total_power_inv(), ls.emissive_count_inv(),
        ls.instance_to_light_base()
    };

    SceneGeometryResources scene_resources {
        _geom->instance_buffer(),
        _geom->instance_transform_buffer(),
        _geom->instance_transform_prev_buffer(),
        ctx.materialPool.buffer(),
        ctx.materialPool.simKeyBuffer()
    };

    cmdlist << _probeShader(
        _sssParamsBuf,
        _sssRadiance,
        ctx.gbufDepth,
        ctx.gbufVis,
        ctx.gbufBaryMotion,
        ctx.frameCount,
        _geom->tlas(),
        ctx.camera,
        scene_resources,
        _geom->vertex_bindless(),
        ctx.materialPool.textures(),
        light_resources
#if NT_ENABLE_PROCEDURAL
        , *ctx.procBindless
#endif
    ).dispatch(ctx.width, ctx.height);
}

void PassSSS::drawUi() {
    if (ImGui::CollapsingHeader("SubSurfaceScattering")) {
        _paramDirty |= ImGui::DragFloat("Distance Scale", &_sssParamsCpu.scatterDistanceScale, .005f, .5f, 2.5f);
        _paramDirty |= ImGui::DragFloat("Radius Quantile", &_sssParamsCpu.radiusQuantile, .005f, .2f, 1.5f);
    }
}

} // namespace core
} // namespace newtype
