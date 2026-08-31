#include "newtype/render/PassGI.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"
#include "newtype/render/ProceduralTrace.h"
#include "newtype/render/MaterialSimilarity.h"
#include "newtype/util/AccumulationTime.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/Log.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>

namespace newtype::core {
	using namespace luisa;
	using namespace luisa::compute;
	using namespace newtype::scene;
	using namespace newtype::render;

	void PassGI::compile(luisa::compute::Device& device, const SurfaceResolverPoly& resolver, bool checkerboard) {
		_checkerboard = checkerboard;
		compileImpl(device, resolver, /*resolverOnly=*/false);
	}

	void PassGI::recompileCallables(luisa::compute::Device& device, const SurfaceResolverPoly& resolver) {
		compileImpl(device, resolver, /*resolverOnly=*/true);
	}

	void PassGI::compileImpl(luisa::compute::Device& device, const SurfaceResolverPoly& resolver, bool resolverOnly) {
	        using Vertex = MeshShape::Vertex;
	        using Triangle = luisa::compute::Triangle;

	        //==========================================================================
	        // GI Initial Sampling kernel (BRDF ray -> secondary hit -> evaluate direct at x2)
	        //==========================================================================
	        _giInitialShader = device.compile<2>([&](
	            BufferVar<GIParams> params,
	            BufferVar<GIReservoir> gi_reservoir_buffer,
	            ImageFloat gbuf_depth,
	            ImageUInt  gbuf_vis,
	            ImageFloat gbuf_bary_motion,
	            ImageUInt  seed_image,
	            AccelVar   accel,
	            Var<util::CameraData> camera,
	            Var<SceneGeometryResources> scene,
	            BindlessVar vertex_bindless,
	            BindlessVar tex_bindless,
	            Var<LightSamplingResources> lights,
	            Var<EnvLightResources> env,
	            Float      env_exposure_val,
	            BufferVar<render::PresampledCandidate> presample_env_tiles,
	            UInt       presample_env_total_entries,
	            UInt       cbField,
	            ImageFloat glass_throughput,
	            UInt       oneBounce,
	            UInt       giScale,
	            UInt       giSuppressAwayFromLight,
	            Float      giSuppressAwayFromLightThresh,
	            UInt       hasTransparentShadowCasters
#if NT_ENABLE_PROCEDURAL
		        ,
		        BindlessVar proc_bindless
		    #endif
	            ) noexcept {
	            set_name("GI_Initial");
	            set_block_size(16u, 16u, 1u);
	            auto p = params.read(0u);

	            // GI bounce callable: trace ray -> reconstruct hit -> NEE
	            // Captures all scene/light resources from enclosing kernel scope.
	            Callable gi_bounce = [&](Float3 origin, Float3 dir, Float3 wo, UInt seed, UInt receiver_inst) noexcept {
	                Var<GIBounceResult> result;
	                result.px = 0.0f; result.py = 0.0f; result.pz = 0.0f;
	                result.nx = 0.0f; result.ny = 0.0f; result.nz = 0.0f;
	                result.tx = 1.0f; result.ty = 0.0f; result.tz = 0.0f;
	                result.tw = 1.0f;
	                result.rad_x = 0.0f; result.rad_y = 0.0f; result.rad_z = 0.0f;
	                result.wi_x = dir.x; result.wi_y = dir.y; result.wi_z = dir.z;
	                result.hit_inst = 0u; result.hit_inst_data_y = 0u; result.hit_prim = 0u;
	                result.hit_bary_u = 0.0f; result.hit_bary_v = 0.0f;
	                result.hit_t = 0.0f;
	                result.seed = seed;
	                result.valid = 0u;

	                auto ray = make_ray(origin, dir, 0.001f, 1e10f);
	                auto hit = render::trace_closest(accel, ray
#if NT_ENABLE_PROCEDURAL
	                    , proc_bindless
#endif
	                );

	                // Reject self-instance hits (BRDF ray hitting adjacent face of same mesh).
// Without this, a closed mesh picks up illumination from its own bright side
// via grazing-angle BRDF rays that self-hit at short distance, producing the
// "GI leak" pattern that matches the mesh's own illuminated regions.
$if(!hit->miss() & Expr{ hit.inst != receiver_inst }) {
	                    UInt hit_inst = hit.inst;
	                    UInt hit_prim = hit.prim;        // mesh: triangle idx; proc: AABB idx (= resolver inst_id)
	                    Float hit_t = hit.committed_ray_t;

	                    Float3 pos;
	                    Float3 ns;
	                    Float3 tangent;
	                    Float  tangent_w;
	                    Float2 hit_bary;                 // mesh: hit.bary; proc: hit.local_bary
	                    UInt   hit_material_layers;      // mesh: hit_inst_data.y; proc: proc.material_layers
	                    UInt4  hit_inst_data;            // mesh-only; shared between vertex interp + x2_surface resolve
#if NT_ENABLE_PROCEDURAL
	                    UInt hit_local_tri = def(0u);    // proc only; 0 for mesh
	                    Bool hit_is_proc   = hit.is_procedural;

	                    // Procedural convention (matches glass PSR - PipelineInit.cpp:1861-1867):
	                    // hit_inst is the TLAS slot shared across every proc AABB (out of range for
	                    // instance_buffer); hit_prim is the AABB index = inst_id for the proc resolver.
	                    $if(hit_is_proc) {
	                        hit_bary         = hit.local_bary;
	                        hit_local_tri    = hit.local_tri;
	                        Var<scene::ProcInstanceData> proc =
	                            proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(hit_prim);
	                        hit_material_layers = proc.material_layers;
	                        pos = ray->origin() + ray->direction() * hit_t;
	                        UInt packed_proc_prim = (1u << 29u) | hit_local_tri;
	                        // Dispatcher handles type 1/2/3/0 + kProcDoubleSided face-normal override
	                        // (Shading.h:1273-1299). Replaces the inline type-switch that missed the
	                        // double-sided branch and indexed proc buffers at the wrong slot.
	                        ns = reconstruct_procedural_normal(
	                            proc_bindless, hit_prim, packed_proc_prim, pos, hit_bary);
	                        tangent = compute::make_float3(1.0f, 0.0f, 0.0f);
	                        tangent_w = 1.0f;
	                    } $else {
#endif
	                        hit_inst_data      = scene.instance_buffer.read(hit_inst);
	                        hit_bary           = hit.bary;
	                        hit_material_layers = hit_inst_data.y;
	                        auto tri = vertex_bindless.buffer<Triangle>(hit_inst_data.w).read(hit_prim);
	                        auto v0 = vertex_bindless.buffer<Vertex>(hit_inst_data.z).read(tri.i0);
	                        auto v1 = vertex_bindless.buffer<Vertex>(hit_inst_data.z).read(tri.i1);
	                        auto v2 = vertex_bindless.buffer<Vertex>(hit_inst_data.z).read(tri.i2);
	                        Float b2 = 1.0f - hit_bary.x - hit_bary.y;
	                        pos = v0->position() * b2 + v1->position() * hit_bary.x + v2->position() * hit_bary.y;
	                        ns = normalize(v0->normal() * b2 + v1->normal() * hit_bary.x + v2->normal() * hit_bary.y);
	                        Float4 t_raw = v0->tangent() * b2 + v1->tangent() * hit_bary.x + v2->tangent() * hit_bary.y;
	                        tangent = normalize(t_raw.xyz());
	                        tangent_w = t_raw.w;
#if NT_ENABLE_PROCEDURAL
	                    };
#endif

	                    Var<MaterialData> mat = scene.material_buffer.read(Expr{ hit_material_layers & 0xFFu });
	                    //Bool is_unlit = mat.type == 12u;

	                    // Resolve x2 surface (textures + polymorphic dispatch) — NEE consumers
	                    // below route through x2_* locals so dynamic albedo textures and custom
	                    // callables appear in GI bounces. NOTE: mesh-only path; procedural x2
	                    // hits are unhandled (NT_ENABLE_PROCEDURAL=0 currently). See
	                    // docs/gi-bounce-shading.md "Out of scope".
	                    SurfaceData x2_surface;
#if NT_ENABLE_PROCEDURAL
	                    $if(hit_is_proc) {
	                        UInt packed_prim = (1u << 29u) | hit_local_tri;
	                        x2_surface = resolve_procedural_surface_textured(
	                            resolver, proc_bindless, tex_bindless,
	                            hit_prim, packed_prim,
	                            pos, wo,
	                            scene.material_buffer,
	                            hit_bary,
	                            0.0f, make_float2(0.0f), 0u, 0u);
	                    } $else {
#endif
	                        x2_surface = resolve_surface_from_instance(
	                            resolver, vertex_bindless, tex_bindless,
	                            hit_inst_data, hit_prim, hit_bary,
	                            scene.material_buffer, wo,
	                            scene.instance_transform_buffer.read(hit_inst),
	                            0.0f, make_float2(0.0f), 0u, 0u);
#if NT_ENABLE_PROCEDURAL
	                    };
#endif
	                    Float3 x2_albedo    = x2_surface.albedo;
	                    Float  x2_roughness = x2_surface.roughness;
	                    Float  x2_metallic  = x2_surface.metallic;
	                    Float  x2_ior       = x2_surface.ior;
	                    UInt   x2_mat_type  = x2_surface.bsdf_type;
	                    // Complex-IOR fields for FrComplex dispatch in NEE/p_hat.
	                    Float3 x2_conductor_eta = x2_surface.attenuation;
	                    Float3 x2_conductor_k   = x2_surface.conductor_k;

	                    // NEE: multi-candidate RIS at x2 (ReSTIR-GI Gap #3 fix).
	                    // Stream kGiInitialCandidateCount triangle candidates + kGiInitialEnvCandidateCount env
	                    // candidates. Each evaluates an unshadowed MC estimate; RIS picks a winner. Only the
	                    // winner traces one shadow ray (same RT budget as the prior single-light NEE).
	                    // The returned radiance is the RIS-unbiased estimator of L_direct(x2):
	                    //   radiance = (w_sum / M) * (MC_sel / p_hat_sel) * visibility
	                    // which lowers variance ~1/M on low-roughness metal GI without adding bias.
	                    Float3 radiance = def(make_float3(0.0f));
	                    $if(Expr{ mat.type == 12u } &mat.meta < 0.5f) {
	                        radiance = x2_albedo;
	                    } $else {
	                        // RIS state (inner to this bounce; not propagated to the GI reservoir)
	                        Float ris_w_sum   = def(0.0f);
	                        UInt  ris_M       = def(0u);
	                        Float3 sel_MC     = def(make_float3(0.0f));
	                        Float  sel_p_hat  = def(0.0f);
                        Float3 sel_dir    = def(make_float3(0.0f, 1.0f, 0.0f));
                        Float  sel_dist   = def(0.0f);  // 0 = env (infinite), >0 = triangle

                        // Hoisted NEE BSDF (bit-identical to the legacy scalar
                        // overloads): the field list mirrors their internal
                        // construction verbatim — bsdf_type pinned to 0, layered
                        // params zeroed, dummy tangent frame. The glass gate stays
                        // external (x2_mat_type at the call sites) because the
                        // const& overloads gate on the pinned bsdf_type instead.
                        // Built once per bounce instead of once per candidate.
                        MaterialBSDF x2_nee_bsdf{
                            x2_albedo, x2_roughness, x2_metallic, x2_ior,
                            0.f, 0.f,
                            0.f, 0.5f,
                            0.f, 1.3f, 0.f,
                            0.f, 0.f,
                            luisa::compute::make_float3(1.f, 0.f, 0.f), 1.f,
                            0u, 0.f, 0.f, 0.f, 0.f,
                            x2_conductor_eta, 0.f, 1.f,
                            x2_conductor_k};
                        x2_nee_bsdf.build_lobe_list();
                        // Same normal the const& evaluate overloads receive below.
                        x2_nee_bsdf.precompute_tangent_rotation(ns);
	                    
	                        // ======== Triangle candidates (alias-table power-weighted) ========
	                        $if(lights.emissive_count > 0u) {
	                            $for(c, kGiInitialCandidateCount) {
	                                UInt cseed = util::xxhash32(make_uint2(seed, c * 2654435761u + 1u));
	                                Float u_select = util::uniform_uint_to_float(cseed);
	                                cseed = util::lcg_ui(cseed);
	                                Float u_tri_x = util::uniform_uint_to_float(cseed);
	                                cseed = util::lcg_ui(cseed);
	                                Float u_tri_y = util::uniform_uint_to_float(cseed);
	                                Float u_scaled = u_select * cast<float>(lights.emissive_count);
	                                UInt idx = cast<uint>(u_scaled);
	                                idx = min(idx, lights.emissive_count - 1u);
	                                Var<AliasEntry> entry = lights.alias_table.read(idx);
	                                UInt light_idx = ite(u_scaled - cast<float>(idx) < entry.pdf,
	                                    entry.triangle_index, entry.alias_index);
	                                light_idx = min(light_idx, lights.emissive_count - 1u);
	                                auto verts = lights.triangle_vertices.read(light_idx);
	                                Float su = sqrt(u_tri_x);
	                                Float2 bary_light = make_float2(1.0f - su, u_tri_y * su);
	                                Float3 light_point = bary_light.x * verts.v0
	                                    + bary_light.y * verts.v1
	                                    + (1.0f - bary_light.x - bary_light.y) * verts.v2;
	                                Float3 to_light = light_point - pos;
	                                Float light_dist = luisa::compute::length(to_light);
	                                Float3 light_dir = to_light * (1.0f / light_dist);
	                                Float cos_x2 = luisa::compute::max(0.0f, luisa::compute::dot(ns, light_dir));
	                                // Reject back-facing and too-close lights before RIS (preserves the
	                                // original 0.05f min-distance guard against firefly-grade radiance)
	                                $if(cos_x2 > 0.0f & Expr{ light_dist > 0.05f }) {
                                    auto tri_light = lights.triangle_lights.read(light_idx);
                                    Float source_pdf;
	                                    if constexpr (kUniformLightSampling) {
	                                        source_pdf = lights.emissive_count_inv / tri_light.area;
	                                    } else {
	                                        source_pdf = tri_light.pdf / tri_light.area;
	                                    }
	                                    source_pdf = luisa::compute::max(source_pdf, 1e-10f);
	                                    // Pre-derive geometry terms the lean helper skips.
	                                    Float3 light_normal = tri_light->normal();
	                                    Float cos_light = max(0.0f, dot(light_normal, -light_dir));
	                                    Float dist_sq = light_dist * light_dist;
                                    Float3 direct = def(make_float3(0.0f));
                                    $if(x2_mat_type != 3u) {
                                        direct = evaluate_direct_illuminance_with_geometry(
                                            x2_nee_bsdf,
                                            tri_light->emission(), light_dir, dist_sq, cos_x2, cos_light,
                                            ns, wo);
                                    };
	                                    Float3 MC = direct / source_pdf;
	                                    Float p_hat_c = luminance(MC);
	                                    $if(p_hat_c > 1e-8f) {
	                                        ris_w_sum = ris_w_sum + p_hat_c;
	                                        ris_M = ris_M + 1u;
	                                        UInt accept_seed = util::xxhash32(make_uint3(seed, c + 7919u, 0x9E3779B9u));
	                                        Float u_acc = util::uniform_uint_to_float(accept_seed);
	                                        $if(u_acc * ris_w_sum < p_hat_c) {
	                                            sel_MC = MC;
	                                            sel_p_hat = p_hat_c;
	                                            sel_dir = light_dir;
	                                            sel_dist = light_dist;
	                                        };
	                                    };
	                                };
	                            };
	                        };
	                    
	                        // ======== Env candidates (from PassDI presampled env tiles) ========
	                        $if(Expr{ env.env_integral > 0.0f } & Expr{ presample_env_total_entries > 0u }) {
	                            $for(ec, kGiInitialEnvCandidateCount) {
	                                UInt eseed = util::xxhash32(make_uint3(seed, ec + 0xabcdefu, 0x1234u));
	                                UInt entry_idx = eseed % presample_env_total_entries;
	                                auto pc = presample_env_tiles.read(entry_idx);
	                                Float3 env_dir = uv_to_direction(pc.bary_u, pc.bary_v,
	                                    env.env_width, env.env_height, env.env_rotation);
	                                Float3 env_rad = eval_envmap_from_uv(pc.bary_u, pc.bary_v,
	                                    env.envmap, env.env_width, env.env_height, env_exposure_val);
	                                Float cos_x2 = luisa::compute::max(0.0f, luisa::compute::dot(ns, env_dir));
	                                $if(cos_x2 > 0.0f) {
                                    Float3 contrib = def(make_float3(0.0f));
                                    $if(x2_mat_type != 3u) {
                                        contrib = evaluate_env_contribution(
                                            x2_nee_bsdf, env_rad, env_dir, ns, wo);
                                    };
	                                    // MC estimate: contrib / env_pdf_solid_angle = contrib * inv_source_pdf
	                                    Float3 MC = contrib * pc.inv_source_pdf;
	                                    Float p_hat_c = luminance(MC);
	                                    $if(p_hat_c > 1e-8f) {
	                                        ris_w_sum = ris_w_sum + p_hat_c;
	                                        ris_M = ris_M + 1u;
	                                        UInt accept_seed = util::xxhash32(make_uint3(seed, ec + 0xfedcbau, 0x9E3779B9u));
	                                        Float u_acc = util::uniform_uint_to_float(accept_seed);
	                                        $if(u_acc * ris_w_sum < p_hat_c) {
	                                            sel_MC = MC;
	                                            sel_p_hat = p_hat_c;
	                                            sel_dir = env_dir;
	                                            sel_dist = 0.0f;  // marker: env (infinite distance)
	                                        };
	                                    };
	                                };
	                            };
	                        };
	                    
	                        // ======== Resolve RIS winner: trace one shadow ray ========
                                $if(ris_M > 0u & Expr{ sel_p_hat > 1e-8f }) {
                                    Float s_offset = max(0.001f * hit_t, 1e-4f);
                                    // sel_dist == 0 marks an env winner; tmax = infinity (1e10f) for env shadow
                                    Float shadow_tmax = ite(sel_dist > 0.0f, sel_dist - s_offset, 1e10f);
                                    auto shadow_ray = make_ray(
                                        pos + ns * s_offset + sel_dir * (0.25f * s_offset),
                                        sel_dir, s_offset, shadow_tmax);
	                            Bool visible = def(true);
	                            $if(hasTransparentShadowCasters == 0u) {
	                                // Opaque scene: any-hit occlusion is exact and skips the
	                                // closest-hit traversal + blocker material resolve below.
	                                visible = !render::trace_occluded(accel, shadow_ray
	                    #if NT_ENABLE_PROCEDURAL
	                                    , proc_bindless
	                    #endif
	                                );
	                            } $else {
	                                auto shadow_hit = render::trace_closest(accel, shadow_ray
	                    #if NT_ENABLE_PROCEDURAL
	                                    , proc_bindless
	                    #endif
	                                );
	                                visible = shadow_hit->miss();
	                                $if(!visible) {
	                                    UInt4 s_inst_data = scene.instance_buffer.read(shadow_hit.inst);
	                                    Var<MaterialData> s_mat = scene.material_buffer.read(Expr{ s_inst_data.y & 0xFFu });
	                                    visible = (s_mat.type == 5u) |
	                                        is_alpha_cutout(s_mat, vertex_bindless, tex_bindless, s_inst_data, shadow_hit.prim, shadow_hit.bary);
	                                };
	                            };
	                            $if(visible) {
	                                Float3 rad_x2 = (ris_w_sum / cast<Float>(ris_M)) * (sel_MC / max(sel_p_hat, 1e-8f));
	                                radiance = rad_x2;
	                            };
	                        };
	                    
	                        // Advance seed so the next call (second bounce) gets decorrelated randomness.
	                        // Per-call LCG step; per-candidate decorrelation is handled via xxhash32 above.
	                        seed = util::lcg_ui(seed);
	                    };
	                    result.px = pos.x; result.py = pos.y; result.pz = pos.z;
	                    result.nx = ns.x; result.ny = ns.y; result.nz = ns.z;
	                    result.tx = tangent.x; result.ty = tangent.y; result.tz = tangent.z;
	                    result.tw = tangent_w;
	                    result.rad_x = radiance.x; result.rad_y = radiance.y; result.rad_z = radiance.z;
	                    result.hit_inst = hit_inst;
	                    result.hit_inst_data_y = hit_material_layers;
	                    result.hit_prim = hit_prim;
	                    result.hit_bary_u = hit_bary.x;
	                    result.hit_bary_v = hit_bary.y;
	                    result.hit_t = hit_t;
	                    result.seed = seed;
#if NT_ENABLE_PROCEDURAL
	                    // Bit-pack: bit 0 = valid, bit 1 = is_procedural, bits 2-31 = local_tri.
	                    result.valid = 1u | (ite(hit_is_proc, 1u, 0u) << 1u) | (hit_local_tri << 2u);
#else
	                    result.valid = 1u;
#endif
	                };

	                return result;
	            };

	            UInt2 rsv       = dispatch_id().xy();
	            UInt2 rsv_res   = dispatch_size().xy();
	            UInt2 coord;
	            UInt2 resolution;
	            UInt  pixel_index;
	            if (_checkerboard) {
	                // When half-res (giScale > 1), initial pass runs at native half-res
	                // without checkerboard conversion - the upsample handles checkerboard mapping
	                $if(giScale > 1u) {
	                    coord = rsv;
	                    resolution = rsv_res;
	                    pixel_index = rsv.y * rsv_res.x + rsv.x;
	                } $else {
	                    coord = make_uint2(rsv.x << 1u, rsv.y);
	                    coord.x = coord.x + ((coord.y + cbField) & 1u);
	                    resolution = make_uint2(rsv_res.x * 2u, rsv_res.y);
	                    pixel_index = rsv.y * rsv_res.x + rsv.x;
	                };
	                $if(rsv.x >= rsv_res.x | rsv.y >= rsv_res.y) { $return(); };
	            } else {
	                coord = rsv;
	                resolution = rsv_res;
	                pixel_index = rsv.y * rsv_res.x + rsv.x;
	                $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };
	            }

	            // Map half-res coord to full-res G-buffer coord (identity when giScale==1)
	            UInt fullres_w = resolution.x * giScale;
	            UInt fullres_h = resolution.y * giScale;
	            UInt2 gbuf_coord = min(make_uint2(coord.x * giScale + giScale / 2u, coord.y * giScale + giScale / 2u),
	                                   make_uint2(fullres_w - 1u, fullres_h - 1u));

	            // Init GI reservoir to invalid
	            Var<GIReservoir> gi_r;
	            gi_r.px = 0.0f; gi_r.py = 0.0f; gi_r.pz = 0.0f;
	            gi_r.packed_normal   = 0u;
	            gi_r.packed_radiance = 0u;
				//gi_r.rx = 0.f; gi_r.ry = 0.f; gi_r.rz = 0.f;
				gi_r.weight_sum = 0.0f;
	            gi_r.target_pdf = 0.0f;
	            gi_r.packed_meta = 0u;

	            UInt4  vis = gbuf_vis.read(gbuf_coord);
	            UInt   inst_id = vis.x;
	            Bool   is_point = ((vis.y >> 30u) & 1u) > 0u;
	            Bool   is_procedural_gi = ((vis.y >> 29u) & 1u) > 0u;

	            // Glass pixels get GI initial (1 fresh sample/frame) but skip temporal
	            // and spatial reuse to prevent weight accumulation blowout.
	            // Point pixels skip GI entirely. Procedural pixels now participate.
	            $if(inst_id != ~0u & !is_point) {
	                UInt   prim_id = vis.y & 0x3FFFFFFFu;
	                Bool   is_glass = (vis.y >> 31u) > 0u;
	                Float  depth = gbuf_depth.read(gbuf_coord).x;
	                Float2 bary = gbuf_bary_motion.read(gbuf_coord).xy();

	                // Reconstruct x1 (primary surface hit)
	                auto   ray = camera->generate_ray(Expr{
	                    (make_float2(gbuf_coord) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)}) * 2.0f - 1.0f
	                });
	                // For glass pixels, use the actual background surface position
	                // (from PSR's stored inst/prim/bary) instead of the virtual depth
	                // projection onto the camera ray -- avoids BRDF rays originating
	                // from a displaced position that can hit the glass sphere.
	                UInt4 inst_data = scene.instance_buffer.read(inst_id);
	                Float3 wo = -normalize(ray->direction());

	                // Resolve surface: procedural builds from material directly,
	                // mesh uses bindless vertex interpolation + texture sampling.
	                Float2 screen_uv = (make_float2(gbuf_coord) + 0.5f) / make_float2(resolution);
	                SurfaceData surface;
#if NT_ENABLE_PROCEDURAL
	                $if(is_procedural_gi) {
	                    // Recompute bary at unjittered pixel center for stable texture sampling.
	                    auto gi_ray_unjit = camera->generate_ray(Expr{
	                        (make_float2(gbuf_coord) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)}) * 2.0f - 1.0f - camera->jitter
	                    });
	                    $if(!is_glass) {
	                        bary = reconstruct_unjittered_bary_procedural(
	                            proc_bindless, inst_id, prim_id, bary,
	                            gi_ray_unjit->origin(), gi_ray_unjit->direction());
	                    };
	                    surface = resolve_procedural_surface_textured(
	                        resolver, proc_bindless, tex_bindless,
	                        inst_id, prim_id,
	                        ray->origin() + ray->direction() * depth, wo,
	                        scene.material_buffer,
	                        bary,
	                        0.0f, screen_uv, resolution.x, resolution.y);
	                } $else {
#endif
	                surface = resolve_surface_from_instance(
	                    resolver, vertex_bindless, tex_bindless,
	                    inst_data, prim_id, bary,
	                    scene.material_buffer, wo,
	                    scene.instance_transform_buffer.read(inst_id),
	                    0.0f, screen_uv, resolution.x, resolution.y);
#if NT_ENABLE_PROCEDURAL
	                };
#endif

	                // For glass PSR: use actual background surface position.
	                // The ray-based position lands on the camera ray at virtual depth, which is inside
	                // the glass sphere -- GI bounces would start inside the glass.
	                // (surface.position is already world-space after resolve_surface_from_instance.)
	                Float3 world_pos = ite(is_glass,
	                    surface.position,
	                    ray->origin() + ray->direction() * depth);

	                // DEBUG leak test: optionally skip GI on surfaces facing away from first light.
	                // Used to test if cabinet-back-face-like surfaces (dark, no direct light) are
	                // the source of the bright-scene speck via some leak path.
	                Bool skip_gi_debug = def(false);
	                $if(giSuppressAwayFromLight != 0u & lights.emissive_count > 0u) {
	                    auto first_light_verts = lights.triangle_vertices.read(0u);
	                    Float3 light_centroid = (first_light_verts.v0 + first_light_verts.v1 + first_light_verts.v2) * (1.0f / 3.0f);
	                    Float3 to_light = light_centroid - world_pos;
	                    Float  light_dist = luisa::compute::length(to_light);
	                    Float3 light_dir = to_light * (1.0f / max(light_dist, 1e-6f));
	                    Float  facing_light = luisa::compute::dot(surface.ns, light_dir);
	                    skip_gi_debug = facing_light < giSuppressAwayFromLightThresh;
	                };

	                // Glass attenuation (computed once, shared by all initial samples)
	                Float glass_att_gi = ite(is_glass, luminance(glass_throughput.read(gbuf_coord).xyz()), 1.0f);

	                // Only trace BRDF ray for non-delta, non-pure-unlit materials
	                // Glass pixels: RIS-combine multiple initial samples (P4).
	                // Non-glass: single sample (no extra cost).
	                // Unlit+GI (meta>=0.5) participates as Lambertian diffuse.
	                auto material = scene.material_buffer.read(Expr{ inst_data.y & 0xFFu });
	                $if(surface.bsdf_type != 3u
	                    & !(surface.bsdf_type == 12u & material.meta < 0.5f)
                    & !skip_gi_debug) {
	                    UInt sample_count = ite(is_glass, kGiGlassInitialSamples, 1u);
	                    Float ris_w_sum = def(0.0f);
	                    UInt  ris_M = def(0u);
	                    Float3 sel_pos = def(make_float3(0.0f));
	                    Float3 sel_nrm = def(make_float3(0.0f));
	                    Float3 sel_rad = def(make_float3(0.0f));
	                    Float  sel_target_pdf = def(0.0f);
	                    UInt base_seed = seed_image.read(gbuf_coord).x;

	                    // Runtime bound (sample_count: 1 for opaque,
	                    // kGiGlassInitialSamples for glass) — same DXC anti-unroll
	                    // rationale as the DI/GI spatial loops: a constant trip count
	                    // lets DXC unroll this ~600-line kernel body 2x.
	                    $for(sample_idx, 0u, sample_count) {
	                        // Per-sample seed: offset by sample_idx to decorrelate
	                        UInt seed = util::xxhash32(make_uint3(base_seed, coord.x + sample_idx * 1337u, coord.y));
	                        UInt brdf_seed = util::xxhash32(make_uint3(seed, coord.x, coord.y + sample_idx * 31u));
	                        UInt coord_hash = util::xxhash32(make_uint2(coord.x + sample_idx * 17u, coord.y));
	                        auto pcg = util::pcg2d(make_uint2(brdf_seed, coord_hash));
	                        Float2 u_brdf = make_float2(
	                            util::uniform_uint_to_float(pcg.x),
	                            util::uniform_uint_to_float(pcg.y));
	                        seed = util::lcg_ui(seed);

	                        MaterialBSDF bsdf_mis = surface.make_bsdf();
	                        Float brdf_pdf = def(0.0f);
	                        Float3 wi = bsdf_mis.sample(wo, surface.ns, u_brdf, brdf_pdf);

	                        $if(brdf_pdf > 1e-6f & Expr{ dot(wi, surface.ns) } > 1e-4f) {
	                            Float offset = max(0.001f * depth, 1e-4f);
	                            auto bounce1 = gi_bounce(world_pos + surface.ns * offset, wi, -wi, seed, inst_id);

	                            $if((bounce1.valid & 1u) != 0u) {
	                                Float3 x2_pos = bounce1->pos();
	                                Float3 x2_ns = bounce1->nrm();
	                                Float3 x2_tangent = bounce1->tangent();
	                                Float  x2_tangent_w = bounce1.tw;
	                                Float3 x2_radiance = bounce1->rad();
	                                Var<MaterialData> x2_material = scene.material_buffer.read(Expr{ bounce1.hit_inst_data_y & 0xFFu });
	                                Bool x2_is_unlit = x2_material.type == 12u;

	                                // ======== Second bounce: x2 -> x3 ========
	                                $if(oneBounce == 0u) {
	                                    $if(x2_material.type != 3u & !(x2_is_unlit & x2_material.meta < 0.5f)) {
	                                        UInt seed2 = util::xxhash32(make_uint3(bounce1.seed, coord.x + 7919u, coord.y + sample_idx * 53u));
	                                        UInt coord_hash2 = util::xxhash32(make_uint2(coord.x + 7919u, coord.y + sample_idx * 53u));
	                                        auto pcg2 = util::pcg2d(make_uint2(seed2, coord_hash2));
	                                        Float2 u_brdf2 = make_float2(
	                                            util::uniform_uint_to_float(pcg2.x),
	                                            util::uniform_uint_to_float(pcg2.y));
	                                        // Resolve x2 surface, branching on bounce1.is_procedural()
	                                        // (mesh path was previously flat-field; proc path is new).
	                                        SurfaceData x2_surface_b2;
#if NT_ENABLE_PROCEDURAL
	                                        $if(bounce1->is_procedural()) {
	                                            UInt packed_prim_b2 = (1u << 29u) | bounce1->local_tri();
	                                            x2_surface_b2 = resolve_procedural_surface_textured(
	                                                resolver, proc_bindless, tex_bindless,
	                                                bounce1.hit_prim, packed_prim_b2,
	                                                x2_pos, -wi,
	                                                scene.material_buffer,
	                                                make_float2(bounce1.hit_bary_u, bounce1.hit_bary_v),
	                                                0.0f, make_float2(0.0f), 0u, 0u);
	                                        } $else {
#endif
	                                            UInt4 x2_inst_data_b2 = scene.instance_buffer.read(bounce1.hit_inst);
	                                            x2_surface_b2 = resolve_surface_from_instance(
	                                                resolver, vertex_bindless, tex_bindless,
	                                                x2_inst_data_b2, bounce1.hit_prim,
	                                                make_float2(bounce1.hit_bary_u, bounce1.hit_bary_v),
	                                                scene.material_buffer, -wi,
	                                                scene.instance_transform_buffer.read(bounce1.hit_inst),
	                                                0.0f, make_float2(0.0f), 0u, 0u);
#if NT_ENABLE_PROCEDURAL
	                                        };
#endif
	                                        MaterialBSDF bsdf_x2 = x2_surface_b2.make_bsdf();
	                                        Float brdf_pdf_x2 = def(0.0f);
	                                        Float3 x2_wo = -wi;
	                                        Float3 wi_x2 = bsdf_x2.sample(x2_wo, x2_ns, u_brdf2, brdf_pdf_x2);
	                                        Float cos_wi_x2 = dot(wi_x2, x2_ns);
	                                        $if(brdf_pdf_x2 > 1e-6f & Expr{ cos_wi_x2 } > 1e-4f) {
	                                            Float x2_off = max(0.001f * bounce1.hit_t, 1e-4f);
	                                            auto bounce2 = gi_bounce(x2_pos + x2_ns * x2_off, wi_x2, -wi_x2, bounce1.seed, bounce1.hit_inst);
	                                            $if((bounce2.valid & 1u) != 0u) {
	                                                Float3 x3_radiance = bounce2->rad();
	                                                Float cos_x2 = luisa::compute::max(0.0f, cos_wi_x2);
	                                                Float3 throughput_x2 = bsdf_x2.evaluate(x2_wo, wi_x2, x2_ns) * cos_x2 / brdf_pdf_x2;
	                                                x2_radiance = x2_radiance + throughput_x2 * x3_radiance;
	                                            };
	                                        };
	                                    };
	                                };

	                                // Clamp radiance to prevent firefly contamination
	                                Float x2_rad_lum = luminance(x2_radiance);
	                                x2_radiance = ite(x2_rad_lum > p.giMaxRadiance,
	                                    x2_radiance * (p.giMaxRadiance / x2_rad_lum), x2_radiance);

	                                // Compute p_hat at x1 for this sample
	                                Float p_hat = gi_evaluate_p_hat(
                                    bsdf_mis,
                                    wo, wi, surface.ns,
                                    x2_radiance);

	                                // Apply glass attenuation
	                                p_hat = p_hat * glass_att_gi;

	                                // RIS stream accumulation
	                                Float w = p_hat / brdf_pdf;
	                                ris_w_sum = ris_w_sum + w;
	                                ris_M = ris_M + 1u;
	                                UInt ris_seed = util::xxhash32(make_uint3(pixel_index, base_seed, sample_idx));
	                                Float ris_u = util::uniform_uint_to_float(ris_seed);
	                                $if(ris_u * ris_w_sum < w) {
	                                    sel_pos = x2_pos;
	                                    sel_nrm = x2_ns;
	                                    sel_rad = x2_radiance;
	                                    sel_target_pdf = p_hat;
	                                };
	                            };
	                        };

	                        };

	                    // Create reservoir from RIS-selected candidate.
                    // Trace shadow ray (x1 -> x2) to filter occluded samples
                    // upfront. Without this filter, occluded GI samples
                    // propagate through temporal/spatial reuse and produce
                    // noise patterns before shade rejects them. Visible
                    // samples get visibility=1, vis_age=0; the shade pass
                    // can skip the shadow trace when motion is below
                    // threshold (RTXDI-style reuse).
	                    $if(ris_M > 0u & sel_target_pdf > 1e-8f) {
	                        Float3 gi_to_sample = sel_pos - world_pos;
	                        Float  gi_dist      = luisa::compute::length(gi_to_sample);
	                        Float3 gi_dir       = gi_to_sample * (1.0f / max(gi_dist, 1e-6f));
	                            Float  gi_off = max(0.001f * depth, 1e-4f);
			                        auto   gi_shadow    = make_ray(
                                        world_pos + surface.ns * gi_off + gi_dir * (0.25f * gi_off),
		                                gi_dir, gi_off, gi_dist - gi_off);
	                        Bool gi_vis = def(true);
	                        $if(hasTransparentShadowCasters == 0u) {
	                            // Opaque scene: any-hit occlusion is exact and skips the
	                            // closest-hit traversal + blocker material resolve below.
	                            gi_vis = !render::trace_occluded(accel, gi_shadow
#if NT_ENABLE_PROCEDURAL
	                                , proc_bindless
#endif
	                            );
	                        } $else {
	                            auto   gi_sh_hit    = render::trace_closest(accel, gi_shadow
#if NT_ENABLE_PROCEDURAL
	                                , proc_bindless
#endif
	                            );
	                            $if(!gi_sh_hit->miss()) {
	                                UInt4  gs_inst_data = scene.instance_buffer.read(gi_sh_hit.inst);
	                                Var<MaterialData> gs_mat = scene.material_buffer.read(
	                                    Expr{ gs_inst_data.y & 0xFFu });
	                                Bool gs_tp = (gs_mat.type == 3u | gs_mat.type == 5u | gs_mat.type == 11u);
	                                Bool gs_cutout = is_alpha_cutout(
	                                    gs_mat, vertex_bindless, tex_bindless,
	                                    gs_inst_data, gi_sh_hit.prim, gi_sh_hit.bary);
	                                gi_vis = gs_tp | gs_cutout;
	                            };
	                        };
	
	                        $if(gi_vis) {
	                            gi_r.px = sel_pos.x; gi_r.py = sel_pos.y; gi_r.pz = sel_pos.z;
	                            gi_r->set_nrm(sel_nrm);
	                            gi_r->set_rad(sel_rad);
	                            // Apply RTXDI OFF-mode finalize at write: weight_sum becomes
	                            // post-finalize form. For M=1 collapses to 1/brdf_pdf (matches
	                            // RTXDI MakeGIReservoir). For multi-candidate RIS produces
	                            // standard RIS W = (1/M) * sum(p_hat/pdf) / p_hat_sel.
	                            Float init_denom = cast<Float>(ris_M) * max(sel_target_pdf, 1e-6f);
	                            gi_r.weight_sum = min(ris_w_sum / init_denom, _giWSumCap);
	                            gi_r.target_pdf = sel_target_pdf;
	                            gi_r->set_M(ris_M);
	                            gi_r->set_age(0u);
	                            gi_r->set_visibility(1u);
	                            gi_r->set_vis_age(0u);
	                            gi_r->set_roughness(cast<UInt>(luisa::compute::saturate(surface.roughness) * 63.0f + 0.5f));
	                        };
	                    };

	                    seed_image.write(gbuf_coord, make_uint4(util::xxhash32(make_uint2(pixel_index, base_seed))));
	                };
	            };

	            gi_reservoir_buffer.write(pixel_index, gi_r);
	        });

	        if (!resolverOnly) {
	        //==========================================================================
	        // GI Upsample kernel (bilinear half-res -> full-res)
	        //==========================================================================
	        _giUpsampleShader = device.compile<2>([&](
	            BufferVar<GIReservoir> output_full,
	            BufferVar<GIReservoir> input_half,
	            UInt       half_width,
	            UInt       half_height
	            ) noexcept {
	            set_name("GI_Upsample");
	            set_block_size(16u, 16u, 1u);
	            UInt2 rsv = dispatch_id().xy();
	            UInt2 rsv_res = dispatch_size().xy();
	            UInt2 coord;
	            UInt2 resolution;
	            UInt pixel_index;
	            if (_checkerboard) {
	                // Half-res dispatch: convert to full-res pixel position
	                coord = make_uint2(rsv.x << 1u, rsv.y);
	                resolution = make_uint2(rsv_res.x * 2u, rsv_res.y);
	                pixel_index = rsv.y * rsv_res.x + rsv.x;
	                $if(rsv.x >= rsv_res.x | rsv.y >= rsv_res.y) { $return(); };
	            } else {
	                coord = rsv;
	                resolution = rsv_res;
	                pixel_index = coord.y * resolution.x + coord.x;
	                $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };
	            }

	            // Direct integer-div mapping: each 2x2 block shares one half-res reservoir.
	            // GI initial maps half-res (i,j) -> gbuf_coord (i*2+1, j*2+1),
	            // so inverse is coord/2 (integer division).
	            UInt half_idx = (coord.y / 2u) * half_width + (coord.x / 2u);
	            Var<GIReservoir> r = input_half.read(half_idx);

	            Var<GIReservoir> result;
	            $if(r->is_valid()) {
	                result = r;
	            } $else {
	                result.px = 0.0f; result.py = 0.0f; result.pz = 0.0f;
	                result.packed_normal   = 0u;
					//result.rx = 0.f; result.ry = 0.f; result.rz = 0.f;
	                result.packed_radiance = 0u;
	                result.weight_sum = 0.0f;
	                result.target_pdf = 0.0f;
	                result.packed_meta = 0u;
                };

	            output_full.write(pixel_index, result);
	        });
	        } // end resolverOnly gate (Upsample)

	        //==========================================================================
	        // GI Temporal Reuse kernel (merge current with previous frame, Jacobian)
	        //==========================================================================
	        _giTemporalReuseShader = device.compile<2>([&](
	            BufferVar<GIParams> params,
	            BufferVar<GIReservoir> gi_reservoir_buffer,
	            BufferVar<GIReservoir> gi_reservoir_prev,
	            ImageFloat gbuf_depth,
	            ImageUInt  gbuf_vis,
	            ImageFloat gbuf_bary_motion,
	            UInt       frame_count,
	            Var<util::CameraData> camera,
	            Var<SceneGeometryResources> scene,
	            BindlessVar vertex_bindless,
	            BindlessVar tex_bindless,
	            UInt       cbField,
	            ImageFloat glass_throughput,
	            ImageFloat gbuf_depth_prev,
	            ImageUInt  gbuf_vis_prev,
	            ImageFloat denoise_normal_prev,
	            UInt       tier3BiasCorrectionEnabled,
	            UInt       giScale
#if NT_ENABLE_PROCEDURAL
		        ,
		        BindlessVar proc_bindless
		    #endif
	            ) noexcept {
	            set_name("GI_Temporal");
	            set_block_size(16u, 16u, 1u);
	            auto p = params.read(0u);
	            UInt2 rsv       = dispatch_id().xy();
	            UInt2 rsv_res   = dispatch_size().xy();
	            UInt2 coord;
	            UInt2 resolution;
	            UInt  pixel_index;
	            if (_checkerboard) {
	                // Half-res mode (giScale > 1): dispatch is the native half-res
	                // grid with dense packing (no checkerboard), matching initial.
	                $if(giScale > 1u) {
	                    coord = rsv;
	                    resolution = rsv_res;
	                    pixel_index = rsv.y * rsv_res.x + rsv.x;
	                } $else {
	                    coord = make_uint2(rsv.x << 1u, rsv.y);
	                    coord.x = coord.x + ((coord.y + cbField) & 1u);
	                    resolution = make_uint2(rsv_res.x * 2u, rsv_res.y);
	                    pixel_index = rsv.y * rsv_res.x + rsv.x;
	                };
	                $if(rsv.x >= rsv_res.x | rsv.y >= rsv_res.y) { $return(); };
	            } else {
	                coord = rsv;
	                resolution = rsv_res;
	                pixel_index = rsv.y * rsv_res.x + rsv.x;
	                $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };
	            }

	            // Map dispatch coord to the full-res G-buffer coord (identity when
	            // giScale==1). `resolution` stays in dispatch space for reservoir
	            // indexing / candidate bounds; fullres_* drive all G-buffer taps,
	            // NDCs and screen UVs.
	            UInt  fullres_w = resolution.x * giScale;
	            UInt  fullres_h = resolution.y * giScale;
	            UInt2 gbuf_coord = min(make_uint2(coord.x * giScale + giScale / 2u, coord.y * giScale + giScale / 2u),
	                                   make_uint2(fullres_w - 1u, fullres_h - 1u));

	            UInt4  cur_vis = gbuf_vis.read(gbuf_coord);
	            UInt   cur_inst = cur_vis.x;
	            Bool   cur_is_glass = (cur_vis.y >> 31u) > 0u;
	            Bool   is_point_gi = ((cur_vis.y >> 30u) & 1u) > 0u;
	            Bool   is_procedural_gi_tp = ((cur_vis.y >> 29u) & 1u) > 0u;

	            // Skip temporal reuse for glass pixels -- glass GI accumulates
	            // incorrectly through temporal/spatial reuse (RTXDI-PT uses a dedicated
	            // GlassPass instead). GI initial provides 1 fresh sample/frame for glass.
	            // Skip point/procedural pixels -- no bindless vertex data.
	            $if(cur_inst != ~0u & !cur_is_glass & !is_point_gi) {
	                Var<GIReservoir> current = gi_reservoir_buffer.read(pixel_index);
	                // Initialize streaming accumulator via cross-form identity:
	                // weight_sum_post_finalize * M * target_pdf = raw_input.
	                // Matches RTXDI CombineGIReservoirs on empty accumulator with input
	                // as newReservoir (SpatialResampling.hlsli:46-49). Operand order
	                // (weight_sum, M, target_pdf) preserved for FP-associativity parity.
	                Float w_acc = current.weight_sum * cast<Float>(current->M()) * current.target_pdf;
	                UInt   M_acc = current->M();
	                Float  cur_depth = gbuf_depth.read(gbuf_coord).x;

	                // Reconstruct current pixel world position
	                Float2 ndc = (make_float2(gbuf_coord) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)}) * 2.0f - 1.0f;
	                auto   ray = camera->generate_ray(ndc);
	                Float3 wo = -normalize(ray->direction());
	                UInt   cur_prim_id = cur_vis.y & 0x3FFFFFFFu;
	                Float4 bary_motion_t = gbuf_bary_motion.read(gbuf_coord);
	                Float2 cur_bary = bary_motion_t.xy();
	                UInt4 inst_data = scene.instance_buffer.read(cur_inst);
	                // Gap D: hoist current pixel similarity key once per pixel for
	                // cross-instance material-similarity lookup. Consumed when a
	                // temporal candidate has a different instance (rare: disocclusions).
	                Float3 cur_key_cache = scene.sim_key_buffer.read(Expr{ inst_data.y & 0xFFu });
	                Float3 world_pos;
#if NT_ENABLE_PROCEDURAL
	                $if(is_procedural_gi_tp) {
	                    world_pos = ray->origin() + ray->direction() * cur_depth;
	                } $else {
#endif
	                // cur_is_glass is excluded by the outer temporal gate, so this is
	                // always the non-glass ray-eqn path. world_pos is the camera-ray
	                // hit point in world space — no reconstruct_object_position needed.
	                world_pos = ray->origin() + ray->direction() * cur_depth;
#if NT_ENABLE_PROCEDURAL
	                };
#endif

	                // Object-motion-aware prev-depth reference: reproject the
	                // object-space hit through the previous instance transform so a
	                // surface moving toward/away from the camera still matches its
	                // prev-frame depth (the static-surface assumption in the depth
	                // test below rejected valid history for moving geometry).
	                // Procedural instances have no transform-buffer entry.
	                Float3 depth_ref_pos = world_pos;
#if NT_ENABLE_PROCEDURAL
	                $if(!is_procedural_gi_tp) {
#endif
	                Float3 obj_pos_gt = reconstruct_object_position(
	                    vertex_bindless, inst_data.z, inst_data.w, cur_prim_id, cur_bary);
	                Float4x4 prev_xform_gt = scene.instance_transform_prev_buffer.read(cur_inst);
	                depth_ref_pos = (prev_xform_gt * make_float4(obj_pos_gt, 1.0f)).xyz();
#if NT_ENABLE_PROCEDURAL
	                };
#endif

	                // Reproject to previous frame via motion vector
	                Float2 motion = bary_motion_t.zw();
	                Float2 prev_uv = (ndc + motion + 1.0f) * 0.5f;
	                // Candidate search happens in dispatch space (half-res pixels
	                // when giScale=2); motion vectors live in full-res UV space.
	                Float2 prevFloat = (prev_uv * make_float2(Expr{make_uint2(fullres_w, fullres_h)}) - 0.5f)
	                    / cast<Float>(giScale);
	                // Stochastic rounding: add per-frame random jitter before flooring
	                // to break temporal lock along screen axes at near-zero motion
	                UInt sr_seed = util::xxhash32(make_uint3(pixel_index, frame_count, 17u));
	                Float rx = util::uniform_uint_to_float(sr_seed);
	                Float ry = util::uniform_uint_to_float(util::xxhash32(make_uint2(sr_seed, 1u)));
	                // Per-pixel randomized age limit: 0.5..1.0 * _giMaxAge
	                // Staggers reservoir expiration across frames to avoid coherent ghost patterns
	                UInt age_seed = util::xxhash32(make_uint3(pixel_index, frame_count, 99u));
	                Float maxAgeF = cast<Float>(p.giMaxAge) * (0.5f + 0.5f * util::uniform_uint_to_float(age_seed));
	                UInt maxAge = cast<uint>(maxAgeF);
	                // Primary temporal candidate: stochastic-rounded reprojection (same as original)
	                Int2   prev_coord = make_int2(
	                    cast<int>(luisa::compute::floor(prevFloat.x + rx)),
	                    cast<int>(luisa::compute::floor(prevFloat.y + ry)));

	                //Bool in_bounds = prev_coord.x >= 0 & prev_coord.x < cast<int>(resolution.x)
	                //    & prev_coord.y >= 0 & prev_coord.y < cast<int>(resolution.y);

	                Bool merged = def(false);

	                // Bias-correction tracking (Tier 3 BASIC).
	                UInt    input_M             = current->M();
	                Float   input_target_pdf    = current.target_pdf;
	                Bool    merge_happened      = def(false);
	                Bool    merged_prev_won     = def(false);
	                UInt    merged_prev_M       = def(0u);
	                UInt2   merged_cand_uint    = def(make_uint2(0u));

	                // RTXDI-style 5+1 temporal candidate search (matching reference
	                // TemporalResampling.hlsli: 5 temporal + 1 fallback).
	                // Candidate 0: stochastic-rounded primary at motion vector
	                // Candidates 1-4: jittered cardinal offsets around motion vector
	                // Candidate 5: fallback at current pixel (disocclusion recovery)
	                UInt jit_seed = util::xxhash32(make_uint3(pixel_index, frame_count, 23u));
	                UInt jit_start = jit_seed & 3u;
	                Int jit_r = (_checkerboard ? 2 : 1) * cast<int>(giScale);

	                // Lazy surface resolve: only computed on first valid candidate
	                Bool surface_resolved = def(false);
                        SurfaceData surface;
                        MaterialBSDF bsdf_mis;
                        Float glass_att_t = def(1.0f);

	                $for(ti, 6u) {
	                    $if(!merged) {
	                        Int2 cand_coord;
	                        $if(ti == 0u) {
	                            cand_coord = prev_coord;
	                        } $elif(ti == 5u) {
	                            cand_coord = make_int2(cast<int>(coord.x), cast<int>(coord.y));
	                        } $else {
	                            // 4 cardinal directions with per-pixel random rotation
	                            UInt d = (jit_start + ti - 1u) & 3u;
	                            Int dx = ite(d == 0u, 1, ite(d == 2u, -1, 0));
	                            Int dy = ite(d == 1u, 1, ite(d == 3u, -1, 0));
	                            cand_coord = prev_coord + make_int2(dx * jit_r, dy * jit_r);
	                        };

	                        $if(cand_coord.x >= 0 & cand_coord.x < cast<int>(resolution.x)
	                           & cand_coord.y >= 0 & cand_coord.y < cast<int>(resolution.y)) {
	                                UInt2 cand_uint = make_uint2(cast<uint>(cand_coord.x), cast<uint>(cand_coord.y));
	                                UInt cand_pixel = cast<uint>(cand_coord.y) * resolution.x + cast<uint>(cand_coord.x);
	                                if (_checkerboard) {
	                                    // Half-res candidates use dense packing (no checkerboard);
	                                    // full-res candidates use the checkerboard-compacted index.
	                                    $if(giScale == 1u) {
	                                        cand_pixel = cast<uint>(cand_coord.y) * rsv_res.x + (cast<uint>(cand_coord.x) >> 1u);
	                                    };
	                                }
	                                // Prev-frame G-buffer images are full-res: tap through the
	                                // same center-of-block mapping initial uses (identity at giScale==1).
	                                UInt2 cand_gbuf = min(make_uint2(cand_uint.x * giScale + giScale / 2u, cand_uint.y * giScale + giScale / 2u),
	                                                       make_uint2(fullres_w - 1u, fullres_h - 1u));
	                            // Surface-independent validation runs BEFORE the 32B
	                            // reservoir read: the ~20B prev G-buffer taps gate it, so
	                            // candidates rejected by the depth/instance tests (the
	                            // common case under motion) never touch the reservoir, and
	                            // pixels whose candidates all fail here skip the lazy
	                            // surface resolve below as well.
	                            //
	                            // Surface similarity (RTXDI_IsValidNeighbor parity).
	                            // Without this, a reservoir from a different surface can
	                            // leak through reprojection and stick at a screen pixel
	                            // as the camera moves.
	                            //
	                            // The ti==5 fallback (current-pixel coord) still skips
	                            // the depth/normal tests -- those are unreliable right at
	                            // a disocclusion seam -- but the same-instance gate must
	                            // hold. Without it, when dark geometry moves out of the
	                            // way and reveals bright geometry, the fallback reads prev
	                            // frame's reservoir (which was for the dark surface) and
	                            // merges it into the bright pixel; prev's stale sample
	                            // then dominates for a few frames and produces the
	                            // disocclusion flash.
	                            Float prev_depth_at = gbuf_depth_prev.read(cand_gbuf).x;
	                            UInt4  prev_vis_at  = gbuf_vis_prev.read(cand_gbuf);
	                            // Compensate camera-z motion: prev_depth_at stores t along the
	                            // prev-frame primary ray. depth_ref_pos carries the object's
	                            // prev-transform reprojected position, so this also matches
	                            // for geometry moving under a static camera.
	                            Float3 to_prev_cam        = depth_ref_pos - camera.prev_position;
	                            Float  expected_prev_depth = luisa::compute::length(to_prev_cam);
	                            Float  depth_diff         = luisa::compute::abs(prev_depth_at - expected_prev_depth)
	                                                  / max(max(prev_depth_at, expected_prev_depth), 0.01f);
	                            Bool   same_instance = (prev_vis_at.x == cur_inst);
	                            // Gap D: cross-instance material-similarity gate.
	                            // Mirrors RTXDI RAB_AreMaterialsSimilar (RAB_Material.hlsli:112-128).
	                            // Procedural instances are not in instance_buffer; skip
	                            // material-similarity lookup when current or prev is procedural.
	                            Bool prev_is_proc_t = ((prev_vis_at.y >> 29u) & 1u) > 0u;
	                            $if(!same_instance & !is_procedural_gi_tp & !prev_is_proc_t & prev_vis_at.x != ~0u) {
	                                UInt4 prev_inst_data = scene.instance_buffer.read(prev_vis_at.x);
	                                Float3 prev_key = scene.sim_key_buffer.read(Expr{ prev_inst_data.y & 0xFFu });
	                                same_instance = are_materials_similar_keys(
	                                    prev_key, cur_key_cache,
	                                    p.giMatSimRoughness, p.giMatSimF0, p.giMatSimAlbedo);
	                            };
	                            // neighbor_ok's surface-independent half: false means the
	                            // full gate below would reject this candidate anyway.
	                            Bool pre_ok = same_instance
	                                       & ((ti == 5u)
	                                          | (depth_diff < p.spatialDepthThresh));
	                            $if(pre_ok) {
	                                Var<GIReservoir> prev_r = gi_reservoir_prev.read(cand_pixel);

	                                $if(prev_r->is_valid() & prev_r->age() < maxAge) {
	                                // Lazy surface resolve on first valid candidate
	                                $if(!surface_resolved) {
	                                    Float2 screen_uv_gt = (make_float2(gbuf_coord) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)});
#if NT_ENABLE_PROCEDURAL
	                                    $if(is_procedural_gi_tp) {
	                                        // Recompute bary at unjittered pixel center for stable texture sampling.
	                                        auto gitp_ray_unjit = camera->generate_ray(Expr{ ndc - camera->jitter });
	                                        cur_bary = reconstruct_unjittered_bary_procedural(
	                                            proc_bindless, cur_inst, cur_prim_id, cur_bary,
	                                            gitp_ray_unjit->origin(), gitp_ray_unjit->direction());
	                                        surface = resolve_procedural_surface_textured(
	                                            resolver, proc_bindless, tex_bindless,
	                                            cur_inst, cur_prim_id, world_pos, wo,
	                                            scene.material_buffer,
	                                            cur_bary,
	                                            0.0f, screen_uv_gt, fullres_w, fullres_h);
	                                    } $else {
#endif
	                                    surface = resolve_surface_from_instance(
	                                        resolver, vertex_bindless, tex_bindless,
	                                        inst_data, cur_prim_id, cur_bary,
	                                        scene.material_buffer, wo,
	                                        scene.instance_transform_buffer.read(cur_inst),
	                                        0.0f, screen_uv_gt, fullres_w, fullres_h);
#if NT_ENABLE_PROCEDURAL
	                                    };
#endif
	                                    glass_att_t = ite(cur_is_glass, luminance(glass_throughput.read(gbuf_coord).xyz()), 1.0f);
	                                    bsdf_mis = surface.make_bsdf();
                                    surface_resolved = def(true);
	                                };

	                                // Normal test needs the resolved surface, so it runs
	                                // after the lazy resolve; the depth/instance halves of
	                                // the gate already ran before the reservoir read.
	                                Float3 prev_ns_at   = denoise_normal_prev.read(cand_gbuf).xyz();
	                                Float  n_dot        = luisa::compute::dot(surface.ns, prev_ns_at);
	                                Bool   neighbor_ok  = same_instance
	                                                      & ((ti == 5u)
	                                                         | ((n_dot > p.spatialNormalThresh)
	                                                            & (depth_diff < p.spatialDepthThresh)));

	                                // Radiance freshness: smooth rejection for lighting changes
	                                Float3 prev_rad = prev_r->rad();
	                                Float prev_rad_lum = luminance(prev_rad);
	                                Float cur_rad_lum = luminance(current->rad());
	                                Float freshnessWeight = def(1.0f);
	                                $if(current->M() > 0u & current.target_pdf > 1e-8f & prev_rad_lum > 1e-4f) {
	                                    Float rad_ratio = prev_rad_lum / max(cur_rad_lum, 1e-6f);
	                                    Float logRatio = luisa::compute::abs(luisa::compute::log2(max(rad_ratio, 1e-6f)));
	                                    freshnessWeight = luisa::compute::saturate(1.0f - (logRatio - 1.0f) / 2.0f);
	                                };

	                                Float3 to_sample = prev_r->pos() - world_pos;
	                                Float3 wi = to_sample * rsqrt(Expr{ max(dot(to_sample, to_sample), 1e-6f) });

	                                Float p_hat_new = gi_evaluate_p_hat(
                                    bsdf_mis,
                                    wo, wi, surface.ns, prev_rad);

	                                p_hat_new = p_hat_new * glass_att_t;

	                                $if(p_hat_new > 1e-8f) {
	                                    Float jacobian = p_hat_new / max(prev_r.target_pdf, 1e-6f);
	                                    Float logJac = luisa::compute::abs(luisa::compute::log2(max(jacobian, 1e-6f)));
	                                    Float jacobian_weight = luisa::compute::saturate(1.0f - logJac / 4.0f);

	                                    Float prev_rough_dec = cast<Float>(prev_r->roughness()) / 63.0f;
	                                    Bool rough_ok_t = luisa::compute::abs(prev_rough_dec - surface.roughness) < p.giRoughnessInvalidationThresh;
	                                    $if(jacobian_weight > 0.01f & freshnessWeight > 0.01f & neighbor_ok & rough_ok_t) {
	                                        jacobian = clamp(jacobian, 1.0f / 3.0f, 3.0f);
	                                        // RTXDI combine: risWeight = targetPdf * weightSum * M.
	                                        // prev.weight_sum is post-finalize so the formula folds in domain
	                                        // conversion. Algebraically equal to old formula under new rep.
	                                        Float w_prev = p_hat_new * prev_r.weight_sum * cast<Float>(prev_r->M())
	                                                     * jacobian_weight * freshnessWeight;
	                                        Float w_sum_merged = w_acc + w_prev;
	                                        UInt  M_merged = current->M() + prev_r->M();

	                                        // MIS bias correction
	                                        Float w_c = current.target_pdf * cast<Float>(current->M());
	                                        Float w_p = p_hat_new * cast<Float>(prev_r->M());
	                                        Float mis_total = w_c + w_p;
	                                        Float u = util::uniform_uint_to_float(Expr{ util::xxhash32(make_uint2(pixel_index, frame_count)) });
	                                        Float threshold = ite(mis_total > 0.0f, w_p / mis_total, 0.0f);

	                                        Bool prev_won = u < threshold;
	                                        $if(prev_won) {
	                                            current.px = prev_r.px; current.py = prev_r.py; current.pz = prev_r.pz;
	                                            // Copy packed x2 fields verbatim: a decode→re-encode
	                                            // roundtrip here would re-quantize radiance/normal every
	                                            // frame the temporal winner persists.
	                                            current.packed_normal = prev_r.packed_normal;
	                                            current.packed_radiance = prev_r.packed_radiance;
	                                            current.target_pdf = p_hat_new;
	                                            // prev_r wins: inherit prev visibility (its shadow was
	                                            // traced for prev receiver position; vis_age advances).
	                                            // When current wins, current.sample stays but vis_age
	                                            // still advances one frame.
	                                            current->set_visibility(prev_r->visibility());
	                                            current->set_vis_age(prev_r->vis_age() + 1u);
	                                        } $else {
	                                            current->set_vis_age(current->vis_age() + 1u);
	                                        };

	                                        UInt M_capped = min(M_merged, p.giTemporalMaxM);
	                                        Float w_scale = cast<float>(M_capped) / cast<float>(max(M_merged, 1u));
	                                        w_acc = min(w_sum_merged * w_scale, p.giWSumCap);
	                                        M_acc = M_capped;
	                                        current->set_age(prev_r->age() + 1u);
	                                        merged = def(true);

	                                        // Record for second-pass bias correction
	                                        merge_happened   = def(true);
	                                        merged_prev_won  = prev_won;
	                                        merged_prev_M    = prev_r->M();
	                                        merged_cand_uint = cand_uint;
	                                    };
	                                };
	                            };
	                            };
	                        };
	                    };
	                };

	                // Commit accumulators to current before finalize.
	                current.weight_sum = w_acc;
	                current->set_M(M_acc);
	                $if(surface_resolved) {
	                	current->set_roughness(cast<UInt>(luisa::compute::saturate(surface.roughness) * 63.0f + 0.5f));
	                };

	                // BASIC bias correction (RTXDI-BASIC, TemporalResampling.hlsli:185-217).
	                // Single-neighbor piSum MIS. Re-evaluates the winner's pdf at the merged
	                // neighbor's surface (inverse query), then rescales weight_sum via
	                // finalize_gi so weight() yields the bias-corrected multiplier at shade.
	                Bool bias_corrected_t = def(false);
	                        $if(tier3BiasCorrectionEnabled != 0u & merge_happened) {
	                            UInt2 merged_gbuf = min(make_uint2(merged_cand_uint.x * giScale + giScale / 2u, merged_cand_uint.y * giScale + giScale / 2u),
	                                                    make_uint2(fullres_w - 1u, fullres_h - 1u));
	                            UInt4 merged_vis_at   = gbuf_vis.read(merged_gbuf);
	                            Float merged_depth_at = gbuf_depth.read(merged_gbuf).x;
	                            Float2 merged_bary_at = gbuf_bary_motion.read(merged_gbuf).xy();
	                            UInt  merged_inst_at  = merged_vis_at.x;
	                            UInt  merged_prim_at  = merged_vis_at.y & 0x3FFFFFFFu;
	                            Bool  merged_is_proc  = ((merged_vis_at.y >> 29u) & 1u) > 0u;

	                            Float2 merged_ndc = (make_float2(merged_gbuf) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)}) * 2.0f - 1.0f;
	                    auto   merged_ray = camera->generate_ray(merged_ndc);
	                    Float3 merged_world_pos = merged_ray->origin() + merged_ray->direction() * merged_depth_at;
	                    Float3 merged_wo = -normalize(merged_ray->direction());

	                    // BASIC inverse query: resolve merged candidate surface regardless
	                    // of geometry type (mesh or procedural).
	                    $if(merged_inst_at != ~0u) {
	                        UInt4   merged_inst_data = scene.instance_buffer.read(merged_inst_at);
	                            Float2  merged_screen_uv = (make_float2(merged_gbuf) + 0.5f) / make_float2(Expr{make_uint2(fullres_w, fullres_h)});
	                        SurfaceData merged_surface;
#if NT_ENABLE_PROCEDURAL
	                        $if(merged_is_proc) {
	                            merged_surface = resolve_procedural_surface_textured(
	                                resolver, proc_bindless, tex_bindless,
	                                merged_inst_at, merged_prim_at, merged_world_pos, merged_wo,
	                                scene.material_buffer, merged_bary_at,
	                                0.0f, merged_screen_uv, fullres_w, fullres_h);
	                        }
	                        $else {
#endif
	                            merged_surface = resolve_surface_from_instance(
	                                resolver, vertex_bindless, tex_bindless,
	                                merged_inst_data, merged_prim_at, merged_bary_at,
	                                scene.material_buffer, merged_wo,
	                                scene.instance_transform_buffer.read(merged_inst_at),
	                                0.0f, merged_screen_uv, fullres_w, fullres_h);
#if NT_ENABLE_PROCEDURAL
	                        };
#endif

	                        // Inverse query: winner's pdf at the merged neighbor's surface.
                        MaterialBSDF merged_bsdf = merged_surface.make_bsdf();
	                        Float3 to_sample_m = current->pos() - merged_world_pos;
	                        Float3 wi_m = to_sample_m * rsqrt(Expr{ max(dot(to_sample_m, to_sample_m), 1e-6f) });
	                        Float ps_merged = gi_evaluate_p_hat(
                                merged_bsdf,
                                merged_wo, wi_m, merged_surface.ns, current->rad());

	                        // piSum = input.M * selectedTargetPdf + merged.M * ps_merged
	                        // pi    = winner's pdf at the surface that originated it.
	                        Float piSum = cast<Float>(input_M) * current.target_pdf
	                            + cast<Float>(merged_prev_M) * ps_merged;
	                        Float pi    = ite(merged_prev_won, ps_merged, current.target_pdf);

	                        // Divergence gate: skip bias correction when pdfs too similar.
	                        // BASIC inherently over-corrects similar distributions (wall-to-wall
	                        // diffuse) producing ~1/(M*p) amplification. Threshold 0 = always apply.
	                        Float pdf_max_t = max(current.target_pdf, ps_merged);
	                        Float pdf_min_t = max(min(current.target_pdf, ps_merged), 1e-6f);
	                        Float pdf_ratio_t = pdf_max_t / pdf_min_t;
	                        Bool  diverged_t = pdf_ratio_t >= p.giBiasCorrectionDivergenceThresh;

	                        $if(piSum > 1e-10f & diverged_t) {
	                            // RTXDI-faithful BASIC finalize: pi / (selectedTargetPdf * piSum).
	                            // finalize_gi handles zero-denominator via ite guard matching RTXDI.
	                            finalize_gi(current, pi, current.target_pdf * piSum);
	                            current.weight_sum = min(current.weight_sum, p.giWSumCap);  // app-specific re-cap
	                            bias_corrected_t = def(true);
	                        };
	                    };
	                };

	                // OFF-mode finalize (also fallback if BASIC skipped above).
	                $if(!bias_corrected_t) {
	                    finalize_gi(current, 1.0f,
	                                cast<Float>(current->M()) * max(current.target_pdf, 1e-6f));
	                };

	                gi_reservoir_buffer.write(pixel_index, current);
	            };
	        });
			//==========================================================================
	        // GI Spatial Reuse kernel (share GI reservoirs with neighbors)
	        //==========================================================================
	        _giSpatialReuseShader = device.compile<2>([&](
	            BufferVar<GIParams> params,
	            BufferVar<GIReservoir> gi_reservoir_output,
	            BufferVar<GIReservoir> gi_reservoir_input,
	            ImageFloat gbuf_depth,
	            ImageUInt  gbuf_vis,
	            ImageFloat gbuf_bary_motion,
	            AccelVar   accel,
	            UInt       frame_count,
	            Var<SceneGeometryResources> scene,
	            BindlessVar vertex_bindless,
	            BindlessVar tex_bindless,
	            Var<util::CameraData> camera,
	            UInt       cbField,
	            ImageFloat glass_throughput,
	            ImageFloat denoise_normal,
	            UInt       tier3BiasCorrectionEnabled
#if NT_ENABLE_PROCEDURAL
		        ,
		        BindlessVar proc_bindless
		    #endif
	            ) noexcept {
	            set_name("GI_Spatial");
	            set_block_size(16u, 16u, 1u);
	            auto p = params.read(0u);
	            UInt2 rsv       = dispatch_id().xy();
	            UInt2 rsv_res   = dispatch_size().xy();
	            UInt2 coord;
	            UInt2 resolution;
	            UInt  pixel_index;
	            if (_checkerboard) {
	                coord = make_uint2(rsv.x << 1u, rsv.y);
	                coord.x = coord.x + ((coord.y + cbField) & 1u);
	                resolution = make_uint2(rsv_res.x * 2u, rsv_res.y);
	                pixel_index = rsv.y * rsv_res.x + rsv.x;
	                $if(rsv.x >= rsv_res.x | rsv.y >= rsv_res.y) { $return(); };
	            } else {
	                coord = rsv;
	                resolution = rsv_res;
	                pixel_index = rsv.y * rsv_res.x + rsv.x;
	                $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };
	            }

	            Var<GIReservoir> r = gi_reservoir_input.read(pixel_index);

	            UInt4 vis = gbuf_vis.read(coord);
	            UInt  inst_id = vis.x;
	            Bool is_glass_sp = (vis.y >> 31u) > 0u;
	            Bool is_point_gi = ((vis.y >> 30u) & 1u) > 0u;
	            Bool is_procedural_gi_sp = ((vis.y >> 29u) & 1u) > 0u;

	            // Skip spatial reuse for glass pixels (same reason as temporal).
	            $if(inst_id != ~0u & !is_glass_sp & !is_point_gi) {
	                // Save pre-merge state for post-merge visibility revert.
	                // If spatial merge replaces r's x2 with a neighbor's x2
	                // that's occluded from the current receiver, we restore
	                // r_pre to keep a usable (initial-verified) x2.
	                Var<GIReservoir> r_pre = r;
	                Float depth = gbuf_depth.read(coord).x;

	                // Reconstruct current pixel world pos + normal
	                auto   ray = camera->generate_ray(Expr{
	                    (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f
	                    });
	                Float3 wo = -normalize(ray->direction());
	                UInt   prim_id_sp = vis.y & 0x3FFFFFFFu;
	                Float2 bary_sp = gbuf_bary_motion.read(coord).xy();
	                UInt4 inst_data = scene.instance_buffer.read(inst_id);
	                // Gap D: hoist current pixel similarity key for cross-instance
	                // material-similarity lookup in spatial reuse (same pattern as temporal).
	                Float3 cur_key_cache_sp = scene.sim_key_buffer.read(Expr{ inst_data.y & 0xFFu });
	                Float3 world_pos;
	                SurfaceData surface;
#if NT_ENABLE_PROCEDURAL
	                $if(is_procedural_gi_sp) {
	                    world_pos = ray->origin() + ray->direction() * depth;
	                    // Recompute bary at unjittered pixel center for stable texture sampling.
	                    auto gisp_ray_unjit = camera->generate_ray(Expr{
	                        (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter
	                    });
	                    bary_sp = reconstruct_unjittered_bary_procedural(
	                        proc_bindless, inst_id, prim_id_sp, bary_sp,
	                        gisp_ray_unjit->origin(), gisp_ray_unjit->direction());
	                    surface = resolve_procedural_surface_textured(
	                        resolver, proc_bindless, tex_bindless,
	                        inst_id, prim_id_sp, world_pos, wo,
	                        scene.material_buffer,
	                        bary_sp,
	                        0.0f,
	                        Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
	                        resolution.x, resolution.y);
	                } $else {
#endif
	                // is_glass_sp is excluded by the outer spatial gate, so this is
	                // always the non-glass ray-eqn path. world_pos is the camera-ray
	                // hit point in world space — no reconstruct_object_position needed.
	                world_pos = ray->origin() + ray->direction() * depth;

	                Float2 screen_uv_gs = (make_float2(coord) + 0.5f) / make_float2(resolution);
	                surface = resolve_surface_from_instance(
	                    resolver, vertex_bindless, tex_bindless,
	                    inst_data, prim_id_sp, bary_sp,
	                    scene.material_buffer, wo,
	                    scene.instance_transform_buffer.read(inst_id),
	                    0.0f, screen_uv_gs, resolution.x, resolution.y);
#if NT_ENABLE_PROCEDURAL
	                };
#endif


	                // Hoist bsdf_mis for spatial reuse p_hat (single construction per pixel)
                        MaterialBSDF bsdf_mis = surface.make_bsdf();
                        // Glass attenuation (hoisted before neighbor loop, gated on is_glass)
	                Float glass_att_s = ite(is_glass_sp, luminance(glass_throughput.read(coord).xyz()), 1.0f);

	                // Spatial reuse: try merge with K random disk neighbors
	                // Golden-angle spiral + randomized start for well-distributed sampling
	                // gi_spatial_count is a runtime UInt so DXC can't unroll the
	                // loop below (same fix as DI commit 29b6cda). kGiSpatialNeighborCount
	                // remains the compile-time MAX the runtime value must not exceed.
	                const UInt gi_spatial_count = p.giSpatialNeighborCount;
	                UInt gi_spatial_radius = p.giSpatialRadius;
	                constexpr float golden_angle = 2.39996322973f; // pi * (3 - sqrt(5))

	                // Per-pixel per-frame random offset to break correlation
	                UInt base_seed = util::xxhash32(make_uint2(pixel_index, frame_count));
	                Float angle_offset = util::uniform_uint_to_float(base_seed) * 6.28318530718f;

	                // Bias-correction tracking (Tier 3 BASIC).
	                UInt    input_M_sp         = r->M();
	                Int     selected_neighbor  = def(-1);  // -1 = input won, 0..3 = neighbor n won
	                UInt    cached_mask        = def(0u);  // Option B: bit n set = neighbor n passed all filters + merged

	                // Initialize streaming accumulator via cross-form identity:
	                // weight_sum_post_finalize * M * target_pdf = raw_input.
	                // Matches RTXDI CombineGIReservoirs on empty accumulator with input
	                // as newReservoir. Operand order preserved for FP-associativity.
	                Float w_acc = r.weight_sum * cast<Float>(r->M()) * r.target_pdf;
	                UInt   M_acc = r->M();

	                $for(n, 0u, gi_spatial_count) {
	                    // Golden-angle rotation gives well-separated directions
	                    Float angle = angle_offset + cast<float>(n) * golden_angle;
	                    // Stratified radius: each neighbor covers its own ring
	                    UInt r_seed = util::xxhash32(make_uint2(base_seed, n + 200u));
	                    Float r_disk = sqrt(Expr{ (cast<float>(n) + util::uniform_uint_to_float(r_seed)) / cast<float>(gi_spatial_count) }) * cast<float>(gi_spatial_radius);

	                    Int nc_x = cast<Int>(coord.x) + cast<Int>(r_disk * cos(angle));
	                    Int nc_y = cast<Int>(coord.y) + cast<Int>(r_disk * sin(angle));

	                    //Bool valid = nc_x >= 0 & nc_x < cast<Int>(resolution.x)
	                    //           & nc_y >= 0 & nc_y < cast<Int>(resolution.y);

	                    $if(Expr{ nc_x >= 0 & nc_x < cast<Int>(resolution.x)
	                            & nc_y >= 0 & nc_y < cast<Int>(resolution.y) }) {
	                        UInt2 nc_uint = make_uint2(cast<uint>(nc_x), cast<uint>(nc_y));
	                        UInt  n_pixel = cast<uint>(nc_y) * resolution.x + cast<uint>(nc_x);
	                        if (_checkerboard)
	                            n_pixel = cast<uint>(nc_y) * rsv_res.x + (cast<uint>(nc_x) >> 1u);

	                        Float n_depth = gbuf_depth.read(nc_uint).x;
	                        UInt4  n_vis = gbuf_vis.read(nc_uint);

	                        // Edge rejection. Procedural neighbors ARE allowed into the
	                        // first-pass merge — it doesn't resolve their surface (reads
	                        // denoise_normal + reservoir packed fields). They are excluded
	                        // from cached_mask below so the second-pass bias correction
	                        // (which calls resolve_surface_from_instance) never sees them.
	                        Float depth_diff = abs(n_depth - depth) / max(depth, 0.01f);

	                        Bool n_is_glass = (n_vis.y >> 31u) != 0u;
	                        Bool n_is_proc  = ((n_vis.y >> 29u) & 1u) > 0u;
	                        // Gap D: spatial cross-instance material-similarity gate.
	                        // Mirrors RTXDI RAB_AreMaterialsSimilar. Pools variance from
	                        // neighbor surfaces under high-contrast lighting (carpet +
	                        // strong local light).
	                        Bool same_material = (n_vis.x == inst_id);
	                        // Procedural current or neighbor: skip lookup (not in instance_buffer).
	                        $if(!same_material & !is_procedural_gi_sp & !n_is_proc & n_vis.x != ~0u) {
	                            UInt4 n_inst_data = scene.instance_buffer.read(n_vis.x);
	                            Float3 n_key = scene.sim_key_buffer.read(Expr{ n_inst_data.y & 0xFFu });
	                            same_material = are_materials_similar_keys(
	                                n_key, cur_key_cache_sp,
	                                p.giMatSimRoughness, p.giMatSimF0, p.giMatSimAlbedo);
	                        };
	                        Bool similar = same_material & (depth_diff < p.spatialDepthThresh) & (!n_is_glass);

	                        $if(similar) {
	                            Float3 n_ns = denoise_normal.read(nc_uint).xyz();
	                            Float normal_dot = luisa::compute::dot(surface.ns, n_ns);

	                            $if(normal_dot > p.spatialNormalThresh) {
	                                Var<GIReservoir> nr = gi_reservoir_input.read(n_pixel);

	                                $if(nr->is_valid() & nr.target_pdf > 1e-6f) {
	                                    // Radiance consistency: reject neighbors whose stored radiance
	                                    // diverges from the current pixel's radiance (e.g. stale bright
	                                    // data from a light that moved away). Same smooth curve as temporal.
	                                    Float3 nr_rad = nr->rad();
	                                    Float nr_rad_lum = luminance(nr_rad);
	                                    Float cur_rad_lum_sp = luminance(r->rad());
	                                    Float sp_freshness = def(1.0f);
	                                    $if(r->M() > 0u & r.target_pdf > 1e-8f & nr_rad_lum > 1e-4f) {
	                                        Float sp_ratio = nr_rad_lum / max(cur_rad_lum_sp, 1e-6f);
	                                        Float sp_logRatio = luisa::compute::abs(luisa::compute::log2(max(sp_ratio, 1e-6f)));
	                                        sp_freshness = luisa::compute::saturate(1.0f - (sp_logRatio - 1.0f) / 2.0f);
	                                    };

	                                    // Evaluate p_hat for neighbor's sample at current position
	                                    Float3 sample_pos = nr->pos();
	                                    Float3 sample_nrm = nr->nrm();
	                                    Float3 sample_rad = nr_rad;

	                                    Float3 to_sample = sample_pos - world_pos;
	                                    Float3 wi = to_sample * rsqrt(max(dot(to_sample, to_sample), 1e-6f));


	                                    Float p_hat_new = gi_evaluate_p_hat(
                                        bsdf_mis,
                                        wo, wi, surface.ns, sample_rad);

	                                    // Apply glass attenuation (hoisted before loop)
	                                    p_hat_new = p_hat_new * glass_att_s;

	                                    $if(p_hat_new > 1e-8f & sp_freshness > 0.01f) {
	                                        // Geometric Jacobian (RTXDI-style, ReSTIR GI paper Eq.11)
	                                        // Reconstruct neighbor world position from same-frame camera
	                                        Float2 n_ndc = (make_float2(nc_uint) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
	                                        auto   n_ray = camera->generate_ray(n_ndc);
	                                        Float3 n_world_pos = n_ray->origin() + n_ray->direction() * n_depth;

	                                        Float jacobian = gi_geometric_jacobian(
	                                            world_pos, n_world_pos, sample_pos, sample_nrm);

	                                        // Pdf-ratio gate (matches temporal's jacobian_weight at line 920-922).
	                                        // Without this, a neighbor whose selected sample has very low
	                                        // target_pdf (high weight_sum post-finalize) dominates the merge
	                                        // when current's surface sees the same sample at much higher pdf.
	                                        // Result: high weight_sum propagates along seams (the "GI flushes
	                                        // along edge seam" artifact). Temporal already has this gate;
	                                        // spatial was missing it — the asymmetry IS the bug.
	                                        Float pdf_ratio_s = p_hat_new / max(nr.target_pdf, 1e-6f);
	                                        Float log_pdf_ratio_s = luisa::compute::abs(luisa::compute::log2(max(pdf_ratio_s, 1e-6f)));
	                                        Float pdf_ratio_weight_s = luisa::compute::saturate(1.0f - log_pdf_ratio_s / 4.0f);

	                                        Float n_rough_dec = cast<Float>(nr->roughness()) / 63.0f;
	                                        Bool n_rough_ok = luisa::compute::abs(n_rough_dec - surface.roughness) < p.giRoughnessInvalidationThresh;
	                                        $if(jacobian > 0.0f & n_rough_ok & pdf_ratio_weight_s > 0.01f) {
	                                            // RTXDI combine: targetPdf (includes geometric jacobian)
	                                            // * weightSum * M. nr.weight_sum is post-finalize.
	                                            // Multiplied by pdf_ratio_weight_s to smoothly reject
	                                            // BRDF-shifted merges (pdf ratio > 16 or < 1/16 → weight = 0).
	                                            Float w_neighbor = p_hat_new * jacobian * nr.weight_sum
	                                                         * cast<Float>(nr->M()) * sp_freshness
	                                                         * pdf_ratio_weight_s;
	                                            Float w_sum_merged = w_acc + w_neighbor;

	                                            Float threshold = w_neighbor / max(w_sum_merged, 1e-10f);
	                                            Float u = util::uniform_uint_to_float(
	                                                util::xxhash32(make_uint3(pixel_index, frame_count, cast<UInt>(n) + 200u)));

	                                            $if(u < threshold) {
	                                                r.px = nr.px; r.py = nr.py; r.pz = nr.pz;
	                                                // Copy packed x2 fields verbatim: a decode→re-encode
	                                                // roundtrip here would re-quantize radiance/normal on
	                                                // every spatial merge that the neighbor wins.
	                                                r.packed_normal = nr.packed_normal;
	                                                r.packed_radiance = nr.packed_radiance;
	                                                r.target_pdf = p_hat_new;
	                                                // Neighbor wins: its shadow trace was for a
	                                                // different receiver position, so we cannot
	                                                // reuse it. Force a fresh trace in shade.
	                                                r->set_visibility(0u);
	                                                r->set_vis_age(0u);
	                                                // Record winning neighbor index for bias correction (single
	                                                // DSL assignment — host-side if/else chain only worked
	                                                // when n was a compile-time constant during unrolling).
	                                                selected_neighbor = cast<Int>(n);
	                                            };
	                                            UInt M_spatial_merged = M_acc + nr->M();
	                                            UInt M_spatial_capped = min(M_spatial_merged, p.giSpatialMaxM);
	                                            Float w_scale = cast<float>(M_spatial_capped) / cast<float>(max(M_spatial_merged, 1u));
	                                            w_acc = min(w_sum_merged * w_scale, p.giWSumCap);
	                                            M_acc = M_spatial_capped;

	                                                // Option B: mark this neighbor for second pass.
	                                                cached_mask = cached_mask | (1u << cast<UInt>(n));
	                                        };
	                                    };
	                                };
	                            };
	                        };
	                    };
	                };

	                // Commit accumulators to r before finalize.
	                r.weight_sum = w_acc;
	                r->set_M(M_acc);
	                r->set_roughness(cast<UInt>(luisa::compute::saturate(surface.roughness) * 63.0f + 0.5f));

	                // BASIC bias correction (RTXDI-BASIC, SpatialResampling.hlsli:122-180).
	                // 2nd-loop piSum MIS: re-resolve each merged neighbor's surface,
	                // evaluate winner's pdf at each, sum piSum. Procedural neighbors
	                // excluded via cached_mask (resolve_surface_from_instance is mesh-only).
	                Bool bias_corrected_s = def(false);
	                // Skip when any procedural neighbor merged -- can't resolve proc surface,
	                // so piSum would miss proc's pdf and multiplier would blow up W.
	                $if(tier3BiasCorrectionEnabled != 0u) {
	                    Float piSum = cast<Float>(input_M_sp) * r.target_pdf;
	                    Float pi    = r.target_pdf;  // default: input won
	                    // Divergence tracking across input pdf and all merged neighbor ps_n.
	                    Float max_ps_s = r.target_pdf;
	                    Float min_ps_s = r.target_pdf;
	                    // Winner is read-only in this pass -- cache decoded pos/rad once.
	                    Float3 r_pos_cached = r->pos();
	                    Float3 r_rad_cached = r->rad();

	                    $for(n, 0u, gi_spatial_count) {
	                        // Option B: skip neighbors that did not merge in first pass.
	                        // cached_mask bit n set implies in-bounds + similar + normal + valid + non-proc.
	                        UInt bit_n = 1u << cast<UInt>(n);
	                        $if((cached_mask & bit_n) != 0u) {
	                            Float angle_n = angle_offset + cast<float>(n) * golden_angle;
	                            UInt  r_seed_n = util::xxhash32(make_uint2(base_seed, n + 200u));
	                            Float r_disk_n = sqrt(Expr{ (cast<float>(n) + util::uniform_uint_to_float(r_seed_n)) / cast<float>(gi_spatial_count) }) * cast<float>(gi_spatial_radius);
	                            Int   nc_x_n = cast<Int>(coord.x) + cast<Int>(r_disk_n * cos(angle_n));
	                            Int   nc_y_n = cast<Int>(coord.y) + cast<Int>(r_disk_n * sin(angle_n));
	                            UInt2  nc_uint_n = make_uint2(cast<uint>(nc_x_n), cast<uint>(nc_y_n));
	                            UInt   n_pixel_n = cast<uint>(nc_y_n) * resolution.x + cast<uint>(nc_x_n);
	                            if (_checkerboard)
	                            n_pixel_n = cast<uint>(nc_y_n) * rsv_res.x + (cast<uint>(nc_x_n) >> 1u);

	                            Float  n_depth_n = gbuf_depth.read(nc_uint_n).x;
	                            UInt4  n_vis_n   = gbuf_vis.read(nc_uint_n);
	                            Float2 n_bary_n  = gbuf_bary_motion.read(nc_uint_n).xy();

	                            Var<GIReservoir> nr_n = gi_reservoir_input.read(n_pixel_n);

	                            // Resolve neighbor surface in current frame
	                            Float2 n_ndc_n = (make_float2(nc_uint_n) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
	                            auto   n_ray_n = camera->generate_ray(n_ndc_n);
	                            Float3 n_world_pos_n = n_ray_n->origin() + n_ray_n->direction() * n_depth_n;
	                            Float3 n_wo_n = -normalize(n_ray_n->direction());
	                            UInt4   n_inst_data_n = scene.instance_buffer.read(n_vis_n.x);
	                            Float2  n_screen_uv_n = (make_float2(nc_uint_n) + 0.5f) / make_float2(resolution);
	                            UInt    n_prim_n = n_vis_n.y & 0x3FFFFFFFu;
#if NT_ENABLE_PROCEDURAL
	                            Bool    n_is_proc_n = ((n_vis_n.y >> 29u) & 1u) > 0u;
#endif
	                            SurfaceData n_surface_n;
#if NT_ENABLE_PROCEDURAL
	                            $if(n_is_proc_n) {
	                                n_surface_n = resolve_procedural_surface_textured(
	                                    resolver, proc_bindless, tex_bindless,
	                                    n_vis_n.x, n_prim_n, n_world_pos_n, n_wo_n,
	                                    scene.material_buffer, n_bary_n,
	                                    0.0f, n_screen_uv_n, resolution.x, resolution.y);
	                            }
	                            $else {
#endif
	                                n_surface_n = resolve_surface_from_instance(
	                                    resolver, vertex_bindless, tex_bindless,
	                                    n_inst_data_n, n_prim_n, n_bary_n,
	                                    scene.material_buffer, n_wo_n,
	                                    scene.instance_transform_buffer.read(n_vis_n.x),
	                                    0.0f, n_screen_uv_n, resolution.x, resolution.y);
#if NT_ENABLE_PROCEDURAL
	                            };
#endif

	                            // Winner's pdf at this neighbor's surface
                            MaterialBSDF n_bsdf_n = n_surface_n.make_bsdf();
	                            Float3 to_sample_n = r_pos_cached - n_world_pos_n;
	                            Float3 wi_n = to_sample_n * rsqrt(Expr{ max(dot(to_sample_n, to_sample_n), 1e-6f) });
	                            Float ps_n = gi_evaluate_p_hat(
                                n_bsdf_n,
                                n_wo_n, wi_n, n_surface_n.ns, r_rad_cached);

	                            piSum = piSum + ps_n * cast<Float>(nr_n->M());
	                            max_ps_s = max(max_ps_s, ps_n);
	                            min_ps_s = min(min_ps_s, ps_n);
	                            $if(selected_neighbor == cast<Int>(n)) {
	                                pi = ps_n;
	                            };
	                        };
	                    };

	                    // Divergence gate: skip when all pdfs (input + neighbors) too similar.
	                    Float pdf_ratio_s = max_ps_s / max(min_ps_s, 1e-6f);
	                    Bool  diverged_s = pdf_ratio_s >= p.giBiasCorrectionDivergenceThresh;

	                    $if(piSum > 1e-10f & diverged_s) {
	                        // RTXDI-faithful BASIC finalize: pi / (selectedTargetPdf * piSum).
	                        finalize_gi(r, pi, r.target_pdf * piSum);
	                        r.weight_sum = min(r.weight_sum, p.giWSumCap);  // app-specific re-cap
	                        bias_corrected_s = def(true);
	                    };
	                };

                // OFF-mode finalize (also fallback if BASIC skipped above).
                $if(!bias_corrected_s) {
                    finalize_gi(r, 1.0f,
                               cast<Float>(r->M()) * max(r.target_pdf, 1e-6f));
                };

                    // === Option 2: post-merge current→x2 visibility check ===
                    // If spatial merge replaced r's x2 with a neighbor's x2
                    // (selected_neighbor >= 0), verify the inherited x2 is
                    // visible from the current receiver. If occluded — e.g.,
                    // a wall pixel inherited an x2 whose line-of-sight from
                    // this pixel passes through the cornell-box sphere/cube —
                    // revert r to its pre-merge state. Pre-merge x2 was
                    // already verified visible by initial's shadow ray, so
                    // shade gets a usable sample instead of a persistent
                    // dark patch. Without this revert, occluded x2 persists
                    // across frames via temporal merge (giTemporalMaxM up
                    // to 64), producing the disocclusion shadow patterns.
                    $if(selected_neighbor >= 0) {
                        Float3 to_x2 = r->pos() - world_pos;
                        Float  x2_dist = luisa::compute::length(to_x2);
                        Float3 x2_dir = to_x2 * (1.0f / max(x2_dist, 1e-6f));
                        Float  x2_off = max(0.001f * depth, 1e-4f);
                        auto   x2_ray = make_ray(world_pos + surface.ns * x2_off,
                                                  x2_dir, 0.0f, x2_dist - x2_off);
                        Bool   x2_occluded = render::trace_occluded(accel, x2_ray
#if NT_ENABLE_PROCEDURAL
                            , proc_bindless
#endif
                        );
                        $if(x2_occluded) {
                            r = r_pre;
                        };
                    };
	            };

	            // Always write — glass/point/sky/invalid pixels skip spatial merge above
	            // but still need their input reservoir forwarded to the output slot. The
	            // GI reservoir buffer is double-buffered with a per-frame flip; if the
	            // skip path leaves the output slot unwritten, shade (which reads the
	            // post-flip slot) sees stale data from before. After a GI off->on
	            // toggle with camera motion, that manifests as stale GI inside glass
	            // objects while opaque looks fresh. Writing unchanged `r` for skipped
	            // pixels keeps the output slot in sync.
	            gi_reservoir_output.write(pixel_index, r);
	        });

	        if (!resolverOnly) {
	        //==========================================================================
	        // Boiling Filter (GI): discard outlier reservoirs within 16x16 blocks
	        //==========================================================================
	        _boilingFilterGI = device.compile<2>([&](
	            BufferVar<GIReservoir> gi_reservoir_buffer,
	            Float strength,
	            UInt cbField
	            ) noexcept {
	            set_name("GI_BoilingFilter");
	            set_block_size(16u, 16u, 1u);
	            UInt2 rsv       = dispatch_id().xy();
	            UInt2 rsv_res   = dispatch_size().xy();
	            // OOB threads stay alive and contribute zeros so the wave/block
	            // reductions remain well-defined at screen edges. Early-return
	            // here would diverge the warp and break the block average.
	            Bool  in_bounds  = !(any(rsv >= rsv_res));
	            UInt  pixel_index = rsv.y * rsv_res.x + rsv.x;
	            UInt  local_id   = (dispatch_id().y % 16u) * 16u + (dispatch_id().x % 16u);

	            Shared<float> shared_warp_w(8u);
	            Shared<uint>  shared_warp_v(8u);

	            // All threads in half-res dispatch are active checkerboard pixels
	            //Bool is_active = inBounds;

	            Var<GIReservoir> r;
	            $if(in_bounds) {
	                r = gi_reservoir_buffer.read(pixel_index);
	            } $else {
	                r.px = 0.0f; r.py = 0.0f; r.pz = 0.0f;
	                r.packed_normal   = 0u;
	                r.packed_radiance = 0u;
	                r.weight_sum      = 0.0f;
	                r.target_pdf      = 0.0f;
	                r.packed_meta     = 0u;
	            };
	            // RTXDI parity: detect fireflies via the per-reservoir shade contribution
	            // proxy using the RAW weight_sum, matching RTXDI GI/BoilingFilter.hlsli:27
	            // (luminance(rad) * weightSum, uncapped). The shade path still applies the
	            // min(W, 20) cap from GIReservoir.h::weight() — that's a separate concern.
	            // Using weight() here would clip the firefly signature down into the noise
	            // floor and the BF couldn't distinguish a true firefly from a normal bright
	            // sample. Mask by in_bounds so OOB threads contribute zero to reductions.
	            Float contrib = ite(in_bounds, luminance(r->rad()) * r.weight_sum, 0.0f);
	            Bool  valid   = ite(in_bounds, r->is_valid(), false);

	            UInt warp_id = local_id / 32u;
	            UInt lane_id = local_id % 32u;

	            // Warp-level reduction -- no sync needed within a warp
	            Float warp_total_w = warp_active_sum(contrib);
	            UInt  warp_total_v = warp_active_sum(ite(valid, 1u, 0u));

	            // Lane 0 writes per-warp result to shared memory (8 entries, 64 bytes)
	            $if(lane_id == 0u) {
	                shared_warp_w[warp_id] = warp_total_w;
	                shared_warp_v[warp_id] = warp_total_v;
	            };
	            sync_block();

	            // 8 cross-warp results: one smem entry per lane + warp shuffle
	            // reduction (was lane 0's serial 8-iteration sum while the rest
	            // of the warp idled). Index clamped so non-reader lanes stay in
	            // bounds; ite masks their contribution to zero. Note: shuffle
	            // reduction reorders the additions vs the serial loop, so avg_w
	            // can differ at ULP level.
	            Bool sum_reader = lane_id < 8u;
	            Float lane_w = ite(sum_reader, shared_warp_w[min(lane_id, 7u)], 0.0f);
	            UInt  lane_v = ite(sum_reader, shared_warp_v[min(lane_id, 7u)], 0u);
	            Float total_w = warp_active_sum(lane_w);
	            UInt  total_v = warp_active_sum(lane_v);

	            // warp_active_sum broadcasts the total to every lane directly
	            Float avg_w = total_w / max(cast<float>(total_v), 1.0f);

	            // Outlier test runs everywhere including screen edges. OOB threads
	            // already have valid=false from the in_bounds gate above, so they
	            // never enter this branch.
	            Float multiplier = 10.0f / max(strength, 1e-6f) - 9.0f;
	            $if(valid & (contrib > avg_w * multiplier)) {
	                r->set_M(0u);
	                r.weight_sum = 0.0f;
	                r.target_pdf = 0.0f;
	            };

	            gi_reservoir_buffer.write(pixel_index, r);
	        });
	        } // end resolverOnly gate (BoilingFilterGI)

		}
	//==========================================================================
	// PassGI::createImages
	//==========================================================================

	void PassGI::createImages(luisa::compute::Device& device, uint width, uint height) {
		uint rsv_w = _checkerboard ? (width + 1u) / 2u : width;
		uint pixel_count = rsv_w * height;
		_giResBuf[0] = device.create_buffer<GIReservoir>(pixel_count);
		_giResBuf[1] = device.create_buffer<GIReservoir>(pixel_count);

	_halfWidth  = (width + 1u) / 2u;
	_halfHeight = (height + 1u) / 2u;
	uint half_pixel_count = _halfWidth * _halfHeight;
	_giResBufHalf[0]      = device.create_buffer<GIReservoir>(half_pixel_count);
	_giResBufHalf[1]      = device.create_buffer<GIReservoir>(half_pixel_count);
	_giReservoirBufferFull = device.create_buffer<GIReservoir>(pixel_count);
	_giInitialSnapshot     = device.create_buffer<GIReservoir>(pixel_count);
	_giParamsBuf           = device.create_buffer<GIParams>(1u);
}

void PassGI::release() {
    _giResBuf[0].release();
    _giResBuf[1].release();
    _giResBufHalf[0].release();
    _giResBufHalf[1].release();
    _giReservoirBufferFull.release();
    _giInitialSnapshot.release();
}

	//==========================================================================
	// PassGI::_populateGiParams
	//==========================================================================
	void PassGI::_populateGiParams(luisa::compute::CommandList& cmdlist) noexcept {
		// FPS-aware accumulation: overwrite _giMaxAge in place from giAccumulationTime.
		if (_giAccumulationTimeEnabled) {
			float fps = util::clampSmoothedFps(
				ci::app::getWindow()->getApp()->getAverageFps());
			_giMaxAge = util::computeAccumulatedFrames(_giAccumulationTime, fps);
		}
		_giParamsCpu.giSpatialRadius                = _giSpatialRadius;
		_giParamsCpu.giSpatialNeighborCount         = kGiSpatialNeighborCount;
		_giParamsCpu.giWSumCap                      = _giWSumCap;
		_giParamsCpu.giMaxAge                       = _giMaxAge;
		_giParamsCpu.giTemporalMaxM                 = _giTemporalMaxM;
		_giParamsCpu.giSpatialMaxM                  = _giSpatialMaxM;
		_giParamsCpu.giMaxRadiance                  = _giMaxRadiance;
		_giParamsCpu.giRoughnessInvalidationThresh  = _giRoughnessInvalidationThresh;
		_giParamsCpu.giMatSimRoughness              = _giMatSimRoughness;
		_giParamsCpu.giMatSimF0                     = _giMatSimF0;
		_giParamsCpu.giMatSimAlbedo                 = _giMatSimAlbedo;
		_giParamsCpu.giMISRoughness                 = _giMISRoughness;
		_giParamsCpu.giBiasCorrectionDivergenceThresh = _giBiasCorrectionDivergenceThresh;
		_giParamsCpu.spatialNormalThresh            = _spatialNormalThresh;
		_giParamsCpu.spatialDepthThresh             = _spatialDepthThresh;
		_giParamsCpu.giTier3BiasCorrectionEnabled   = _giTier3BiasCorrectionEnabled ? 1u : 0u;
		_giParamsCpu.giSuppressAwayFromLight        = _giSuppressAwayFromLight ? 1u : 0u;
		_giParamsCpu.giSuppressAwayFromLightThresh  = _giSuppressAwayFromLightThresh;
		cmdlist << _giParamsBuf.copy_from(&_giParamsCpu);
	}

	//==========================================================================
	// PassGI::renderInitial
	//==========================================================================

	void PassGI::renderInitial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
		_populateGiParams(cmdlist);
		auto& ls = ctx.lightSampler;
		auto& geom = ctx.geometry;
		uint dw = _giHalfRes ? _halfWidth  : ctx.width;
		uint dh = _giHalfRes ? _halfHeight : ctx.height;
		// Checkerboard halving only for full-res dispatch (half-res initial fills entire buffer)
		if (_checkerboard && !_giHalfRes) dw = (dw + 1u) / 2u;
		auto& buf = _giHalfRes ? halfReservoirBuffer() : _giResBuf[_giResIdx];
		cmdlist << _giInitialShader(
			_giParamsBuf,
			buf,
			ctx.gbufDepth, ctx.gbufVis, ctx.gbufBaryMotion,
			ctx.seedImage,
			ctx.geometry.tlas(),
			ctx.camera,
			SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() },
			geom.vertex_bindless(),
			ctx.materialPool.textures(),
			LightSamplingResources{ ls.triangle_buffer(), ls.vertex_buffer(),
			    ls.alias_table(), ls.emissive_triangle_count(),
			    ls.total_power_inv(), ls.emissive_count_inv(),
			    ls.instance_to_light_base() },
			EnvLightResources{ ls.envmap_image(), ls.env_cdf_marginal(),
			    ls.env_cdf_conditional(), ls.env_integral(),
			    ls.env_width(), ls.env_height(),
			    ls.env_rotation_matrix() },
			ls.env_exposure(),
			*_presampleEnvTilesPtr,
			_presampleEnvTotalEntries,
			ctx.cbField,
			ctx.glassThroughput,
			_giOneBounce ? 1u : 0u,
			_giHalfRes ? 2u : 1u,
			_giSuppressAwayFromLight ? 1u : 0u,
			_giSuppressAwayFromLightThresh,
			ctx.geometry.has_transparent_shadow_casters() ? 1u : 0u
#if NT_ENABLE_PROCEDURAL
	        , *_procBindlessPtr
#endif
		).dispatch(dw, dh);
	}

	//==========================================================================
	// PassGI::renderTemporal
	//==========================================================================

	void PassGI::renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
		auto& geom = ctx.geometry;
		// Half-res mode: temporal runs on the half-res pair (dense packing, no
		// checkerboard) with the half-res history as prev.
		auto& cur  = _giHalfRes ? halfReservoirBuffer()      : reservoirBuffer();
		auto& prev = _giHalfRes ? halfReservoirPrevBuffer()  : reservoirPrevBuffer();
		uint dw = _giHalfRes ? _halfWidth  : (_checkerboard ? (ctx.width + 1u) / 2u : ctx.width);
		uint dh = _giHalfRes ? _halfHeight : ctx.height;
		cmdlist << _giTemporalReuseShader(
			_giParamsBuf,
			cur,
			prev,
			ctx.gbufDepth, ctx.gbufVis, ctx.gbufBaryMotion,
			ctx.frameCount,
			ctx.camera,
			SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() },
			geom.vertex_bindless(),
			ctx.materialPool.textures(),
			ctx.cbField,
			ctx.glassThroughput,
			ctx.gbufDepthPrev, ctx.gbufVisPrev, ctx.denoiseNormalPrev,
			_giTier3BiasCorrectionEnabled ? 1u : 0u,
			_giHalfRes ? 2u : 1u
#if NT_ENABLE_PROCEDURAL
			, *_procBindlessPtr
#endif
		).dispatch(dw, dh);
	}

	//==========================================================================
	// PassGI::renderBoiling
	//==========================================================================

	void PassGI::renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength) {
		// Half-res mode: boiling runs on the half-res buffer (dense packing).
		auto& buf = _giHalfRes ? halfReservoirBuffer() : reservoirBuffer();
		uint dw = _giHalfRes ? _halfWidth  : (_checkerboard ? (ctx.width + 1u) / 2u : ctx.width);
		uint dh = _giHalfRes ? _halfHeight : ctx.height;
		cmdlist << _boilingFilterGI(
			buf,
			strength,
			ctx.cbField
		).dispatch(dw, dh);
	}

	//==========================================================================
	// PassGI::renderSpatial
	//==========================================================================

	void PassGI::renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
		auto& geom = ctx.geometry;
		// Spatial reads from current slot, writes to the other slot (no copy needed)
		cmdlist << _giSpatialReuseShader(
			_giParamsBuf,
			reservoirPrevBuffer(),          // 1: output (write to the free slot)
			reservoirBuffer(),              // 1: input (read from current slot),
			ctx.gbufDepth, ctx.gbufVis, ctx.gbufBaryMotion,
			geom.tlas(),
			ctx.frameCount,
			SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() },
			geom.vertex_bindless(),
			ctx.materialPool.textures(),
			ctx.camera,
			ctx.cbField,
			ctx.glassThroughput,
			ctx.denoiseNormal,
			_giTier3BiasCorrectionEnabled ? 1u : 0u
#if NT_ENABLE_PROCEDURAL
			, *_procBindlessPtr
#endif
		).dispatch(_checkerboard ? (ctx.width + 1u) / 2u : ctx.width, ctx.height);
	}

	//==========================================================================
	// PassGI::copyReservoirs
	//==========================================================================

	void PassGI::copyReservoirs(luisa::compute::CommandList& cmdlist) {
		// Debug early-return paths: copy this frame's output to the history
		// slot so next frame's temporal sees it (mimics the end-of-frame flip).
		if (_giHalfRes) cmdlist << halfReservoirBuffer().copy_to(halfReservoirPrevBuffer());
		else            cmdlist << reservoirBuffer().copy_to(reservoirPrevBuffer());
	}

	//==========================================================================
	// PassGI::snapshotInitial
	//==========================================================================

	void PassGI::snapshotInitial(luisa::compute::CommandList& cmdlist) noexcept {
		// Frozen copy of current reservoirs (post-renderInitial, pre-renderTemporal)
		// consumed by shade shader for RTXDI-style initial-vs-final MIS.
		cmdlist << _giInitialSnapshot.copy_from(reservoirBuffer());
	}

	//==========================================================================
	// PassGI::renderUpsample
	//==========================================================================

	void PassGI::renderUpsample(luisa::compute::CommandList& cmdlist, uint width, uint height) {
		// Runs AFTER temporal+boiling in half-res mode, so read the current
		// half-res slot those passes wrote.
		cmdlist << _giUpsampleShader(
			_giReservoirBufferFull,
			halfReservoirBuffer(),
			_halfWidth,
			_halfHeight
		).dispatch(_checkerboard ? (width + 1u) / 2u : width, height);
	}

	//==========================================================================
	// PassGI::drawUi
	//==========================================================================

	void PassGI::drawUi() {
	    if (ImGui::CollapsingHeader("GI")) {
	        ImGui::Checkbox("Enable GI", &_enabled);
	        if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Master toggle for the ReSTIR GI pass.");
	        if (_enabled) {
	            ImGui::Checkbox("GI 1-Bounce", &_giOneBounce);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = 1 bounce (faster, darker). OFF = 2 bounces.");
	            ImGui::Checkbox("GI Half-Resolution", &_giHalfRes);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = initial+temporal+boiling run at half res, upsampled after boiling (~2x faster than before with checkerboard, ~4x without). OFF = full-res.");
	            ImGui::Checkbox("GI Tier 3 Bias Correction", &_giTier3BiasCorrectionEnabled);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = reject cross-surface fireflies (may dim low-roughness metal). OFF by default.");
	            ImGui::SliderFloat("GI Bias Correction Divergence Thresh", &_giBiasCorrectionDivergenceThresh, 0.0f, 10.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = skip bias correction more often (preserves diffuse brightness, less firefly rejection). Lower = always apply.");
	            ImGui::SliderFloat("GI MIS Roughness", &_giMISRoughness, 0.0f, 1.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = stronger MIS (cleaner glossy bounces, dimmer). Lower = pure GI sampling (brighter, noisier).");
	            ImGui::Checkbox("Delta Branch NEE (mirror metals)", &_deltaBranchNEEEnabled);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = run NEE at mirror-hit x2 for r<kMinRoughness metals (closes the 'reflected geometry stays black' gap). OFF by default = bit-identical to baseline.");
	            ImGui::Checkbox("GI Suppress Away From Light", &_giSuppressAwayFromLight);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("DEBUG. ON = zero GI on surfaces facing away from light. Leave OFF for normal use.");
	            ImGui::SliderFloat("GI Suppress Away-From-Light Thresh", &_giSuppressAwayFromLightThresh, -1.0f, 1.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = suppress GI on more grazing surfaces. Lower = stricter (only near-vertical-to-light).");
	            ImGui::SliderFloat("GI Roughness Invalidation Thresh", &_giRoughnessInvalidationThresh, 0.0f, 0.1f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = allow merges across larger roughness gaps (more reuse, possible leaking). Lower = sharper edges, more noise.");
	            ImGui::SliderFloat("GI Material Similarity Roughness", &_giMatSimRoughness, 0.0f, 1.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = allow merges between more different-roughness materials. Lower = stricter matching.");
	            ImGui::SliderFloat("GI Material Similarity F0", &_giMatSimF0, 0.0f, 1.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = allow merges between more different-F0 materials. Lower = stricter.");
	            ImGui::SliderFloat("GI Material Similarity Albedo", &_giMatSimAlbedo, 0.0f, 1.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = allow merges between more different-albedo materials. Lower = stricter.");
	            ImGui::SliderFloat("GI Max Radiance (firefly clamp)", &_giMaxRadiance, 1.0f, 20.0f);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Clamps x2 bounce radiance at initial. Lower = fewer/shorter fireflies on disocclusion; too low darkens legitimate bright GI.");
	            ImGui::Checkbox("FPS-aware GI accumulation", &_giAccumulationTimeEnabled);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("When ON, GI Max Age below is derived each frame from giAccumulationTime * live FPS.\nAlso drives DI visMaxAge (with /15 ratio). Wall-clock GI lag stays constant across framerates.");
	            if (_giAccumulationTimeEnabled) {
	                ImGui::SameLine();
	                ImGui::PushItemWidth(140.f);
	                ImGui::SliderFloat("GI accum (s)", &_giAccumulationTime, 0.1f, 1.5f, "%.3f");
	                if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Target GI temporal-blur tau in seconds. 0.5 = 30 frames @ 60 fps.");
	                ImGui::PopItemWidth();
	            }
	            ImGui::BeginDisabled(_giAccumulationTimeEnabled);
	            ImGui::SliderInt("GI Max Age (frames)", reinterpret_cast<int*>(&_giMaxAge), 1, 60);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Max reservoir age before forced reset. Higher = longer temporal reuse = faster disocclusion recovery. RTXDI default 30.");
	            ImGui::EndDisabled();
	            ImGui::SliderInt("GI Spatial Radius (px)", reinterpret_cast<int*>(&_giSpatialRadius), 4, 64);
	            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Spatial-reuse neighbor disk radius. Higher = more candidates at screen-edge disocclusion; too high bleeds across edges. RTXDI default 32.");
	        }
	    }
	}

	//==========================================================================
	// PassGI::Config serialization
	//==========================================================================

	void PassGI::toJson(ci::Json& j) const {
	    j = ci::Json{
	        {"enabled",                _enabled},
	        {"giOneBounce",            _giOneBounce},
	        {"giHalfRes",              _giHalfRes},
	        {"giSpatialRadius",        _giSpatialRadius},
	        {"giWSumCap",              _giWSumCap},
	        {"giMaxAge",               _giMaxAge},
	        {"giTemporalMaxM",         _giTemporalMaxM},
	        {"giSpatialMaxM",          _giSpatialMaxM},
	        {"giMaxRadiance",          _giMaxRadiance},
	        {"giRoughnessInvalidationThresh", _giRoughnessInvalidationThresh},
	        {"giMatSimRoughness",  _giMatSimRoughness},
	        {"giMatSimF0",         _giMatSimF0},
	        {"giMatSimAlbedo",     _giMatSimAlbedo},
	        {"giMISRoughness",          _giMISRoughness},
	        {"deltaBranchNEEEnabled",   _deltaBranchNEEEnabled},
	        {"giTier3BiasCorrectionEnabled", _giTier3BiasCorrectionEnabled},
	        {"giBiasCorrectionDivergenceThresh", _giBiasCorrectionDivergenceThresh},
	        {"giSuppressAwayFromLight", _giSuppressAwayFromLight},
	        {"giSuppressAwayFromLightThresh", _giSuppressAwayFromLightThresh},
	        {"spatialNormalThresh",    _spatialNormalThresh},
	        {"spatialDepthThresh",     _spatialDepthThresh},
	        {"giAccumulationTimeEnabled", _giAccumulationTimeEnabled},
	        {"giAccumulationTime",     _giAccumulationTime},
	    };
	}

	void PassGI::fromJson(const ci::Json& j) {
	    if (!j.is_object()) return;
	    _enabled                = j.value("enabled",                _enabled);
	    _giOneBounce            = j.value("giOneBounce",            _giOneBounce);
	    _giHalfRes              = j.value("giHalfRes",              _giHalfRes);
	    _giSpatialRadius        = j.value("giSpatialRadius",        _giSpatialRadius);
	    _giWSumCap              = j.value("giWSumCap",              _giWSumCap);
	    _giMaxAge               = j.value("giMaxAge",               _giMaxAge);
	    _giTemporalMaxM         = j.value("giTemporalMaxM",         _giTemporalMaxM);
	    _giSpatialMaxM          = j.value("giSpatialMaxM",          _giSpatialMaxM);
	    _giMaxRadiance          = j.value("giMaxRadiance",          _giMaxRadiance);
	    _giRoughnessInvalidationThresh = j.value("giRoughnessInvalidationThresh", _giRoughnessInvalidationThresh);
	    _giMatSimRoughness  = j.value("giMatSimRoughness",  _giMatSimRoughness);
	    _giMatSimF0         = j.value("giMatSimF0",         _giMatSimF0);
	    _giMatSimAlbedo     = j.value("giMatSimAlbedo",     _giMatSimAlbedo);
		_giMISRoughness          = j.value("giMISRoughness",          _giMISRoughness);
		_deltaBranchNEEEnabled   = j.value("deltaBranchNEEEnabled",   _deltaBranchNEEEnabled);
		_giTier3BiasCorrectionEnabled = j.value("giTier3BiasCorrectionEnabled", _giTier3BiasCorrectionEnabled);
		_giBiasCorrectionDivergenceThresh = j.value("giBiasCorrectionDivergenceThresh", _giBiasCorrectionDivergenceThresh);
	    _giSuppressAwayFromLight = j.value("giSuppressAwayFromLight", _giSuppressAwayFromLight);
	    _giSuppressAwayFromLightThresh = j.value("giSuppressAwayFromLightThresh", _giSuppressAwayFromLightThresh);
	    _spatialNormalThresh    = j.value("spatialNormalThresh",    _spatialNormalThresh);
	    _spatialDepthThresh     = j.value("spatialDepthThresh",     _spatialDepthThresh);
	    _giAccumulationTimeEnabled = j.value("giAccumulationTimeEnabled", _giAccumulationTimeEnabled);
	    _giAccumulationTime    = j.value("giAccumulationTime",     _giAccumulationTime);
	}

	}
