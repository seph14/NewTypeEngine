#include "newtype/render/PassDI.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"
#include "newtype/render/BSDF.h"
#include "newtype/render/MaterialSimilarity.h"
#include "newtype/render/ProceduralTrace.h"
#include "newtype/util/UiHelper.h"
#include "newtype/util/AccumulationTime.h"
#include "newtype/core/Renderer.h" // zeroInitBuffers() submits on Renderer::stream()
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>

namespace newtype::core {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace newtype::scene;
    using namespace newtype::render;

    // Sentinel for environment light in reservoir light_idx
    static constexpr uint kEnvLightSentinel = ~1u;  // 0xFFFFFFFE

    /// Pre-fetched light sample — captures every bindless read that p_hat
    /// evaluation needs for a given canonical reservoir, so spatial reuse
    /// can hoist those reads out of the neighbor loop (the canonical sample
    /// is the same across all 8-16 neighbor iterations).
    struct LightSample {
        Bool   is_env;
        // Local light fields (valid when !is_env)
        Float3 emission;
        Float3 light_normal;  // precomputed geometric normal from TriangleLight
        Float3 light_point;   // precomputed barycentric position on the light
        Float  pdf;           // 0 for hidden lights — disables p_hat
        // Env light fields (valid when is_env)
        Float3 env_radiance;
        Float3 env_dir;
    };

    /// Fetch all bindless data needed to evaluate p_hat for a reservoir's
    /// selected light. Performs the reads once; the resulting LightSample
    /// can be passed to evaluate_p_hat_from_sample at any surface position
    /// without further bindless traffic.
    [[nodiscard]] inline LightSample fetch_light_sample(
        Bool    is_env,
        UInt    light_idx,
        Float   light_bary_u, Float light_bary_v,
        const BufferVar<LightSampler::TriangleLight>      &triangle_lights,
        const BufferVar<LightSampler::TriangleVertexData> &triangle_vertices,
        const ImageVar<float> &envmap, UInt env_w, UInt env_h,
        Float3x3 env_rot, Float env_exp) noexcept {

        LightSample s;
        s.is_env       = is_env;
        s.emission     = def<luisa::float3>(0.0f, 0.0f, 0.0f);
        s.light_normal = def<luisa::float3>(0.0f, 0.0f, 0.0f);
        s.light_point  = def<luisa::float3>(0.0f, 0.0f, 0.0f);
        s.pdf          = def(0.0f);
        s.env_radiance = def<luisa::float3>(0.0f, 0.0f, 0.0f);
        s.env_dir      = def<luisa::float3>(0.0f, 1.0f, 0.0f);

        $if(is_env) {
            s.env_radiance = eval_envmap_from_uv(light_bary_u, light_bary_v, envmap, env_w, env_h, env_exp);
            s.env_dir      = uv_to_direction(light_bary_u, light_bary_v, env_w, env_h, env_rot);
        } $else {
            auto tri_light = triangle_lights.read(light_idx);
            auto verts     = triangle_vertices.read(light_idx);
            s.emission     = tri_light->emission();
            s.light_normal = tri_light->normal();
            s.pdf          = tri_light.pdf;
            Float b0       = 1.0f - light_bary_u - light_bary_v;
            s.light_point  = b0 * verts.v0 + light_bary_u * verts.v1 + light_bary_v * verts.v2;
        };
        return s;
    }

    /// Evaluate p_hat from a pre-fetched LightSample — zero bindless reads.
    /// Position/normal/view-dependent; everything else is baked into `s`.
    /// BSDF-flavored: pass the caller's pre-built MaterialBSDF so coat/fuzz
    /// /composed-LobeList propagate into target_pdf.
    [[nodiscard]] inline Float evaluate_p_hat_from_sample(
        const MaterialBSDF& bsdf,
        const LightSample &s,
        Float3 world_pos, Float3 ns, Float3 wo) noexcept {

        Float p_hat = def(0.0f);
        $if(s.is_env) {
            p_hat = evaluate_p_hat_env(
                bsdf, s.env_radiance, s.env_dir, ns, wo);
        } $else {
            $if(s.pdf > 0.0f) {
                p_hat = evaluate_p_hat(
                    bsdf,
                    s.emission, s.light_normal, s.light_point, world_pos, ns, wo);
            };
        };
        return p_hat;
    }

    /// RTXDI MFactor (ref: Rtxdi/Utils/Math.hlsli:117).
    /// Asymmetric sample-count multiplier for pairwise-style neighbor streaming:
    /// only penalizes when the destination PDF (q1) is smaller than the source
    /// PDF (q0). When q1 >= q0, returns 1.0 (no penalty). With exponent=8 this
    /// drops a 0.5 ratio to 0.004, vs ~0.5 for the soft log2(jac)/4 fade it
    /// replaces — catches rotation-driven BRDF-lobe misalignment on narrow metals.
    [[nodiscard]] inline Float rtxdi_mfactor(Float q0, Float q1, Float exponent) noexcept {
        Float ratio = min(q1 / max(q0, 1e-20f), 1.0f);
        return clamp(luisa::compute::pow(ratio, exponent), 0.0f, 1.0f);
    }

    void PassDI::release() {
        _geom = nullptr;
        _resBuf[0].release();
        _resBuf[1].release();
        _presampleLocalTiles.release();
        _presampleEnvTiles.release();
        _boilingStatsBuf.release();
    }

    void PassDI::compile(luisa::compute::Device& device, scene::Geometry& geom,
                           const render::SurfaceResolverPoly& resolver, bool checkerboard) {
        _geom = &geom;
        _checkerboard = checkerboard;
        compileImpl(device, resolver, /*resolverOnly=*/false);
    }

    void PassDI::recompileCallables(luisa::compute::Device& device,
                                    const render::SurfaceResolverPoly& resolver) {
        compileImpl(device, resolver, /*resolverOnly=*/true);
    }

    void PassDI::compileImpl(luisa::compute::Device& device,
                             const render::SurfaceResolverPoly& resolver, bool resolverOnly) {
        // Bake the compile-time specialization snapshot: these are the exact
        // values the former UInt kernel args (diBiasCorrectionEnabled /
        // hasTransparentShadowCasters) would have carried this frame, so the
        // specialized shaders are bit-identical to the runtime-arg form.
        // Flips are detected by specializationChanged() at the render-thread
        // safe point and recompile through recompileCallables.
        _bakedDiBiasCorrectionEnabled = _diBiasCorrectionEnabled ? 1u : 0u;
        _bakedTransparentShadowCasters =
            (_geom != nullptr && _geom->has_transparent_shadow_casters()) ? 1u : 0u;
        _bakedHasGlass =
            (_geom != nullptr && _geom->has_visible_glass()) ? 1u : 0u;
        _bakedCgnsEnabled = _cgnsEnabled ? 1u : 0u;
        // C++ const so DXC folds the dead branch (the _checkerboard idiom):
        // has_glass gates the unjit classification trace, the medium-list
        // init, and both PSR glass branches — glassless scenes compile them
        // out entirely (perf review R2 item 8).
        const uint has_glass = _bakedHasGlass;
        //------------------------------------------------------------------
        // Phase 2 fixed presample pools (resolution-independent, ~1 MB each at
        // 32 B/entry vs the former ~73 MB @1080p screen-tile pools). Lazily
        // allocated once — compileImpl also runs for recompileCallables, and a
        // mid-frame re-create would orphan the buffers bound into the current
        // command list. Pools survive resizes; the dirty gates trigger the
        // first fill after allocation.
        //------------------------------------------------------------------
        // Validity check (not .size()==0): the buffers are default-constructed
        // (invalid) until the first allocation, and Buffer::size() asserts on
        // an invalid buffer — Debug aborted on the first compile (crash
        // 2026-09-19). operator bool is assert-free and also handles release().
        if (!_presampleLocalTiles) {
            _presampleLocalTiles = device.create_buffer<PresampledCandidate>(
                kPresamplePoolTileCount * kPresampleTileSize);
            _presampleEnvTiles = device.create_buffer<PresampledCandidate>(
                kPresamplePoolTileCount * kPresampleEnvTileSize);
            _presampleLocalDirty = true;
            _presampleEnvDirty = true;
        }
        if (!_boilingStatsBuf) {
            _boilingStatsBuf = device.create_buffer<float>(2u);
        }
        //==========================================================================
        // Reservoir zero-fill kernel (see ZeroReservoirShaderType note in
        // PassDI.h): one thread per entry, all fields zero — M()==0 == invalid.
        //==========================================================================
        _zeroReservoirShader = device.compile<1>([&](
            BufferVar<Reservoir> buf,
            UInt                 count
        ) noexcept {
            set_name("DI_ZeroReservoirs");
            set_block_size(256u, 1u, 1u);
            UInt i = dispatch_id().x;
            $if(i < count) {
                Var<Reservoir> z;
                // P1-11: light_idx = ~0u (no sample). 0u is a LEGAL triangle
                // index — only safe while M==0 gates every consumer; the
                // empty-but-alive semantics (P1-1) make the explicit sentinel
                // required at any future fetch site.
                z.light_idx     = ~0u;
                z.w_sum         = 0.0f;
                z.target_pdf    = 0.0f;
                z.light_bary_u  = 0.0f;
                z.light_bary_v  = 0.0f;
                z.packed_meta   = 0u;
                buf.write(i, z);
            };
        });
        //==========================================================================
        // G-Buffer kernel (primary ray pass + PSR trace-through + alpha cutout)
        //==========================================================================
        _gbufShader = device.compile<2>([&](
            ImageFloat              gbuf_depth,
            ImageUInt               gbuf_vis,
            ImageFloat              gbuf_bary_motion,
            ImageFloat              glass_throughput,
            ImageFloat              gbuf_velocity,
            ImageFloat              gbuf_depth_upscale,
            Var<util::CameraData>   camera,
            AccelVar                accel,
            Var<SceneGeometryResources> scene,
            BindlessVar             vertex_bindless,
            BindlessVar             tex_bindless,
            UInt                    psr_bounce_budget
#if NT_ENABLE_PROCEDURAL
	            , BindlessVar proc_bindless
	        #endif
#if NT_ENABLE_SHARC
            ,
            ImageFloat              rough_glass_info
#endif
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("DI_GBuffer");
            UInt2 coord      = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();

            // Guard against partial-block threads
            $if(any(coord >= resolution)) { $return(); };

            auto ray = camera->generate_ray(Expr{
                (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f
            });

            // Save original camera ray for virtual depth computation (PSR)
            Float3 cam_origin = ray->origin();
            Float3 cam_dir    = ray->direction();

            // Stable glass classification: trace unjittered pixel-center ray to
            // decide if this pixel is glass. Glass pixels follow the unjit ray
            // throughout the PSR loop (no silhouette flicker, all derived values
            // stable for free). Opaque pixels keep the jittered ray for TAA.
            // Silhouette pixels where unjit says opaque but jit refracts through
            // glass still shimmer — accepted trade for preserving TAA on opaque.
            // PSR iteration-0 seed (perf review R2 item 7a): when this pixel
            // classifies as glass, `ray` below IS the post-invisible-skip unjit
            // classification ray, and the PSR loop's first iteration would
            // re-trace that identical ray — re-hitting and re-skipping the
            // same camera-invisible geometry. The classification hit is
            // stashed here so iteration 0 can consume it instead. Bit-
            // identical by construction: deterministic trace of the same ray.
            // (Only written when has_glass; stays false in glassless
            // specializations, so DXC folds the seed branch out entirely.)
            Bool   glass_seeded = def(false);
            Float  seed_depth   = def(0.0f);
            UInt   seed_inst    = def(0u);
            UInt   seed_prim    = def(0u);
            Float2 seed_bary    = def(make_float2(0.0f));
            Bool   seed_miss    = def(true);
#if NT_ENABLE_PROCEDURAL
            Bool   seed_is_procedural = def(false);
            UInt   seed_local_tri     = def(0u);
            Float2 seed_local_bary    = def(make_float2(0.0f));
#endif
            // C++-time gate (has_glass is the baked specialization constant):
            // glassless scenes never emit the unjit trace at all.
            if (has_glass != 0u) {
                auto ray_unjit_first = camera->generate_ray(Expr{
                    (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter
                });
                auto unjit_first_hit = render::trace_closest(accel, ray_unjit_first
#if NT_ENABLE_PROCEDURAL
                    , proc_bindless
#endif
                );
                // Pass through camera-invisible geometry: an invisible light in
                // front of glass must not defeat the classification (the light
                // would read as type-5, the pixel would keep its jittered ray).
                // Retraces only while the current hit is invisible, bounded at
                // 4 skips — zero extra cost for scenes without such shapes.
                $for(unjit_skip, 4u) {
                    Bool unjit_hit_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                    unjit_hit_proc = unjit_first_hit.is_procedural;
#endif
                    Bool unjit_hit_invisible = def(false);
                    $if(!unjit_first_hit->miss() & !unjit_hit_proc) {
                        UInt4 uskip = scene.instance_buffer.read(unjit_first_hit.inst);
                        unjit_hit_invisible = (uskip.x & kCameraInvisibleFlag) != 0u;
                    };
                    $if(unjit_hit_invisible) {
                        Float uskip_off = max(0.001f * unjit_first_hit.committed_ray_t, 1e-4f);
                        Float3 uskip_pos = ray_unjit_first->origin()
                            + ray_unjit_first->direction() * unjit_first_hit.committed_ray_t;
                        ray_unjit_first = make_ray(uskip_pos + ray_unjit_first->direction() * uskip_off,
                            ray_unjit_first->direction(), 0.0f, 1e10f);
                        unjit_first_hit = render::trace_closest(accel, ray_unjit_first
#if NT_ENABLE_PROCEDURAL
                            , proc_bindless
#endif
                        );
                    } $else { $break; };
                };
                Bool unjit_first_is_glass = def(false);
                $if(!unjit_first_hit->miss()) {
#if NT_ENABLE_PROCEDURAL
                    // Procedural first hit: inst is the TLAS slot of the proc BLAS
                    // (out of range for instance_buffer); prim is the proc AABB index.
                    $if(unjit_first_hit.is_procedural) {
                        Var<scene::ProcInstanceData> proc_inst =
                            proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                                .read(unjit_first_hit.prim);
                        auto umat = scene.material_buffer.read(Expr{ proc_inst.material_layers & 0xFFu });
                        UInt ubsdf = get_effective_bsdf_type(umat);
                        unjit_first_is_glass = (ubsdf == 3u) | (ubsdf == 11u);
                    } $else {
#endif
                        UInt4 uinst = scene.instance_buffer.read(unjit_first_hit.inst);
                        auto  umat  = scene.material_buffer.read(Expr{ uinst.y & 0xFFu });
                        UInt  ubsdf = get_effective_bsdf_type(umat);
                        unjit_first_is_glass = (ubsdf == 3u) | (ubsdf == 11u);
#if NT_ENABLE_PROCEDURAL
                    };
#endif
                };
                $if(unjit_first_is_glass) {
                    ray        = ray_unjit_first;
                    cam_origin = ray->origin();
                    cam_dir    = ray->direction();
                    // Hand the classification hit to PSR iteration 0 (item 7a).
                    glass_seeded = true;
                    seed_depth   = unjit_first_hit.committed_ray_t;
                    seed_inst    = unjit_first_hit.inst;
                    seed_prim    = unjit_first_hit.prim;
                    seed_bary    = unjit_first_hit.bary;
                    seed_miss    = unjit_first_hit->miss();
#if NT_ENABLE_PROCEDURAL
                    seed_is_procedural = unjit_first_hit.is_procedural;
                    seed_local_tri     = unjit_first_hit.local_tri;
                    seed_local_bary    = unjit_first_hit.local_bary;
#endif
                };
            }

            // Accumulated glass throughput (RGB = absorption, A = accumulated Fresnel reflectivity)
            Float3 throughput    = def(make_float3(1.0f));
            Float  fresnel_accum = def(0.0f);

            Float  depth    = def(3.402823466e38f);
            UInt   inst_id  = def(~0u);
            UInt   prim_id  = def(~0u);
            Float2 bary     = def(make_float2(0.0f));
            Float2 motion   = def(make_float2(0.0f));
            UInt   glass_bounces = def(0u);
            // True when the loop's last iteration continued the ray ($continue)
            // instead of terminating on a hit/miss — i.e. the budget ran out
            // while the ray was still in flight. A genuine miss or opaque hit
            // clears it; see the straight-through fallback after the loop.
            Bool psr_pending = def(false);
            // Object-space hit position of the final opaque hit, for object
            // motion vectors (reprojected through the prev instance transform).
            Float3 obj_pos   = def(make_float3(0.0f));

#if NT_ENABLE_SHARC
            // Phase 3 rough-glass classification: lobe parameters of the FIRST
            // visible (true-hit) dielectric interface, written to the side image
            // for the post-denoise gather. Roughness > kRoughGlassEpsilon at the
            // gather side flags the pixel as rough glass; the deterministic PSR
            // chain below is unchanged (it stays the smooth-mean fallback while
            // the cache is cold — plan §7 Phase 3).
            Float first_glass_rough = def(0.0f);
            Float first_glass_eta   = def(1.0f);   // eta_i/eta_t at the interface
            Float first_glass_thin  = def(0.0f);   // 1 = ThinDielectric
#if NT_ENABLE_DISPERSION
            Float first_glass_disp  = def(0.0f);   // Abbe V of the first interface
#endif                                           // (side image .w; 0 = off)
#endif

            using Triangle = luisa::compute::Triangle;
            using Vertex   = newtype::util::Vertex;

            // PSR trace-through loop: continue through dielectrics and alpha-cutout surfaces.
            // The budget counts every dielectric interface crossing (each costs one
            // iteration) and the background surface needs one more. A cube inside a
            // glass sphere is already 4 crossings + background = 5, so the budget
            // must exceed the plain single-object 3: 8 covers three nested volumes
            // (6 crossings) plus TIR/false-hit headroom. The bound arrives as the
            // runtime psr_bounce_budget arg (value PassDI::kMaxGlassBounces) so
            // DXC cannot statically unroll the ~200-line body.

            // --- Nested-dielectric medium list (Schmidt & Budge 2002) ---
            // Tracks the dielectric media the PSR ray has entered so every
            // interface evaluates Fresnel/refraction against the actual
            // surrounding IORs instead of hard-coding air. Entry priority is
            // the dielectric's material.metallic (dead for type 3): lower
            // value = higher priority, 0 = default. A surface whose interior
            // priority loses to the best medium on the list is a FALSE hit —
            // the ray passes straight through (that interior is cut out by
            // the higher-priority medium) but the list still tracks its
            // enter/exit. Equal priorities never cut, so default scenes
            // behave exactly like plain interface tracking. Slots at or above
            // `count` hold air (lowest priority, IOR 1) so scans stay
            // poison-free and an empty list degenerates to air. Coincident
            // volumes must slightly overlap (entry surface of the inner
            // medium in front of the outer medium's exit surface) — exactly
            // coincident surfaces resolve to one hit like before.
            static constexpr uint kMediumListCap = kMaxGlassBounces + 1u;
            static constexpr float kMediumAirPriority = 1e9f;
            Local<float> medium_ior{kMediumListCap};
            Local<float> medium_priority{kMediumListCap};
            // Glassless specialization (has_glass==0): the init stores and all
            // medium-list users (the glass branches below) compile out, so DXC
            // drops the arrays — no per-thread init cost (perf review R2 item 8).
            if (has_glass != 0u) {
                for (uint i = 0u; i < kMediumListCap; ++i) {
                    medium_ior[i] = 1.0f;
                    medium_priority[i] = kMediumAirPriority;
                }
            }
            UInt medium_count = def(0u);

            // Best (lowest-value = highest-priority) entry, preferring the
            // most recent among ties; an all-air list returns an air slot.
            auto medium_best_index = [&]() noexcept {
                UInt best = def(0u);
                Float best_prio = def(kMediumAirPriority + 1.0f);
                for (uint i = 0u; i < kMediumListCap; ++i) {
                    Bool take = (medium_priority[i] < best_prio)
                              | ((medium_priority[i] == best_prio) & (UInt(i) > best));
                    best = ite(take, UInt(i), best);
                    best_prio = ite(take, medium_priority[i], best_prio);
                }
                return best;
            };
            // IOR of the medium the ray is currently in (air when empty).
            auto medium_current_ior = [&]() noexcept {
                return medium_ior[medium_best_index()];
            };
            // IOR the ray would enter after leaving the (p, eta) medium: the
            // best remaining entry, air if nothing else is tracked. Only the
            // most recent (p, eta) entry is excluded — the one remove() would
            // delete — so identical nested media (same priority and IOR) still
            // find each other, not air.
            auto medium_next_ior = [&](Float p, Float eta) noexcept {
                UInt excl = def(kMediumListCap);
                for (uint i = 0u; i < kMediumListCap; ++i) {
                    Bool hit = (UInt(i) < medium_count)
                             & (medium_priority[i] == p) & (medium_ior[i] == eta);
                    excl = ite(hit, UInt(i), excl);
                }
                UInt best = def(0u);
                Float best_prio = def(kMediumAirPriority + 1.0f);
                for (uint i = 0u; i < kMediumListCap; ++i) {
                    Bool take = (UInt(i) != excl)
                              & ((medium_priority[i] < best_prio)
                              | ((medium_priority[i] == best_prio) & (UInt(i) > best)));
                    best = ite(take, UInt(i), best);
                    best_prio = ite(take, medium_priority[i], best_prio);
                }
                return medium_ior[best];
            };
            // True hit: the surface's interior does not lose to the best medium.
            auto medium_is_true_hit = [&](Float p) noexcept {
                return p <= medium_priority[medium_best_index()];
            };
            auto medium_push = [&](Float p, Float eta) noexcept {
                $if(medium_count < kMediumListCap) {
                    medium_priority[medium_count] = p;
                    medium_ior[medium_count] = eta;
                    medium_count = medium_count + 1u;
                };
            };
            // Remove the most recent (p, eta) entry (compacting shift).
            auto medium_remove = [&](Float p, Float eta) noexcept {
                UInt match = def(kMediumListCap);
                for (uint i = 0u; i < kMediumListCap; ++i) {
                    Bool hit = (UInt(i) < medium_count)
                             & (medium_priority[i] == p) & (medium_ior[i] == eta);
                    match = ite(hit, UInt(i), match);
                }
                $if(match < medium_count) {
                    for (uint i = 0u; i + 1u < kMediumListCap; ++i) {
                        Bool move = (UInt(i) >= match) & (UInt(i) + 1u < medium_count);
                        medium_priority[i] = ite(move, medium_priority[i + 1u], medium_priority[i]);
                        medium_ior[i] = ite(move, medium_ior[i + 1u], medium_ior[i]);
                    }
                    medium_count = medium_count - 1u;
                    medium_priority[medium_count] = kMediumAirPriority;
                    medium_ior[medium_count] = 1.0f;
                };
            };

            $for(bounce, psr_bounce_budget) {
                // Iteration-0 hit fields: seeded from the classification trace
                // on glass pixels (item 7a — see the seed locals above), else
                // traced here. The names match the former direct reads so the
                // loop body below is unchanged.
                Float  hit_depth = def(0.0f);
                UInt   hit_inst  = def(0u);
                UInt   hit_prim  = def(0u);
                Float2 hit_bary  = def(make_float2(0.0f));
                Bool   hit_miss  = def(true);
#if NT_ENABLE_PROCEDURAL
                Bool   hit_is_procedural = def(false);
                UInt   hit_local_tri     = def(0u);
                Float2 hit_local_bary    = def(make_float2(0.0f));
#endif
                $if(glass_seeded & (bounce == 0u)) {
                    hit_depth = seed_depth;
                    hit_inst  = seed_inst;
                    hit_prim  = seed_prim;
                    hit_bary  = seed_bary;
                    hit_miss  = seed_miss;
#if NT_ENABLE_PROCEDURAL
                    hit_is_procedural = seed_is_procedural;
                    hit_local_tri     = seed_local_tri;
                    hit_local_bary    = seed_local_bary;
#endif
                } $else {
                    auto hit = render::trace_closest(accel, ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                    hit_depth = hit.committed_ray_t;
                    hit_inst  = hit.inst;
                    hit_prim  = hit.prim;
                    hit_bary  = hit.bary;
                    hit_miss  = hit->miss();
#if NT_ENABLE_PROCEDURAL
                    hit_is_procedural = hit.is_procedural;
                    hit_local_tri     = hit.local_tri;
                    hit_local_bary    = hit.local_bary;
#endif
                };

                $if(hit_miss) {
                    // Ray escaped — write miss data
                    inst_id = ~0u;
                    depth   = 3.402823466e38f;
                    psr_pending = false;
                    $break;
                };

#if NT_ENABLE_PROCEDURAL
                // Procedural hit: reconstruct normal via proc_bindless, check for glass PSR
                $if(hit_is_procedural) {
                    UInt   proc_inst_id = hit_prim;
                    UInt   local_tri = hit_local_tri;
                    Float2 hit_bary  = hit_local_bary;

                    Float3 hit_pos = ray->origin() + ray->direction() * hit_depth;
                    Float3 wo      = -ray->direction();

                    // Read procedural instance data and material
                    Var<scene::ProcInstanceData> proc_inst =
                        proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(proc_inst_id);
                    auto material = scene.material_buffer.read(Expr{ proc_inst.material_layers & 0xFFu });

                    // Reconstruct normal via procedural helpers
                    Float3 geo_ns = reconstruct_procedural_normal(
                        proc_bindless, proc_inst_id,
                        (1u << 29u) | local_tri, hit_pos, hit_bary);
                    Float3 ns = ite(dot(wo, geo_ns) < 0.0f, -geo_ns, geo_ns);

                    // Dielectric PSR: trace through procedural glass. Interface
                    // IORs come from the nested-dielectric medium list; the
                    // priority (material.metallic) resolves overlapping media.
                    // C++-gated on the baked has_glass specialization.
                    UInt mat_bsdf = get_effective_bsdf_type(material);
                    if (has_glass != 0u)
                    $if(mat_bsdf == 3u | mat_bsdf == 11u) {

                        // Resolve surface so custom callables can drive ior/attenuation/
                        // attenuation_distance/albedo for the PSR computation. Mirrors
                        // the mesh path. Branch decision still uses
                        // get_effective_bsdf_type(material) per the SurfaceData invariant.
                        Float2 screen_uv_gbuf = (make_float2(coord) + 0.5f) / make_float2(resolution);
                        SurfaceData glass_surface = resolve_procedural_surface_textured(
                            resolver, proc_bindless, tex_bindless,
                            proc_inst_id, (1u << 29u) | local_tri, hit_pos, wo,
                            scene.material_buffer, hit_bary,
                            0.0f, screen_uv_gbuf, resolution.x, resolution.y);

                        // Callable-driven glass blend roll — see the mesh branch
                        // (same static-IGN contract; rolled-opaque stores the
                        // procedural hit with the glass bit clear).
                        Float blend_roll = ign_static(coord);
                        Bool rolled_opaque = (glass_surface.glass_blend < 1.0f)
                                           & !(blend_roll < glass_surface.glass_blend);
                        $if(rolled_opaque) {
                            inst_id = proc_inst_id;
                            prim_id = (1u << 29u) | local_tri;
                            depth   = hit_depth;
                            bary    = hit_bary;
                            psr_pending = false;
                            $break;
                        };
                        glass_bounces = glass_bounces + 1u;

                        Float mat_ior  = glass_surface.ior;
                        Float mat_prio = material.metallic;
                        Bool  is_thin  = mat_bsdf == 11u;
                        Float cos_wo_geo = dot(wo, geo_ns);
                        Bool  entering   = cos_wo_geo > 0.0f;
                        Float3 refract_normal = ite(entering, geo_ns, -geo_ns);
                        Float  cos_i = abs(dot(wo, refract_normal));

                        // The PSR chain is d-line (dispersion-blind): a per-frame
                        // stochastic channel pick here rotated the refraction
                        // direction, TIR topology, Fresnel and medium list every
                        // frame, breaking the stable-G-buffer contract ReLAX
                        // history and ReSTIR reuse depend on (speckle at any
                        // roughness — docs/dispersion_speckle_fix_plan.md RC1).
                        // Dispersion lives only in the rough-glass gather taps;
                        // smooth-glass fringes return with Phase 3's replay.
                        Float mat_ior_eff = mat_ior;

                        // False hit: this interior is cut out by a higher-priority
                        // medium on the list — pass straight through, no Fresnel,
                        // no absorption, but keep tracking the enter/exit.
                        Bool true_hit = ite(is_thin, true, medium_is_true_hit(mat_prio));
                        $if(!true_hit) {
                            $if(entering) {
                                medium_push(mat_prio, mat_ior_eff);
                            } $else {
                                medium_remove(mat_prio, mat_ior_eff);
                            };
                            Float fh_offset = max(0.001f * hit_depth, 1e-4f);
                            ray = make_ray(hit_pos + ray->direction() * fh_offset,
                                           ray->direction(), 0.0f, 1e10f);
                            psr_pending = true;
                            $continue;
                        };

                        // Interface IOR pair from the medium list. Thin walls pair
                        // against the surrounding medium on both sides; volumes
                        // enter the surface's interior or exit to what remains.
                        Float eta_i = ite(entering, medium_current_ior(), mat_ior_eff);
                        Float eta_t = ite(is_thin,
                            ite(entering, mat_ior_eff, medium_current_ior()),
                            ite(entering, mat_ior_eff, medium_next_ior(mat_prio, mat_ior_eff)));
                        Float F = fresnel_dielectric(cos_i, eta_i, eta_t);

#if NT_ENABLE_SHARC
                        // First visible interface only. D-line ratio + Abbe in .w
                        // (see the mesh path).
                        $if(glass_bounces == 1u) {
                            first_glass_rough = glass_surface.roughness;
#if NT_ENABLE_DISPERSION
                            // The chain is d-line, so eta_i/eta_t IS the d-line
                            // ratio the gather re-disperses per tap.
                            first_glass_eta = eta_i / eta_t;
                            first_glass_disp = ite(
                                (glass_surface.dispersion > 0.0f) & !is_thin,
                                glass_surface.dispersion, 0.0f);
#else
                            first_glass_eta   = eta_i / eta_t;
#endif
                            first_glass_thin  = ite(is_thin, 1.0f, 0.0f);
                        };
#endif

                        Float3 absorption = make_float3(1.0f);
                        $if(!is_thin) {
                            $if(glass_surface.attenuation_distance > 0.0f) {
                                Float ratio = hit_depth / glass_surface.attenuation_distance;
                                absorption = make_float3(
                                    pow(glass_surface.attenuation.x, ratio),
                                    pow(glass_surface.attenuation.y, ratio),
                                    pow(glass_surface.attenuation.z, ratio));
                            };
                        } $else {
                            absorption = glass_surface.albedo;
                        };

                        Float3 wi = ite(is_thin, -wo,
                            refract_dir(wo, refract_normal, eta_i / eta_t));
                        Bool is_tir = ite(is_thin, false, dot(wi, refract_normal) > 0.0f);

                        fresnel_accum = ite(is_tir, fresnel_accum,
                            fresnel_accum + (1.0f - fresnel_accum) * F);
                        throughput = throughput * ite(is_tir, absorption, (1.0f - F) * absorption);

                        // The list follows transmitted crossings only (TIR is a
                        // reflection event; a thin wall bounds no volume).
                        $if(!is_thin & !is_tir) {
                            $if(entering) {
                                medium_push(mat_prio, mat_ior_eff);
                            } $else {
                                medium_remove(mat_prio, mat_ior_eff);
                            };
                        };

                        Float3 offset_dir = ite(is_tir, refract_normal,
                            ite(is_thin, -refract_normal, -refract_normal));
                        ray = make_ray(hit_pos + offset_dir * max(0.001f * hit_depth, 1e-4f), wi, 0.0f, 1e10f);
                        psr_pending = true;
                        $continue;
                    };

                    // Opaque procedural hit — store and break
                    inst_id = proc_inst_id;
                    prim_id = (1u << 29u) | local_tri;
                    depth   = hit_depth;
                    bary    = hit_bary;
                    psr_pending = false;
                    $break;
                };
#endif

                // Read hit data (hoisted hit_* locals above — seeded or traced)
                Float3 hit_pos  = ray->origin() + ray->direction() * hit_depth;
                Float3 wo       = -ray->direction();

                // Read instance data, material, and instance transform once per hit.
                // The transform is reused by the geo_ns transform below and by the
                // alpha-cutout / glass-PSR resolve_surface calls further down (XIR
                // has no LICM/CSE, so without this hoist each use is a separate
                // buffer fetch — 3 fetches for alphacut-or-glass pixels).
                UInt4   inst_data = scene.instance_buffer.read(hit_inst);
                auto    material  = scene.material_buffer.read(Expr{ inst_data.y & 0xFFu });
                Float4x4 hit_xform = scene.instance_transform_buffer.read(hit_inst);

                // --- Camera-invisible geometry (light shapes that illuminate
                // without rendering): pass through like alpha cutout. Light
                // sampling power is untouched, and shadow/GI rays still hit
                // the mesh (type-5 emissives never block shadow rays), so
                // illumination is identical to the visible case.
                $if((inst_data.x & kCameraInvisibleFlag) != 0u) {
                    Float offset = max(0.001f * hit_depth, 1e-4f);
                    ray = make_ray(hit_pos + ray->direction() * offset, ray->direction(), 0.0f, 1e10f);
                    psr_pending = true;
                    $continue;
                };

                // Reconstruct geometry normal via bindless vertex access + transform to world space.
                // For glass pixels the ray is already unjit (set before the loop), so hit_bary
                // and downstream normal/refraction/absorption/bg_pos are stable automatically.
                Bool is_double_sided_gbuf = (inst_data.x & kDoubleSidedFlag) != 0u;
                // Single fused fetch (1 triangle + 3 vertex reads) shared by the
                // normal, UV, and object-position reconstructions below — XIR has
                // no CSE, so separate reconstruct_* calls re-fetch per attribute.
                MeshTriVerts hit_tv = read_mesh_triangle(
                    vertex_bindless, inst_data.z, inst_data.w, hit_prim);
                Float3 obj_geo_ns = ite(is_double_sided_gbuf,
                    reconstruct_face_normal(hit_tv),
                    reconstruct_normal(hit_tv, hit_bary));
                Float3 geo_ns   = transform_normal(hit_xform, obj_geo_ns);
                Float3 ns       = ite(dot(wo, geo_ns) < 0.0f, -geo_ns, geo_ns);

                // --- Alpha cutout (deterministic threshold test) ---
                // Run resolve_surface() so custom resolvers can modify albedo_alpha.
                // Without this, procedural materials (e.g. checkerboard) can't drive cutout.
                $if(material.alphacut > 0.0f) {
                    Float2 hit_uv_gbuf = reconstruct_uv(hit_tv, hit_bary);
                    Float2 screen_uv_gbuf = (make_float2(coord) + 0.5f) / make_float2(resolution);
                    SurfaceData alpha_surface = resolve_surface(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data.z, inst_data.w, hit_prim, hit_bary,
                        material, wo, hit_xform,
                        0.0f, screen_uv_gbuf, resolution.x, resolution.y,
                        0u, hit_inst);
                    $if(alpha_surface.albedo_alpha < alpha_surface.alphacut) {
                        Float offset = max(0.001f * hit_depth, 1e-4f);
                        ray = make_ray(hit_pos + ray->direction() * offset, ray->direction(), 0.0f, 1e10f);
                        psr_pending = true;
                        $continue;
                    };
                };

                // --- Dielectric PSR: trace through glass surfaces ---
                // Interface IORs come from the nested-dielectric medium list
                // (Schmidt & Budge priorities; see the list setup above the
                // loop), so glass-in-glass/liquid evaluates each interface
                // against the actual surrounding media instead of air.
                // C++-gated on the baked has_glass specialization — glassless
                // scenes can never take this branch (has_visible_glass()==false
                // ⇒ no visible type-3/11 material), so it compiles out.
                UInt mat_bsdf = get_effective_bsdf_type(material);
                if (has_glass != 0u)
                $if(mat_bsdf == 3u | mat_bsdf == 11u) {

                    // Resolve surface so custom callables can drive ior/attenuation/
                    // attenuation_distance/albedo for the PSR computation. Mirrors
                    // the alpha-cutout path above. Branch decision still uses
                    // get_effective_bsdf_type(material) per the SurfaceData invariant.
                    Float2 screen_uv_gbuf = (make_float2(coord) + 0.5f) / make_float2(resolution);
                    SurfaceData glass_surface = resolve_surface(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data.z, inst_data.w, hit_prim, hit_bary,
                        material, wo, hit_xform,
                        0.0f, screen_uv_gbuf, resolution.x, resolution.y,
                        0u, hit_inst);

                    // --- Callable-driven glass blend roll (docs/glass_blend_plan.md) ---
                    // glass_blend < 1 stochastically selects glass vs opaque per
                    // pixel with a STATIC IGN dither (no frame term — RC1: the
                    // roll outcome must be stable across frames for ReLAX
                    // history/ReSTIR reuse). Both outcomes share the unjittered
                    // classification geometry. Rolled-opaque stores THIS surface
                    // with the glass bit clear (fresnel_accum stays 0, throughput
                    // identity) and no glass_bounces/SHARC side effects; consumers
                    // reclass it via reclass_blend_rolled_opaque. Energy:
                    // E[pixel] = blend·glass + (1−blend)·diffuse with UNSCALED
                    // contributions — never scale throughput by blend here.
                    Float blend_roll = ign_static(coord);
                    Bool rolled_opaque = (glass_surface.glass_blend < 1.0f)
                                       & !(blend_roll < glass_surface.glass_blend);
                    $if(rolled_opaque) {
                        inst_id = hit_inst;
                        prim_id = hit_prim;
                        depth   = hit_depth;
                        bary    = hit_bary;
                        obj_pos = reconstruct_object_position(hit_tv, hit_bary);
                        psr_pending = false;
                        $break;
                    };
                    glass_bounces    = glass_bounces + 1u;

                    Float mat_ior  = glass_surface.ior;
                    // Interior priority (overloaded metallic slot, dead for type 3;
                    // read raw so RMA textures cannot reach it).
                    Float mat_prio = material.metallic;
                    // Use geometry normal (not flipped ns) to determine entering vs leaving
                    Float cos_wo_geo = dot(wo, geo_ns);
                    Bool  entering   = cos_wo_geo > 0.0f;

                    // Normal facing toward the incoming ray for Fresnel/refraction
                    Float3 refract_normal = ite(entering, geo_ns, -geo_ns);
                    Float  cos_i          = abs(dot(wo, refract_normal));

                    Bool is_thin = mat_bsdf == 11u;

                    // The PSR chain is d-line (dispersion-blind): a per-frame
                    // stochastic channel pick here rotated the refraction
                    // direction, TIR topology, interface Fresnel and the medium
                    // list every frame (seed rewritten per frame), so the stored
                    // background surface — motion vectors, virtual depth, the
                    // surface the denoiser shades — plus glass_throughput and
                    // fresnel_accum all flipped between three realizations.
                    // ReLAX history is guided by exactly that motion/depth:
                    // flip regions reset to 1-spp salt-and-pepper (speckle even
                    // at roughness 0; docs/dispersion_speckle_fix_plan.md RC1).
                    // Dispersion moved to the rough-glass gather taps (exact
                    // per-channel estimator); smooth-glass fringes return with
                    // Phase 3's post-denoise replay.
                    Float mat_ior_eff = mat_ior;

                    // False hit: this interior is cut out by a higher-priority
                    // medium on the list — pass straight through, no Fresnel,
                    // no absorption, but keep tracking the enter/exit.
                    Bool true_hit = ite(is_thin, true, medium_is_true_hit(mat_prio));
                    $if(!true_hit) {
                        $if(entering) {
                            medium_push(mat_prio, mat_ior_eff);
                        } $else {
                            medium_remove(mat_prio, mat_ior_eff);
                        };
                        Float fh_offset = max(0.001f * hit_depth, 1e-4f);
                        ray = make_ray(hit_pos + ray->direction() * fh_offset,
                                       ray->direction(), 0.0f, 1e10f);
                        psr_pending = true;
                        $continue;
                    };

                    // Interface IOR pair from the medium list. Thin walls pair
                    // against the surrounding medium on both sides; volumes
                    // enter the surface's interior or exit to what remains.
                    Float eta_i = ite(entering, medium_current_ior(), mat_ior_eff);
                    Float eta_t = ite(is_thin,
                        ite(entering, mat_ior_eff, medium_current_ior()),
                        ite(entering, mat_ior_eff, medium_next_ior(mat_prio, mat_ior_eff)));
                    Float F = fresnel_dielectric(cos_i, eta_i, eta_t);

#if NT_ENABLE_SHARC
                    // First visible interface only (glass_bounces was just
                    // incremented for this crossing).
                    $if(glass_bounces == 1u) {
                        first_glass_rough = glass_surface.roughness;
                        // D-LINE ratio: the chain is d-line, so eta_i/eta_t is
                        // the d-line pair the gather taps re-disperse per tap
                        // from this + the Abbe value in .w. First interface
                        // always pairs against air (the medium list is
                        // untouched at glass_bounces == 1).
#if NT_ENABLE_DISPERSION
                        first_glass_eta = eta_i / eta_t;
                        first_glass_disp = ite(
                            (glass_surface.dispersion > 0.0f) & !is_thin,
                            glass_surface.dispersion, 0.0f); // thin → off
#else
                        first_glass_eta   = eta_i / eta_t;
#endif
                        first_glass_thin  = ite(is_thin, 1.0f, 0.0f);
                    };
#endif

                    // Absorption through glass (Beer-Lambert law)
                    // Thin dielectric: use attenuation as simple tint (thin wall, no distance)
                    Float3 absorption = make_float3(1.0f);
                    $if(!is_thin) {
                        $if(glass_surface.attenuation_distance > 0.0f) {
                            Float3 att   = glass_surface.attenuation;
                            Float  ratio = hit_depth / glass_surface.attenuation_distance;
                            absorption = make_float3(
                                pow(att.x, ratio),
                                pow(att.y, ratio),
                                pow(att.z, ratio));
                        };
                    }
                    $else {
                        // Thin wall: albedo IS the transmission tint
                        absorption = glass_surface.albedo;
                    };

                    // Compute refracted/reflected direction
                    // Thin dielectric: straight through (no refraction), no TIR possible
                    Float3 wi   = ite(is_thin, -wo,
                        refract_dir(wo, refract_normal, eta_i / eta_t));
                    Bool is_tir = ite(is_thin, false, dot(wi, refract_normal) > 0.0f);

                    // Fresnel accumulation: only for valid refraction interfaces
                    // (TIR redirects the ray but doesn't remove energy from the transmitted path)
                    fresnel_accum = ite(is_tir, fresnel_accum,
                        fresnel_accum + (1.0f - fresnel_accum) * F);

                    // Throughput: attenuate for valid refraction; TIR only applies
                    // absorption. D-line quantities throughout (stable across
                    // frames — no temporal accumulator exists for this image).
                    throughput = throughput * ite(is_tir, absorption, (1.0f - F) * absorption);

                    // The medium list follows transmitted crossings only (TIR is a
                    // reflection event; a thin wall bounds no volume).
                    $if(!is_thin & !is_tir) {
                        $if(entering) {
                            medium_push(mat_prio, mat_ior_eff);
                        } $else {
                            medium_remove(mat_prio, mat_ior_eff);
                        };
                    };

                    // Offset: move to the side where wi points (avoids self-intersection)
                    Float3 offset_dir = ite(is_tir, refract_normal,
                        ite(is_thin, -refract_normal, -refract_normal));
                    ray               = make_ray(hit_pos + offset_dir * max(0.001f * hit_depth, 1e-4f), wi, 0.0f, 1e10f);
                    psr_pending = true;
                    $continue;
                };

                // --- Opaque hit — store and break ---
                inst_id = hit_inst;
                prim_id = hit_prim;
                depth   = hit_depth;
                bary    = hit_bary;
                obj_pos = reconstruct_object_position(hit_tv, hit_bary);
                psr_pending = false;
                $break;
            };

            // Bounce budget exhausted mid-trace (deep glass nesting, TIR chains):
            // the loop ended while the ray was still in flight, leaving the gbuf
            // in its "no hit" state — which the composite blit renders as sky
            // (envmap at the original camera direction, black at 0 intensity).
            // Instead, trace once more straight through — no further refraction,
            // Fresnel or list updates — and store whatever it hits as the
            // background surface. Over-budget stacks then show the surface
            // behind with the accumulated tint instead of the sky artifact.
            $if(psr_pending) {
                auto last_hit = render::trace_closest(accel, ray
                #if NT_ENABLE_PROCEDURAL
                    , proc_bindless
                #endif
                );
                $if(!last_hit->miss()) {
                #if NT_ENABLE_PROCEDURAL
                    $if(last_hit.is_procedural) {
                        inst_id = last_hit.prim;
                        prim_id = (1u << 29u) | last_hit.local_tri;
                        depth   = last_hit.committed_ray_t;
                        bary    = last_hit.local_bary;
                    } $else {
                #endif
                        inst_id = last_hit.inst;
                        prim_id = last_hit.prim;
                        depth   = last_hit.committed_ray_t;
                        bary    = last_hit.bary;
                        UInt4 last_inst_data = scene.instance_buffer.read(last_hit.inst);
                        MeshTriVerts last_tv = read_mesh_triangle(
                            vertex_bindless, last_inst_data.z, last_inst_data.w, last_hit.prim);
                        obj_pos = reconstruct_object_position(last_tv, last_hit.bary);
                #if NT_ENABLE_PROCEDURAL
                    };
                #endif
                };
            };

            // Compute motion vectors for the final (virtual) hit position.
            // NRD 4.17.4 convention: MVs stay UNJITTERED — plain
            // project_prev − project through the unjittered matrices (the
            // former jitter − prev_jitter delta is removed: RTXDI's +=
            // pixelOffset exists to cancel jitter-baked matrices, which ours
            // are not; adding it produced jittered MVs, the opposite of what
            // NRD — and FSR/DLSS — expect. Behavior-identical while
            // primaryJitterEnabled=false since the delta was 0).
            Float2 velocity_px = make_float2(0.0f);
            Float  depth_ndc   = 1.0f;  // sky / no-hit: far plane
            $if(inst_id != ~0u) {
                // Actual world position of the background surface (may be off-camera-ray due to refraction).
                // For glass pixels the ray is unjit, so bg_pos is stable across frames.
                Float3 bg_pos = ray->origin() + ray->direction() * depth;
                // Object motion: reproject the object-space hit through the
                // previous instance transform so moving geometry produces real
                // motion even with a static camera. Bit 29 of prim_id flags
                // procedural hits — they have no transform-buffer entry and
                // keep camera-only motion. The $if guard (not ite) avoids an
                // out-of-range buffer read for procedural instance ids.
                $if((prim_id >> 29u) == 0u) {
                    Float4x4 prev_xform = scene.instance_transform_prev_buffer.read(inst_id);
                    Float3 prev_world = (prev_xform * make_float4(obj_pos, 1.0f)).xyz();
                    motion = camera->project_prev(prev_world) - camera->project(bg_pos);
                } $else {
                    motion = camera->project_prev(bg_pos) - camera->project(bg_pos);
                };
                // Virtual depth: project bg_pos onto original camera ray so that
                // shade pass reconstructs: cam_origin + cam_dir * depth ≈ bg_pos depth
                depth         = dot(bg_pos - cam_origin, cam_dir) / dot(cam_dir, cam_dir);

                // Upscaler inputs from the same virtual hit: velocity in
                // render-resolution pixels (NDC delta * 0.5 * size, so
                // prev_pixel = curr_pixel + mv), depth as NDC z/w of the
                // virtual surface (non-inverted, near/far = camera clips).
                Float2 render_size = make_float2(gbuf_velocity.size());
                velocity_px = motion * 0.5f * render_size;
                Float4 clip_pos = camera->view_proj * make_float4(bg_pos, 1.0f);
                depth_ndc   = clip_pos.z / clip_pos.w;
            };

            // Pack is_glass into prim_id MSB (vis is INT2 — only .x and .y are stored)
            UInt is_glass    = ite(fresnel_accum > 0.0f, 1u, 0u);
            gbuf_depth.write      (coord, make_float4(depth));
            gbuf_vis.write        (coord, make_uint4 (inst_id, Expr{ (is_glass << 31u) | prim_id }, glass_bounces, 0u));
            gbuf_bary_motion.write(coord, make_float4(bary, motion));
            glass_throughput.write(coord, make_float4(throughput, fresnel_accum));
            gbuf_velocity.write     (coord, make_float4(velocity_px, 0.0f, 0.0f));
            gbuf_depth_upscale.write(coord, make_float4(depth_ndc));
#if NT_ENABLE_SHARC
            // Rough-glass classification side image. Written unconditionally
            // (zeros for opaque / smooth pixels) so stale values from a previous
            // frame can never survive a classification change.
            rough_glass_info.write(coord, make_float4(first_glass_rough,
                                                      first_glass_eta,
                                                      first_glass_thin,
#if NT_ENABLE_DISPERSION
                                                      first_glass_disp
#else
                                                      0.0f
#endif
                                                      ));
#endif
        });

        if (!resolverOnly) {
        //==========================================================================
        // Presample Local Lights kernel (fill the fixed pool with alias table
        // samples). Dispatch shape: x = entry-in-tile, y = tile id (same shape
        // as the ref's presample passes). Entry RNG unchanged: xxhash32 over
        // the global entry index + frame.
        //==========================================================================
        _presampleLocalShader = device.compile<2>([&](
            BufferVar<PresampledCandidate> pool_buffer,
            UInt frame_count,
            Var<LightSamplingResources> lights
            ) noexcept {
            set_block_size(64u, 4u, 1u);
            set_name("DI_PresampleLocal");
            UInt entry_id = dispatch_id().x;   // 0 .. kPresampleTileSize-1
            UInt tile_id  = dispatch_id().y;   // 0 .. kPresamplePoolTileCount-1
            UInt global_entry = tile_id * kPresampleTileSize + entry_id;

            // Per-entry seed from xxhash32 + pcg
            UInt s = util::xxhash32(make_uint2(global_entry, frame_count));
            s = util::pcg(s);

            Float u_select = util::uniform_uint_to_float(s);
            s = util::lcg_ui(s);
            Float u_tri_x = util::uniform_uint_to_float(s);
            s = util::lcg_ui(s);
            Float u_tri_y = util::uniform_uint_to_float(s);
            s = util::lcg_ui(s);

            Float u_scaled  = u_select * cast<float>(lights.emissive_count);
            UInt idx        = cast<uint>(u_scaled);
            idx             = min(idx, lights.emissive_count - 1u);

            auto entry      = lights.alias_table->read(idx);
            UInt light_idx  = ite(u_scaled - cast<float>(idx) < entry.pdf,
                                entry.triangle_index, entry.alias_index);
            light_idx       = min(light_idx, lights.emissive_count - 1u);

            Float su        = sqrt(u_tri_x);

            auto tri_light = lights.triangle_lights->read(light_idx);
            // Precompute the bary-interpolated emitter point into the entry
            // (perf review R2 item 3): moves the per-pixel × per-candidate
            // triangle_vertices read out of the candidate loop into this
            // once-per-dirty-frame, coherent pass. Formula order replicates
            // the former consumer exactly (bit-identical).
            auto tri_verts = lights.triangle_vertices->read(light_idx);

            Float source_pdf;
            if constexpr (kUniformLightSampling) {
                source_pdf = lights.emissive_count_inv / tri_light.area;
            } else {
                source_pdf = tri_light.pdf / tri_light.area;
            }

            Var<PresampledCandidate> pc;
            pc.light_idx        = light_idx;
            pc.bary_u           = 1.0f - su;
            pc.bary_v           = u_tri_y * su;
            pc.inv_source_pdf   = 1.0f / luisa::compute::max(source_pdf, 1e-10f);
            pc.light_point      = pc.bary_u * tri_verts.v0 + pc.bary_v * tri_verts.v1
                               + (1.0f - pc.bary_u - pc.bary_v) * tri_verts.v2;

            pool_buffer->write(global_entry, pc);
        });
        } // end resolverOnly gate (PresampleLocal)

        if (!resolverOnly) {
        //==========================================================================
        // Presample Env Lights kernel (fill the fixed pool with envmap CDF
        // samples). Same dispatch shape as the local pool — the former 8×8
        // block tiling and screen-tile strides are gone (the producer/consumer
        // tiling mismatch is dissolved by construction: consumers pick windows
        // by hash, not by screen position).
        //==========================================================================
        _presampleEnvShader = device.compile<2>([&](
            BufferVar<PresampledCandidate> pool_buffer,
            UInt frame_count,
            Var<EnvLightResources> env
            ) noexcept {
            set_block_size(64u, 4u, 1u);
            set_name("DI_PresampleEnv");
            UInt entry_id = dispatch_id().x;   // 0 .. kPresampleEnvTileSize-1
            UInt tile_id  = dispatch_id().y;   // 0 .. kPresamplePoolTileCount-1
            UInt global_entry = tile_id * kPresampleEnvTileSize + entry_id;

            UInt s = util::xxhash32(make_uint2(global_entry, frame_count));
            s = util::pcg(s);

            Float u_env_x = util::uniform_uint_to_float(s);
            s = util::lcg_ui(s);
            Float u_env_y = util::uniform_uint_to_float(s);

            auto env_sample = sample_envmap_cdf(
                make_float2(u_env_x, u_env_y),
                env.env_marginal_cdf, env.env_conditional_cdf,
                env.env_width, env.env_height, env.env_integral, env.env_rotation);
            auto source_pdf = std::get<3>(env_sample);
            source_pdf      = luisa::compute::max(source_pdf, 1e-10f);

            Var<PresampledCandidate> pc;
            pc.light_idx    = kEnvLightSentinel;
            pc.bary_u       = std::get<1>(env_sample);
            pc.bary_v       = std::get<2>(env_sample);
            pc.inv_source_pdf = 1.0f / source_pdf;
            pc.light_point  = make_float3(0.0f);  // unused for env entries

            pool_buffer->write(global_entry, pc);
        });
        } // end resolverOnly gate (PresampleEnv)

        //==========================================================================
        // ReSTIR Candidate Generation kernel
        //==========================================================================
        _candidateShader = device.compile<2>([&](
            BufferVar<DIParams> params,
            BufferVar<Reservoir> reservoir_buffer,
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
            Float env_exposure_val,
            UInt env_candidate_count,
            // Presampled light tiles
            BufferVar<PresampledCandidate> presample_local_tiles,
            BufferVar<PresampledCandidate> presample_env_tiles,
            UInt frame_count,
            UInt cbField,
            // Glass throughput for p_hat attenuation
            ImageFloat glass_throughput
#if NT_ENABLE_PROCEDURAL
	            , BindlessVar proc_bindless
#endif
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("DI_Candidate");
            auto p = params.read(0u);

            // Compile-time-specialized flag (former UInt arg; baked by
            // compileImpl): 0 = opaque scene, env visibility filter can use
            // any-hit directly. C++ const so DXC folds the dead branch away.
            const uint hasTransparentShadowCasters = _bakedTransparentShadowCasters;
            UInt2 rsv       = dispatch_id().xy();
            UInt2 rsv_res   = dispatch_size().xy();
            UInt2 coord;
            UInt2 resolution;
            UInt  pixel_index;
            if (_checkerboard) {
                // Half-res dispatch: convert reservoir position to full-res pixel position
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

            Var<Reservoir> r;
            r.light_idx     = def(~0u);
            r.w_sum         = def(0.0f);
            r.target_pdf    = def(0.0f);
            r.light_bary_u  = def(0.0f);
            r.light_bary_v  = def(0.0f);
            r.packed_meta   = def(0u);

            UInt4  vis      = gbuf_vis.read(coord);
            UInt   inst_id  = vis.x;
            UInt   prim_id  = vis.y & 0x3FFFFFFFu;
            Bool   is_glass = (vis.y >> 31u) > 0u;
            
            Bool hit_geometry   = inst_id != ~0u;
            Bool is_point       = ((vis.y >> 30u) & 1u) > 0u;
            Bool is_procedural  = ((vis.y >> 29u) & 1u) > 0u;

            $if(hit_geometry & !is_point) {
                Bool has_lights = lights.emissive_count > 0u;
                Float  depth = gbuf_depth.read(coord).x;
                Float2 bary = gbuf_bary_motion.read(coord).xy();
                
                auto   ray = camera->generate_ray(Expr{
                    (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f
                });
                Float3 wo = -ray->direction();

                Float3 world_pos = ray->origin() + ray->direction() * depth;

                // Resolve surface - branch for procedural vs mesh
                SurfaceData surface;
#if NT_ENABLE_PROCEDURAL
                $if(is_procedural) {
                    // Recompute bary at unjittered pixel center for stable texture sampling.
                    auto cand_ray_unjit = camera->generate_ray(Expr{
                        (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter
                    });
                    $if(!is_glass) {
                        bary = reconstruct_unjittered_bary_procedural(
                            proc_bindless, inst_id, prim_id, bary,
                            cand_ray_unjit->origin(), cand_ray_unjit->direction());
                    };
                    surface = resolve_procedural_surface_textured(
                        resolver, proc_bindless, tex_bindless,
                        inst_id, prim_id, world_pos, wo,
                        scene.material_buffer,
                        bary,
                        0.0f,
                        Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
                        resolution.x, resolution.y);
                }
                $else {
#endif
                    UInt4 inst_data_local = scene.instance_buffer.read(inst_id);
                    surface = resolve_surface_from_instance(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data_local, prim_id, bary,
                        scene.material_buffer, wo,
                        scene.instance_transform_buffer.read(inst_id),
                        0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
                        resolution.x, resolution.y, inst_id);
#if NT_ENABLE_PROCEDURAL
                };
#endif

                // Blend-rolled-opaque reclass (see Shading.h) — candidate p_hat
                // must evaluate the diffuse side, not a delta dielectric.
                reclass_blend_rolled_opaque(surface, is_glass);

                Float3 ns       = surface.ns;
                Float3 geo_ns   = surface.geo_ns;
                Float3 facing_ns = ite(dot(wo, geo_ns) < 0.f, -geo_ns, geo_ns);
                Float3 tangent  = surface.tangent;
                Float  tangent_w= surface.tangent_w;

                // BSDF for MIS evaluation (lightweight — just stores material params).
                // Probe-covered SSS pixels zero the HK lobe: direct model is the
                // additive Burley probe in the shade shader (no double count).
                // Thin-wall subsurface (diffuse_trans > 0) always keeps the
                // lobes — the probe is suppressed for it. Point pixels never
                // reach here (gated above).
                Bool sss_probe_direct = Expr{ surface.bsdf_type == 6u }
                                      & Expr{ surface.flatness > 0.0f }
                                      & Expr{ surface.diffuse_trans <= 0.0f };
                MaterialBSDF bsdf_mis = surface.make_bsdf(sss_probe_direct);

                // Glass attenuation for p_hat: ReSTIR should importance-sample the attenuated
                // contribution, not the raw background BRDF value.
                Float4 glass_tp = glass_throughput.read(coord);
                Float glass_att = luminance(glass_tp.xyz());

                UInt  seed       = seed_image.read(coord).x;
                Float w_sum_acc  = def(0.0f);

                // Phase 2 randomized windows: each 16×16 screen block picks its
                // pool window by hashing its tile coords + frame index —
                // block-COHERENT (one window serves the wavefront as an L1
                // broadcast; do NOT hash per pixel), per-frame VARYING (a
                // window with zero usable entries for this surface
                // orientation is re-rolled next frame instead of freezing the
                // tile black), and independently salted for local/env. This
                // replaces the frozen screen-tile-indexed windows that caused
                // the stuck 16×16 block artifact.
                UInt2 tile_xy       = coord / kPresampleBlockSize;
                UInt win_local      = util::xxhash32(
                    make_uint3(tile_xy.x, tile_xy.y, frame_count))
                    & (kPresamplePoolTileCount - 1u);
                UInt win_env        = util::xxhash32(
                    make_uint3(tile_xy.x, tile_xy.y, frame_count ^ 0x9E3779B9u))
                    & (kPresamplePoolTileCount - 1u);
                UInt local_tile_base = win_local * kPresampleTileSize;
                UInt env_tile_base   = win_env * kPresampleEnvTileSize;

                // Tile-coherent entry RNG (perf review R2 item 3; RTXDI
                // "coherentRng" pattern, Doc/ShaderAPI.md): every pixel of a
                // 16×16 block derives the per-candidate presample entry pick
                // from ONE seed-image texel (the block's top-left) instead of
                // its own — the whole wavefront then reads the same tile entry
                // per candidate (one cache line serves the group) and the
                // remaining per-pixel light-buffer reads become coherent
                // broadcasts. Candidate SETS correlate within a block; RIS
                // stays unbiased (RTXDI ships this as its default). The
                // accept/reject RNG below still uses the per-pixel seed.
                UInt tile_seed = seed;
                if constexpr (kTileCoherentCandidateRng) {
                    UInt2 tl = make_uint2(coord.x / kPresampleBlockSize * kPresampleBlockSize,
                                          coord.y / kPresampleBlockSize * kPresampleBlockSize);
                    tile_seed = seed_image.read(tl).x;
                };

                // Loop: local light candidates -> env candidates -> BRDF candidates.
                // Runtime counts (perf review R2 item 4): M_total is no longer
                // a compile-time constant, so DXC cannot unroll the loop (the
                // candidate kernel was a fully unrolled 24-iteration body) and
                // the count is A/B-tunable without recompiles (RTXDI Medium
                // preset: 8 local + 1 infinite + 1 env vs our default 15+8+1).
                // The single BRDF candidate stays compile-time (the $else arm).
                const UInt M_light  = p.localLightCandidateCount;
                const UInt M_env    = env_candidate_count;
                const UInt M_total  = M_light + M_env + kBrdfCandidateCount;

                $for(i, M_total) {
                    $if(i < M_light) {
                        // ======== Local light candidate (from presampled tile) ========
                        $if(has_lights) {
                            //UInt read_seed = util::xxhash32(make_uint2(cast<UInt>(i), seed));
                            UInt entry_idx  = util::xxhash32(make_uint2(cast<UInt>(i), tile_seed)) & (kPresampleTileSize - 1u);
                            auto pc         = presample_local_tiles.read(local_tile_base + entry_idx);

                            UInt light_idx      = pc.light_idx;
                            Float bary_u        = pc.bary_u;
                            Float bary_v        = pc.bary_v;
                            //Float inv_source_pdf= pc.inv_source_pdf;

                            auto tri_light      = lights.triangle_lights->read(light_idx);
                            // Emitter point precomputed into the tile entry at
                            // presample time (perf review R2 item 3) — skips the
                            // per-pixel triangle_vertices read per candidate.
                            Float3 light_point  = pc.light_point;

                            // Lighting geometry — computed once and shared between
                            // p_hat evaluation and BRDF-PDF MIS blending below.
                            Float3 to_light     = light_point - world_pos;
                            Float dist_sq       = max(dot(to_light, to_light), 1e-6f);
                            Float3 light_dir    = to_light * rsqrt(dist_sq);
                            Float cos_shading   = max(0.0f, dot(ns, light_dir));
                            Float3 light_normal = tri_light->normal();
                            Float cos_light     = max(0.0f, dot(light_normal, -light_dir));

                            Float p_hat = evaluate_p_hat_with_geometry(
                                bsdf_mis,
                                tri_light->emission(), light_dir, dist_sq, cos_shading, cos_light,
                                ns, wo);
                            p_hat = p_hat * glass_att;

                            // Light-BRDF MIS: blend light area PDF with BRDF area PDF
                            Float brdf_pdf_area     = bsdf_mis.pdf(wo, light_dir, ns)
                                                    * cos_light / max(dist_sq, 1e-6f);
                            //Float light_pdf         = 1.0f / max(pc.inv_source_pdf, 1e-10f);
                            Float blended_pdf       = max(1.0f / max(pc.inv_source_pdf, 1e-10f) + brdf_pdf_area, 1e-10f);
                            Float w_i               = p_hat / blended_pdf;
                            w_sum_acc               = w_sum_acc + w_i;
                            //Float u_accept          = cast<float>(seed) * (1.0f / 4294967296.0f);
                            seed                    = util::lcg_ui(seed);

                            // P1-12: explicit zero-denominator guard — with all-zero
                            // p_hat the raw w_i/w_sum_acc is 0/0 = NaN and the compare
                            // only "works" because NaN < x is false.
                            Float accept_p = ite(w_sum_acc > 0.0f, w_i / w_sum_acc, 0.0f);
                            $if(Expr{ cast<float>(seed) * (1.0f / 4294967296.0f) } < accept_p) {
                                r.light_idx     = light_idx;
                                r.target_pdf    = p_hat;
                                r.light_bary_u  = bary_u;
                                r.light_bary_v  = bary_v;
                            };
                            r->set_M(r->M() + 1u);
                        };
                    }
                    $elif(i < M_light + M_env) {
                        // ======== Environment light candidate (from presampled tile) ========
                        $if(env.env_integral > 0.0f) {
                            UInt read_seed  = util::xxhash32(make_uint2(cast<UInt>(i), tile_seed));
                            UInt entry_idx  = read_seed & (kPresampleEnvTileSize - 1u);
                            auto pc         = presample_env_tiles.read(env_tile_base + entry_idx);

                            Float env_u       = pc.bary_u;
                            Float env_v       = pc.bary_v;

                            Float3 env_dir      = uv_to_direction(env_u, env_v, env.env_width, env.env_height, env.env_rotation);
                            Float3 env_radiance = eval_envmap_from_uv(env_u, env_v, env.envmap, env.env_width, env.env_height, env_exposure_val);

                            Float p_hat = evaluate_p_hat_env(
                                bsdf_mis,
                                env_radiance, env_dir, ns, wo);
                            p_hat = p_hat * glass_att;

                            // Light-BRDF MIS: blend env solid-angle PDF with BRDF solid-angle PDF
                            Float brdf_pdf_sa = bsdf_mis.pdf(wo, env_dir, ns);
                            Float blended_pdf = max(1.0f / max(pc.inv_source_pdf, 1e-10f) + brdf_pdf_sa, 1e-10f);
                            Float w_i         = p_hat / blended_pdf;
                            w_sum_acc         = w_sum_acc + w_i;
                            Float u_accept    = cast<float>(seed) * (1.0f / 4294967296.0f);
                            seed = util::lcg_ui(seed);

                            Float accept_p_env = ite(w_sum_acc > 0.0f, w_i / w_sum_acc, 0.0f);  // P1-12
                            $if(u_accept < accept_p_env) {
                                r.light_idx     = kEnvLightSentinel;
                                r.target_pdf    = p_hat;
                                r.light_bary_u  = env_u;
                                r.light_bary_v  = env_v;
                            };
                            r->set_M(r->M() + 1u);
                        };
                    }
                    $else {
                        // Skip BRDF candidates for narrow-lobe surfaces (roughness < cutoff).
                        // On narrow-lobe metals, the BRDF ray direction is concentrated near
                        // the reflection peak; frame-to-frame random sampling within the lobe
                        // hits/misses bright env features (sun, windows), producing specks.
                        // Env presampling (M_env>=4) provides better coverage for these surfaces.
                        // Diffuse / rough surfaces keep BRDF candidates (no variance issue there).
                        // Exception: a coat layer (composed slot 0) must stay BRDF-sampleable
                        // even over a polished base — at grazing angles the coat is the only
                        // energetic lobe and the BRDF candidate is its sole discoverer
                        // (see docs/clearcoat_sampling.md).
                        $if(surface.roughness >= p.brdfCandidateRoughnessCutoff
                            | Expr{ bsdf_mis.has_composed_lobe_list
                                  & (bsdf_mis.lobe_list.weights[0] > 0.f) }) {
                        // ======== BRDF-sampled candidate ========
                        Float u_brdf_x = util::uniform_uint_to_float(seed);
                        seed = util::lcg_ui(seed);
                        Float u_brdf_y = util::uniform_uint_to_float(seed);
                        seed = util::lcg_ui(seed);

                        Float brdf_pdf;
                        Float3 wi = bsdf_mis.sample(wo, ns, make_float2(u_brdf_x, u_brdf_y), brdf_pdf);

                        $if(brdf_pdf > 1e-8f & luisa::compute::dot(wi, ns) > 0.0f) {
                            Float offset  = max(0.001f * depth, 1e-4f);
                            auto brdf_ray = make_ray(
                                world_pos + facing_ns * offset + wi * (0.25f * offset),
                                wi, offset, 1e10f);
                            auto brdf_hit = render::trace_closest(accel, brdf_ray
#if NT_ENABLE_PROCEDURAL
                                , proc_bindless
#endif
                            );

                            $if(!brdf_hit->miss()) {
                                UInt hit_inst   = brdf_hit.inst;
                                UInt hit_prim   = brdf_hit.prim;
                                Float2 hit_bary = brdf_hit.bary;

                                UInt light_base = lights.instance_to_light_base.read(hit_inst);
                                $if(light_base != ~0u) {
                                    UInt light_idx = min(light_base + hit_prim, lights.emissive_count - 1u);

                                    auto tri_light = lights.triangle_lights->read(light_idx);
                                    $if(tri_light->instance_id == hit_inst) {
                                        auto verts = lights.triangle_vertices->read(light_idx);

                                        Float3 light_point = (1.0f - hit_bary.x - hit_bary.y) * verts.v0
                                            + hit_bary.x * verts.v1
                                            + hit_bary.y * verts.v2;

                                        // BRDF cutoff: discard distant lights that contribute negligibly
                                        Float3 to_light_vec = light_point - world_pos;
                                        Float dist_sq_raw   = dot(to_light_vec, to_light_vec);
                                        Float maxDist       = ((1.0f / p.brdfCutoff - 1.0f) * brdf_pdf);

                                        $if(dist_sq_raw <= maxDist * maxDist) {
                                            // Lighting geometry — computed once, shared between
                                            // p_hat and BRDF-PDF MIS blending.
                                            Float dist_sq       = max(dist_sq_raw, 1e-6f);
                                            Float3 light_dir    = to_light_vec * rsqrt(dist_sq);
                                            Float cos_shading   = max(0.0f, dot(ns, light_dir));
                                            Float3 light_normal = tri_light->normal();
                                            Float cos_light     = max(0.0f, dot(light_normal, -light_dir));

                                            Float p_hat = evaluate_p_hat_with_geometry(
                                                bsdf_mis,
                                                tri_light->emission(), light_dir, dist_sq, cos_shading, cos_light,
                                                ns, wo);
                                            p_hat = p_hat * glass_att;

                                            Float  brdf_pdf_area = brdf_pdf * cos_light / max(dist_sq, 1e-6f);

                                            // Light-BRDF MIS: blend with light's area PDF
                                            Float light_pdf;
                                            if constexpr (kUniformLightSampling) {
                                                light_pdf = lights.emissive_count_inv / max(tri_light.area, 1e-10f);
                                            } else {
                                                light_pdf = tri_light.pdf / max(tri_light.area, 1e-10f);
                                            }
                                            Float blended_pdf = max(light_pdf + brdf_pdf_area, 1e-10f);

                                            Float w_i       = p_hat / blended_pdf;
                                            w_sum_acc       = w_sum_acc + w_i;
                                            Float u_accept  = cast<float>(seed) * (1.0f / 4294967296.0f);
                                            seed            = util::lcg_ui(seed);

                                            Float accept_p_brdf = ite(w_sum_acc > 0.0f, w_i / w_sum_acc, 0.0f);  // P1-12
                                            $if(u_accept < accept_p_brdf) {
                                                r.light_idx = light_idx;
                                                r.target_pdf = p_hat;
                                                r.light_bary_u = hit_bary.x;
                                                r.light_bary_v = hit_bary.y;
                                            };
                                            r->set_M(r->M() + 1u);
                                        };
                                    };
                                } $else {
                                    // BRDF ray missed geometry - env map contribution
                                    $if(env.env_integral > 0.0f) {
                                        Float3 env_dir = wi;
                                        Float3 env_radiance = eval_envmap_radiance(env_dir, env.envmap, env.env_width, env.env_height, env.env_rotation, env_exposure_val);
                                        Float p_hat = evaluate_p_hat_env(
                                            bsdf_mis,
                                            env_radiance, env_dir, ns, wo);
                                        p_hat = p_hat * glass_att;

                                        // MIS blend BRDF PDF with env PDF to prevent
                                        // fireflies when brdf_pdf is small
                                        Float env_pdf    = eval_envmap_pdf(
                                            env_dir, env.env_marginal_cdf, env.env_conditional_cdf,
                                            env.env_width, env.env_height, env.env_integral, env.env_rotation);
                                        Float source_pdf = max(brdf_pdf + env_pdf, 1e-10f);

                                        Float w_i       = p_hat / source_pdf;
                                        w_sum_acc       = w_sum_acc + w_i;
                                        Float u_accept  = cast<float>(seed) * (1.0f / 4294967296.0f);
                                        seed            = util::lcg_ui(seed);

                                        Float accept_p_benv = ite(w_sum_acc > 0.0f, w_i / w_sum_acc, 0.0f);  // P1-12
                                        $if(u_accept < accept_p_benv) {
                                            Float2 env_uv   = direction_to_envmap_uv(env_dir, env.env_width, env.env_height, env.env_rotation);

                                            r.light_idx     = kEnvLightSentinel;
                                            r.target_pdf    = p_hat;
                                            r.light_bary_u  = env_uv.x;
                                            r.light_bary_v  = env_uv.y;
                                        };
                                        r->set_M(r->M() + 1u);
                                    };
                                };
                            };
                        };
                        };  // close $if(roughness >= cutoff) BRDF candidate gate
                    };
                };

                r.w_sum     = luisa::compute::min(w_sum_acc, p.wSumCap);

                // Empty-but-alive reservoirs (docs/di-tile-block-artifact-fix-plan.md
                // P1-1): "nothing accepted" (every candidate had p_hat == 0 — e.g.
                // an env-only surface whose 64-entry tile window holds no usable
                // entry) keeps M > 0 so temporal/spatial reuse can still rescue
                // the pixel from history/neighbors. Only w_sum is zeroed — the
                // boiling filter averages w_sum/M and weight() already returns 0
                // via the target_pdf <= 1e-8 guard. RTXDI parity: per-stratum
                // finalize ends state.M = 1; occluded samples "Keep M".
                Bool found  = r.light_idx != ~0u;
                r.w_sum     = ite(found, r.w_sum, 0.0f);

                // Initial visibility filter for env lights: trace a conservative
                // shadow ray and invalidate the reservoir if occluded by opaque
                // geometry. Env lights are at infinity, so any opaque occluder
                // blocks the contribution entirely. Without this filter, occluded
                // env directions propagate through temporal/spatial reuse and
                // produce the "shadows grow with env exposure" artifact, since
                // env p_hat scales with exposure and dominates pairwise MIS.
                $if(found & (r.light_idx == kEnvLightSentinel)) {
                    Float3 env_dir_init = uv_to_direction(
                        r.light_bary_u, r.light_bary_v,
                        env.env_width, env.env_height, env.env_rotation);
                    Float env_offset = max(0.001f * depth, 1e-4f);
                    auto env_shadow_ray = make_ray(
                        world_pos + facing_ns * env_offset + env_dir_init * (0.25f * env_offset),
                        env_dir_init, env_offset, 1e10f);
                    Bool opaque_occluder = def(false);
                    if (hasTransparentShadowCasters == 0u) {
                        // Opaque scene: any-hit is exact — skips the closest-hit
                        // traversal plus occluder material read below.
                        opaque_occluder = render::trace_occluded(accel, env_shadow_ray
#if NT_ENABLE_PROCEDURAL
                            , proc_bindless
#endif
                        );
                    } else {
                        auto env_shadow_hit = render::trace_closest(accel, env_shadow_ray
#if NT_ENABLE_PROCEDURAL
                            , proc_bindless
#endif
                        );
                        $if(!env_shadow_hit->miss()) {
                            UInt   es_inst      = env_shadow_hit.inst;
                            UInt   es_prim      = env_shadow_hit.prim;
                            Float2 es_bary      = env_shadow_hit.bary;
                            // Procedural blocker: es_inst is the shared proc TLAS slot
                            // (out of range for instance_buffer) — classify via the
                            // AABB's own material layers. Emissive passes; no
                            // alpha-cutout on procedural geometry today.
#if NT_ENABLE_PROCEDURAL
                            $if(env_shadow_hit.is_procedural) {
                                Var<scene::ProcInstanceData> es_proc = proc_bindless
                                    .buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                                    .read(es_prim);
                                auto es_mat = scene.material_buffer.read(
                                    Expr{ es_proc.material_layers & 0xFFu });
                                opaque_occluder = !(es_mat.type == 5u);
                            } $else {
#endif
                            UInt4  es_inst_data = scene.instance_buffer.read(es_inst);
                            auto   es_mat       = scene.material_buffer.read(
                                Expr{ es_inst_data.y & 0xFFu });
                            // Match shade pass trace_shadow: glass(3)/emissive(5)/thin(11) and
                            // alpha-cutouts pass through; everything else is an occluder.
                            Bool is_transparent_type = (es_mat.type == 3u
                                | es_mat.type == 5u
                                | es_mat.type == 11u);
                            Bool is_cutout = is_alpha_cutout(
                                es_mat, vertex_bindless, tex_bindless,
                                es_inst_data, es_prim, es_bary);
                            opaque_occluder = (!is_transparent_type) & (!is_cutout);
#if NT_ENABLE_PROCEDURAL
                            };
#endif
                        };
                    };
                    $if(opaque_occluder) {
                        // Keep the reservoir intact (RTXDI: "Keep M for
                        // correct resampling"). The old full kill zeroed
                        // packed_meta → M=0 → is_valid()=false, resetting the
                        // pixel's whole temporal chain on every occluded
                        // selection and starving penumbra edges of
                        // accumulation (shadow edges dimmed while the center
                        // converged). Mark visibility unconfirmed instead:
                        // shade re-traces and zeroes this frame's
                        // contribution, the temporal re-trace drops the
                        // occluded sample from merges, and spatial consumers
                        // reset visibility on win — the exposure-growth
                        // artifact this filter guards against stays covered.
                        r->set_visibility(0u);
                        r->set_vis_age(0u);
                    };
                };

                seed_image.write(coord, make_uint4(seed));
            };

            reservoir_buffer->write(pixel_index, r);
        });

        //==========================================================================
        // ReSTIR Temporal Reuse kernel
        //==========================================================================
        _temporalReuseShader = device.compile<2>([&](
            BufferVar<DIParams> params,
            BufferVar<Reservoir> reservoir_buffer,
            BufferVar<Reservoir> reservoir_prev,
            ImageFloat gbuf_depth,
            ImageUInt  gbuf_vis,
            ImageFloat gbuf_bary_motion,
            UInt       frame_count,
            Var<util::CameraData> camera,
            Var<SceneGeometryResources> scene,
            BindlessVar vertex_bindless,
            BindlessVar tex_bindless,
            Var<LightSamplingResources> lights,
            Var<EnvLightResources> env,
            Float env_exp,
            UInt cbField,
            // Glass throughput for p_hat attenuation
            ImageFloat glass_throughput,
            // Previous frame G-buffer for temporal validation
            ImageFloat gbuf_depth_prev,
            ImageUInt  gbuf_vis_prev,
            ImageFloat gbuf_normal_prev,
            // RAY_TRACED temporal bias correction (RTXDI Medium preset):
            // re-trace prev's reused sample from the current surface
            UInt       diTemporalBiasRayTraced,
            AccelVar   accel
#if NT_ENABLE_PROCEDURAL
	            , BindlessVar proc_bindless
#endif
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("DI_Temporal");
            auto p = params.read(0u);

            // Compile-time-specialized flags (former UInt args; baked by
            // compileImpl): diBiasCorrectionEnabled (DI BASIC piSum MIS) and
            // hasTransparentShadowCasters. C++ consts so DXC folds the dead
            // branches away.
            const uint diBiasCorrectionEnabled = _bakedDiBiasCorrectionEnabled;
            const uint hasTransparentShadowCasters = _bakedTransparentShadowCasters;
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

            Var<Reservoir> current = reservoir_buffer.read(pixel_index);
            UInt4  cur_vis   = gbuf_vis.read(coord);
            UInt   cur_inst  = cur_vis.x;

            //Bool is_point_tp = ((cur_vis.y >> 30u) & 1u) > 0u;

            // P1-2: no is_valid() gate on the current reservoir — an
            // empty-but-alive center (M>0, no accepted candidate) must still
            // merge: with w_c = target_pdf(0)·M = 0 the MIS threshold makes
            // history win with probability 1 (RTXDI combines curSample
            // unconditionally). Fetches of the center's own light are guarded
            // at their sites (P1-3/P1-4).
            $if(cur_inst != ~0u & !Expr{ ((cur_vis.y >> 30u) & 1u) > 0u }) {
                Float2 ndc          = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
                Float  cur_depth    = gbuf_depth.read(coord).x;
                auto   ray          = camera->generate_ray(ndc);
                Float3 wo           = -ray->direction();
                Bool   cur_is_glass = (cur_vis.y >> 31u) > 0u;
                UInt   cur_prim_id  = cur_vis.y & 0x3FFFFFFFu;
                Float4 bary_motion_tp = gbuf_bary_motion.read(coord);
                Float2 cur_bary     = bary_motion_tp.xy();
                UInt4 inst_data     = scene.instance_buffer.read(cur_inst);
                auto cur_xform      = scene.instance_transform_buffer.read(cur_inst);
#if NT_ENABLE_PROCEDURAL
                Bool cur_is_proc  = ((cur_vis.y >> 29u) & 1u) > 0u;
                // $if (not ite): ite evaluates both arms, making every non-glass
                // pixel pay reconstruct_object_position's bindless tri+vertex reads.
                Float3 world_pos = ray->origin() + ray->direction() * cur_depth;
                $if(cur_is_glass & !cur_is_proc) {
                    world_pos = (cur_xform * make_float4(reconstruct_object_position(vertex_bindless, inst_data.z, inst_data.w, cur_prim_id, cur_bary), 1.0f)).xyz();
                };

                // Resolve surface - branch for procedural vs mesh. NOTE: declare
                // `surface` ONCE here and assign in both branches — declaring it
                // inside $else creates a scoped local that shadows this outer
                // var, leaving the outer surface uninitialized for mesh pixels
                // (silent bug: evaluate_p_hat_from_sample then runs on garbage material
                // params, m_factor collapses to 0, every temporal merge rejects).
                SurfaceData surface;
                $if(cur_is_proc) {
                    // Recompute bary at unjittered pixel center for stable texture sampling.
                    auto tp_ray_unjit = camera->generate_ray(Expr{ ndc - camera->jitter });
                    $if(!cur_is_glass) {
                        cur_bary = reconstruct_unjittered_bary_procedural(
                            proc_bindless, cur_inst, cur_prim_id, cur_bary,
                            tp_ray_unjit->origin(), tp_ray_unjit->direction());
                    };
                    surface = resolve_procedural_surface_textured(
                        resolver, proc_bindless, tex_bindless,
                        cur_inst, cur_prim_id, world_pos, wo,
                        scene.material_buffer,
                        cur_bary,
                        0.0f,
                        Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
                        resolution.x, resolution.y);
                }
                $else {
                    surface = resolve_surface_from_instance(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data, cur_prim_id, cur_bary,
                        scene.material_buffer, wo,
                        cur_xform,
                        0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) }, resolution.x, resolution.y,
                        cur_inst);
                };
#else
                Float3 world_pos = ray->origin() + ray->direction() * cur_depth;
                $if(cur_is_glass) {
                    world_pos = (cur_xform * make_float4(reconstruct_object_position(vertex_bindless, inst_data.z, inst_data.w, cur_prim_id, cur_bary), 1.0f)).xyz();
                };
                SurfaceData surface = resolve_surface_from_instance(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data, cur_prim_id, cur_bary,
                        scene.material_buffer, wo,
                        cur_xform,
                        0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) }, resolution.x, resolution.y,
                        cur_inst);
#endif

                // Blend-rolled-opaque reclass (see Shading.h).
                reclass_blend_rolled_opaque(surface, cur_is_glass);

                // Hoist bsdf_mis for temporal reuse p_hat (single construction per pixel).
                // Probe-covered SSS pixels zero the HK lobe (direct model = the
                // Burley probe; current pixel is point-gated above). Thin-wall
                // subsurface (diffuse_trans > 0) keeps the lobes.
                Bool sss_probe_direct = Expr{ surface.bsdf_type == 6u }
                                      & Expr{ surface.flatness > 0.0f }
                                      & Expr{ surface.diffuse_trans <= 0.0f };
                MaterialBSDF bsdf_mis = surface.make_bsdf(sss_probe_direct);

                Float3 ns       = surface.ns;
                Float3 tangent  = surface.tangent;
                Float  tangent_w= surface.tangent_w;

                // Glass attenuation for p_hat
                Float4 glass_tp = glass_throughput.read(coord);
                Float glass_att = luminance(glass_tp.xyz());

                Float2 motion     = bary_motion_tp.zw();
                Float2 prev_uv    = (ndc + motion + 1.0f) * 0.5f;
                Int2   prev_coord_base = make_int2(prev_uv * make_float2(resolution) - 0.5f);

                // NOTE: no motion-magnitude gate here (the old temporalMotionThresh
                // kill created a circular history-reset contour during dolly/rotate —
                // the |motion| = threshold iso-contour read as a ring artifact).
                // RTXDI ref has no such gate: correctness is guarded by the
                // instance + reprojected-depth + normal checks below, which are
                // position-consistent and cannot partition the image by motion.

                // Compensate camera-z motion (same fix as GI temporal reuse):
                // prev_depth is t along the prev-frame primary ray. Compare against
                // the object's prev-transform reprojected position (depth_ref_pos)
                // so moving geometry matches too, not just static surfaces under
                // camera translation. Current-frame-only data -> hoisted above the
                // neighbor search so every tap reuses it.
                Float3 depth_ref_pos_tp = world_pos;
#if NT_ENABLE_PROCEDURAL
                $if(!cur_is_proc) {
#endif
                Float3 obj_pos_tp = reconstruct_object_position(
                    vertex_bindless, inst_data.z, inst_data.w, cur_prim_id, cur_bary);
                Float4x4 prev_xform_tp = scene.instance_transform_prev_buffer.read(cur_inst);
                depth_ref_pos_tp = (prev_xform_tp * make_float4(obj_pos_tp, 1.0f)).xyz();
#if NT_ENABLE_PROCEDURAL
                };
#endif
                Float3 to_prev_cam         = depth_ref_pos_tp - camera.prev_position;
                Float  expected_prev_depth = luisa::compute::length(to_prev_cam);

                // Ref-parity neighborhood search (RTXDI DI TemporalResampling.hlsli:
                // 71-109). Tap 0 is the reprojected center with permutation sampling
                // (the engine's thin-object decorrelator standing in for
                // RTXDI_ApplyPermutationSampling); taps 1-8 are hashed random offsets
                // within a 4px radius (8px in checkerboard, matching the ref's
                // doubled radius) so history survives the small misreprojections that
                // camera motion produces at edges and grazing surfaces. Every tap runs
                // the same instance + relative-depth + normal validation; the first
                // valid tap carries the merge.
                const float search_radius = _checkerboard ? 8.0f : 4.0f;
                Bool   found_t        = def(false);
                UInt2  prev_coord_uint = def(make_uint2(0u, 0u));
                UInt   prev_pixel      = def(0u);
                UInt4  prev_vis        = def(make_uint4(0u, 0u, 0u, 0u));
                Float  found_normal_dot = def(1.0f);  // found tap's normal sim (temporal MFactor gate)

                $for(search_i, 0u, 9u) {
                    $if(!found_t) {
                        Int2 prev_coord = prev_coord_base;
                        $if(search_i == 0u) {
                            // Permutation sampling: break coherent temporal patterns on thin objects
                            UInt perm_seed   = frame_count * 7919u;
                            Int2 perm_offset = make_int2(
                                cast<int>(perm_seed & 3u),
                                cast<int>((perm_seed >> 2u) & 3u));
                            Int2 prev_coord_perm = prev_coord_base + perm_offset;
                            prev_coord_perm = make_int2(prev_coord_perm.x ^ 3, prev_coord_perm.y ^ 3);
                            prev_coord_perm = prev_coord_perm - perm_offset;

                            // Fall back to unpermuted coordinate when:
                            // 1) XOR pushes OOB at screen edges
                            // 2) Permuted position lands on a different instance (geometric edge crossing)
                            Bool perm_valid = prev_coord_perm.x >= 0 & prev_coord_perm.x < cast<int>(resolution.x)
                                & prev_coord_perm.y >= 0 & prev_coord_perm.y < cast<int>(resolution.y);
                            UInt2 perm_read_coord = make_uint2(
                                cast<uint>(ite(perm_valid, prev_coord_perm.x, prev_coord_base.x)),
                                cast<uint>(ite(perm_valid, prev_coord_perm.y, prev_coord_base.y)));
                            Bool perm_inst_match = Expr{ gbuf_vis_prev.read(perm_read_coord).x == cur_inst };
                            prev_coord = ite(perm_valid & perm_inst_match, prev_coord_perm, prev_coord_base);
                        } $else {
                            Float u1 = util::uniform_uint_to_float(
                                util::xxhash32(make_uint3(pixel_index, frame_count, search_i)));
                            Float u2 = util::uniform_uint_to_float(
                                util::xxhash32(make_uint3(pixel_index, frame_count, search_i + 8u)));
                            prev_coord = prev_coord_base + make_int2(
                                cast<int>((u1 - 0.5f) * search_radius),
                                cast<int>((u2 - 0.5f) * search_radius));
                        };

                        $if(Expr{ prev_coord.x >= 0 & prev_coord.x < cast<int>(resolution.x)
                            & prev_coord.y >= 0 & prev_coord.y < cast<int>(resolution.y) }) {
                            UInt2 cand_coord = make_uint2(cast<uint>(prev_coord.x), cast<uint>(prev_coord.y));
                            UInt  cand_pixel = cast<uint>(prev_coord.y) * resolution.x + cast<uint>(prev_coord.x);
                            if (_checkerboard)
                                cand_pixel = cast<uint>(prev_coord.y) * rsv_res.x + (cast<uint>(prev_coord.x) >> 1u);
                            UInt4 cand_vis = gbuf_vis_prev.read(cand_coord);

                            $if(cand_vis.x == cur_inst) {
                                Var<Reservoir> cand_r = reservoir_prev.read(cand_pixel);
                                // P1-3: light_idx != ~0u made explicit — after P1-1
                                // a history tap can be empty-but-alive; the
                                // target_pdf >= floor term already excludes those
                                // (empties carry target_pdf = 0), this keeps the
                                // sample-validity contract readable at the fetch site.
                                $if(cand_r->is_valid() & cand_r.light_idx != ~0u
                                    & cand_r.target_pdf >= p.targetPdfFloor) {
                                    // Surface similarity check: reject a tap when it
                                    // lands on a different surface than the current hit
                                    Float cand_depth = gbuf_depth_prev.read(cand_coord).x;
                                    Float depth_diff = abs(cand_depth - expected_prev_depth)
                                                        / max(max(cand_depth, expected_prev_depth), 0.01f);
                                    // Normal similarity: read prev-frame world-space normals directly
                                    // (populated by denoiser prefilter at end of previous frame)
                                    Float3 cand_ns     = gbuf_normal_prev.read(cand_coord).xyz();
                                    Float  normal_dot  = luisa::compute::dot(ns, cand_ns);

                                    $if((depth_diff < 0.1f) & (normal_dot > p.spatialNormalThresh)) {
                                        found_t        = true;
                                        prev_coord_uint = cand_coord;
                                        prev_pixel      = cand_pixel;
                                        prev_vis        = cand_vis;
                                        found_normal_dot = normal_dot;
                                    };
                                };
                            };
                        };
                    };
                };

                $if(found_t) {
                    Var<Reservoir> prev_r = reservoir_prev.read(prev_pixel);
                            // Fetch prev's light data once (zero-bindless LightSample).
                            // Reused by the bias-correction inverse query below when
                            // prev's sample wins the merge (same light, same bary).
                            LightSample prev_sample_t = fetch_light_sample(
                                Expr{ prev_r.light_idx == kEnvLightSentinel },
                                prev_r.light_idx, prev_r.light_bary_u, prev_r.light_bary_v,
                                lights.triangle_lights, lights.triangle_vertices,
                                env.envmap, env.env_width, env.env_height,
                                env.env_rotation, env_exp);
                            Float p_hat_new = evaluate_p_hat_from_sample(
                                bsdf_mis, prev_sample_t, world_pos, ns, wo);
                            p_hat_new = p_hat_new * glass_att;

                            Float jacobian = min(p_hat_new / luisa::compute::max(prev_r.target_pdf, p.targetPdfFloor), 3.0f);

                            // RTXDI MFactor (ref: Rtxdi/Utils/Math.hlsli:117). On static surfaces
                            // (normal_dot ≈ 1) RTXDI's basic bias correction via the balance
                            // threshold below is sufficient — skip MFactor cost. Under camera
                            // rotation, even sub-pixel motion produces large BRDF-lobe shifts on
                            // narrow metals; MFactor aggressively rejects stale reservoirs whose
                            // selected light has moved off-lobe. Gate: surface rotated > ~11°.
                            // Skip MFactor when canonical's own target_pdf is degenerate (glass).
                            Float m_factor = ite(current.target_pdf > p.targetPdfFloor * 1000.0f,
                                ite(found_normal_dot < 0.98f,
                                    rtxdi_mfactor(prev_r.target_pdf, p_hat_new, p.mFactorExponent),
                                    1.0f),
                                1.0f);

                            // RAY_TRACED temporal bias correction (RTXDI
                            // TemporalResampling.hlsli:192-200, Medium preset):
                            // re-trace prev's reused sample from the CURRENT
                            // surface; an occluded prev contributes nothing and
                            // current keeps its own fresh candidate (merge and
                            // MIS selection are skipped entirely). Visibility
                            // shortcut (RTXDI enableVisibilityShortcut): skip
                            // the ray when prev's stored visibility is fresh —
                            // shade would trust the same state anyway.
                            Bool prev_visible = def(true);
                            // True when the re-trace's "visible" verdict came
                            // through a tinted-transmit interface (glass/thin/
                            // cutout — the same classes the shade pass's
                            // any(att < 0.999) gate covers; emissive transmits
                            // exactly white). The reservoir can't store RGB
                            // attenuation, so a binary confirmation through
                            // these must not offer shade a reuse entry.
                            Bool vis_transmit_tinted = def(false);
                            Bool vis_shortcut = prev_r->visibility() == 1u
                                & prev_r->vis_age() < p.visMaxAge;
                            $if(diTemporalBiasRayTraced != 0u & !Expr{vis_shortcut}) {
                                Float3 facing_ns_t = ite(dot(wo, ns) < 0.f, -ns, ns);
                                Float  base_t      = max(0.001f * cur_depth, 1e-4f);
                                Float3 dir_t  = def(make_float3(0.0f, 0.0f, 1.0f));
                                Float  tmax_t = def(1e10f);
                                $if(prev_r.light_idx == kEnvLightSentinel) {
                                    dir_t = prev_sample_t.env_dir;
                                } $else {
                                    Float3 to_light_t = prev_sample_t.light_point - world_pos;
                                    Float  dist_t     = luisa::compute::length(to_light_t);
                                    dir_t  = to_light_t * (1.0f / max(dist_t, 1e-6f));
                                    tmax_t = max(dist_t - base_t, 1e-3f);
                                };
                                // NRD GetXoffset origin bias (same as candidate kernel)
                                Float3 org_t = world_pos + facing_ns_t * base_t
                                    + dir_t * (0.25f * base_t);
                                auto vis_ray = make_ray(org_t, dir_t, base_t, tmax_t);
                                Bool occluded = def(false);
                                if (hasTransparentShadowCasters == 0u) {
                                    // Opaque scene: any-hit is exact and cheapest
                                    occluded = render::trace_occluded(accel, vis_ray
#if NT_ENABLE_PROCEDURAL
                                        , proc_bindless
#endif
                                    );
                                } else {
                                    // Conservative classification mirroring the
                                    // candidate env filter: glass(3)/emissive(5)/
                                    // thin(11) and alpha-cutouts pass through.
                                    auto vis_hit = render::trace_closest(accel, vis_ray
#if NT_ENABLE_PROCEDURAL
                                        , proc_bindless
#endif
                                    );
                                    $if(!vis_hit->miss()) {
                                        // Procedural blocker: vis_hit.inst is the shared
                                        // proc TLAS slot (out of range for instance_buffer)
                                        // — classify via the AABB's own material layers.
                                        // Emissive passes; no alpha-cutout on procedural.
#if NT_ENABLE_PROCEDURAL
                                        $if(vis_hit.is_procedural) {
                                            Var<scene::ProcInstanceData> oc_proc = proc_bindless
                                                .buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                                                .read(vis_hit.prim);
                                            auto oc_mat = scene.material_buffer.read(
                                                Expr{ oc_proc.material_layers & 0xFFu });
                                            occluded = !(oc_mat.type == 5u);
                                            vis_transmit_tinted = false;
                                        } $else {
#endif
                                        UInt4 oc_inst_data = scene.instance_buffer.read(vis_hit.inst);
                                        auto  oc_mat = scene.material_buffer.read(
                                            Expr{ oc_inst_data.y & 0xFFu });
                                        Bool is_transparent_type = (oc_mat.type == 3u
                                            | oc_mat.type == 5u
                                            | oc_mat.type == 11u);
                                        Bool is_cutout = is_alpha_cutout(
                                            oc_mat, vertex_bindless, tex_bindless,
                                            oc_inst_data, vis_hit.prim, vis_hit.bary);
                                        occluded = (!is_transparent_type) & (!is_cutout);
                                        vis_transmit_tinted = (oc_mat.type == 3u)
                                            | (oc_mat.type == 11u) | is_cutout;
#if NT_ENABLE_PROCEDURAL
                                        };
#endif
                                    };
                                };
                                prev_visible = !occluded;
                            };

                            $if((m_factor > p.mFactorThreshold) & Expr{prev_visible}) {
                                Float w_sum_merged = current.w_sum + prev_r.w_sum * jacobian * m_factor;
                                UInt  M_merged     = current->M() + prev_r->M();

                                // BASIC bias correction state (RTXDI TemporalResampling.hlsli:173-212).
                                // input_M_t captured BEFORE merge commit = pre-merge canonical M
                                // (matches RTXDI's curSample.M in piSum canonical term).
                                UInt input_M_t       = current->M();
                                Bool merge_happened_t = def(false);
                                Bool prev_won_t       = def(false);

                                // MIS-weighted selection for bias correction:
                                // weight each reservoir proportional to target_pdf * M
                                // instead of raw w_sum, giving better selection when surfaces differ
                                Float w_c       = current.target_pdf * cast<Float>(current->M());
                                Float w_p       = p_hat_new * cast<Float>(prev_r->M());
                                Float mis_total = w_c + w_p;
                                Float threshold = ite(mis_total > 0.0f, w_p / mis_total, 0.0f);
                                UInt seed       = util::xxhash32(make_uint2(pixel_index, frame_count));
                                Float u         = cast<float>(seed) * (1.0f / 4294967296.0f);

                                $if(u < threshold) {
                                    current.light_idx   = prev_r.light_idx;
                                    current.target_pdf  = p_hat_new;
                                    current.light_bary_u = prev_r.light_bary_u;
                                    current.light_bary_v = prev_r.light_bary_v;
                                    // Cache the temporal re-trace result into the
                                    // visibility state (RTXDI_StoreVisibilityInDIReservoir
                                    // semantics — our shade write-back doesn't
                                    // feed temporal's prev slot, so temporal is where the
                                    // freshest verification lives). A traced merge
                                    // stores visibility with age 0 and reset screen
                                    // distance, letting shade skip its shadow ray;
                                    // shortcut merges (ray skipped, state trusted)
                                    // inherit prev's aged state. Exception: a binary
                                    // confirmation through tinted transmitters
                                    // (glass/thin/cutout) parks the age at 15 — shade
                                    // would restore white attenuation on reuse, and
                                    // this ray can't know the tint (same Phase-0
                                    // contract as the shade write-back; visMaxAge
                                    // <= 15 so the reuse gate always fails).
                                    Bool retrace_ran_t = diTemporalBiasRayTraced != 0u
                                        & !Expr{vis_shortcut};
                                    current->set_visibility(ite(retrace_ran_t,
                                        ite(prev_visible, 1u, 0u), prev_r->visibility()));
                                    current->set_vis_age(ite(retrace_ran_t,
                                        ite(prev_visible & vis_transmit_tinted, 15u, 0u),
                                        min(prev_r->vis_age() + 1u, p.visMaxAge)));
                                    current->set_spatial_dist_x(ite(retrace_ran_t,
                                        0, prev_r->spatial_dist_x()
                                        + cast<int>(coord.x) - cast<int>(prev_coord_uint.x)));
                                    current->set_spatial_dist_y(ite(retrace_ran_t,
                                        0, prev_r->spatial_dist_y()
                                        + cast<int>(coord.y) - cast<int>(prev_coord_uint.y)));
                                    prev_won_t       = def(true);
                                    merge_happened_t = def(true);
                                } $else {
                                    merge_happened_t = def(true);
                                };

                                UInt M_capped = min(M_merged, p.temporalMaxM);
                                Float w_scale = cast<float>(M_capped) / cast<float>(max(M_merged, 1u));
                                current.w_sum = luisa::compute::min(w_sum_merged * w_scale, p.wSumCap);
                                current->set_M(M_capped);

                                // BASIC bias correction (RTXDI TemporalResampling.hlsli:173-212;
                                // GI PassGI.cpp:964-1026 template). Single-neighbor piSum with
                                // inverse query at prev surface. current.target_pdf post-merge =
                                // winner's pdf at current surface (RTXDI state.targetPdf).
                                if (diBiasCorrectionEnabled != 0u) {
                                    $if(merge_happened_t) {
                                    Bool prev_is_proc = ((prev_vis.y >> 29u) & 1u) > 0u;
                                    $if(prev_vis.x != ~0u) {
                                        // Reconstruct prev surface. Use world_pos / wo from current
                                        // frame since (depth_diff, normal_dot) gate already certified
                                        // co-surface. Bary: gbuf_bary_motion_prev not retained, fall
                                        // back to current-frame bary at prev pixel (same triangle by
                                        // similarity gate). Read prev world-space normals from
                                        // gbuf_normal_prev (populated by denoiser prefilter).
                                        Float3 prev_world_pos = world_pos;
                                        Float3 prev_wo        = wo;
                                        Float2 prev_bary      = gbuf_bary_motion.read(prev_coord_uint).xy();
                                        UInt   prev_prim      = prev_vis.y & 0x3FFFFFFFu;
                                        UInt4  prev_inst_data = scene.instance_buffer.read(prev_vis.x);
                                        Float2 prev_screen_uv = (make_float2(prev_coord_uint) + 0.5f) / make_float2(resolution);
                                        SurfaceData prev_surface;
#if NT_ENABLE_PROCEDURAL
                                        $if(prev_is_proc) {
                                            prev_surface = resolve_procedural_surface_textured(
                                                resolver, proc_bindless, tex_bindless,
                                                prev_vis.x, prev_prim, prev_world_pos, prev_wo,
                                                scene.material_buffer, prev_bary,
                                                0.0f, prev_screen_uv, resolution.x, resolution.y);
                                        }
                                        $else {
#endif
                                            prev_surface = resolve_surface_from_instance(
                                                resolver, vertex_bindless, tex_bindless,
                                                prev_inst_data, prev_prim, prev_bary,
                                                scene.material_buffer, prev_wo,
                                                scene.instance_transform_buffer.read(prev_vis.x),
                                                0.0f, prev_screen_uv, resolution.x, resolution.y,
                                                prev_vis.x);
#if NT_ENABLE_PROCEDURAL
                                        };
#endif
                                        // Blend-rolled-opaque reclass (see Shading.h) — prev glass bit.
                                        reclass_blend_rolled_opaque(
                                            prev_surface, Expr{ (prev_vis.y >> 31u) > 0u });

                                        // Inverse query: winner's light at prev surface.
                                        // Probe-covered SSS zeroing applies too; a
                                        // point-primitive prev pixel keeps the HK lobe.
                                        // Thin-wall prev pixels keep the lobes.
                                        Bool prev_sss_probe = Expr{ prev_surface.bsdf_type == 6u }
                                                            & Expr{ prev_surface.flatness > 0.0f }
                                                            & Expr{ prev_surface.diffuse_trans <= 0.0f }
                                                            & !Expr{ ((prev_vis.y >> 30u) & 1u) > 0u };
                                        MaterialBSDF prev_bsdf = prev_surface.make_bsdf(prev_sss_probe);
                                        // When prev's sample won, the winner IS prev_sample_t —
                                        // skip the duplicate bindless fetch chain.
                                        Float ps_prev = def(0.0f);
                                        $if(prev_won_t) {
                                            ps_prev = evaluate_p_hat_from_sample(
                                                prev_bsdf, prev_sample_t,
                                                prev_world_pos, prev_surface.ns, prev_wo);
                                        } $else {
                                            // P1-4: reachable with an EMPTY current
                                            // reservoir (P1-2 dropped the center gate;
                                            // both merge weights 0 ⇒ threshold 0 ⇒
                                            // current "won" a sample it doesn't have).
                                            // Guard the fetch — reading
                                            // triangle_lights[~0u] is out of bounds.
                                            $if(current.light_idx != ~0u) {
                                                LightSample winner_sample_t = fetch_light_sample(
                                                    Expr{ current.light_idx == kEnvLightSentinel },
                                                    current.light_idx, current.light_bary_u, current.light_bary_v,
                                                    lights.triangle_lights, lights.triangle_vertices,
                                                    env.envmap, env.env_width, env.env_height,
                                                    env.env_rotation, env_exp);
                                                ps_prev = evaluate_p_hat_from_sample(
                                                    prev_bsdf, winner_sample_t,
                                                    prev_world_pos, prev_surface.ns, prev_wo);
                                            };
                                        };

                                        // piSum = (winner @ current) · input_M + (winner @ prev) · prev_M
                                        // pi    = winner's pdf at the surface that originated it.
                                        Float piSum_t = current.target_pdf * cast<Float>(input_M_t)
                                                      + ps_prev * cast<Float>(prev_r->M());
                                        Float pi_t    = ite(prev_won_t, ps_prev, current.target_pdf);

                                        // DI storage is streaming-form (shade does W = w_sum / (M·target_pdf)).
                                        // RTXDI/GI BASIC finalize: weightSum *= pi / (target_pdf · piSum).
                                        // Solving for equivalent pre-shade multiplier on DI's w_sum:
                                        //   w_sum · X / (M_capped · target_pdf) = w_sum_merged · pi / (target_pdf · piSum)
                                        //   X = pi · M_merged / piSum
                                        // M_merged is pre-cap (input_M + prev_M). For similar diffuse
                                        // this evaluates to 1 (true no-op), matching RTXDI/GI semantics.
                                        Float M_merged_t = cast<Float>(input_M_t + prev_r->M());

                                        $if(piSum_t > 1e-10f) {
                                            current.w_sum = luisa::compute::min(
                                                current.w_sum * pi_t * M_merged_t / piSum_t, p.wSumCap);
                                        };
                                    };
                                };
                                } // if (diBiasCorrectionEnabled) — compile-time

                                reservoir_buffer.write(pixel_index, current);
                            };  // merge gate (m_factor + visibility)
                        };  // found_t (first valid neighborhood tap)
                    };  // current surface valid
                });

        if (!resolverOnly) {
        //==========================================================================
        // Boiling Filter (DI): discard outlier reservoirs within 16x16 blocks
        // P3-1: a frame-global average is blended under the block average so
        // blocks sitting far below the image average don't manufacture an
        // artificially low kill threshold (false rejection of valid bright
        // reservoirs). Computed by the DI_BoilingStats pre-pass below.
        //==========================================================================
        // R3 wave/smem pass: cross-warp reduction via one smem entry per lane +
        // warp_active_sum (the round-1 item-12 pattern GI_BoilingFilter already
        // ships), replacing lane 0's serial 8-iteration sum while the warp
        // idles; OOB threads stay alive with zeroed reservoirs instead of
        // returning before the sync_block (barrier-uniformity hazard; also the
        // GI pattern). Shuffle reduction reorders the float additions, so avg_w
        // can differ at ULP level — gate OFF restores the legacy kernel
        // bit-for-bit for the disabled-gate test.
        constexpr bool kR3WarpReduceBoiling = true;
        _boilingFilterDI = device.compile<2>([&](
            BufferVar<Reservoir> reservoir_buffer,
            Float strength,
            UInt cbField,
            BufferVar<float> boiling_stats
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("DI_BoilingFilter");
            UInt2 rsv       = dispatch_id().xy();
            UInt2 rsv_res   = dispatch_size().xy();
            Bool in_bounds  = !(any(rsv >= rsv_res));
            UInt pixel_index = rsv.y * rsv_res.x + rsv.x;
            UInt local_id    = (dispatch_id().y % 16u) * 16u + (dispatch_id().x % 16u);

            Shared<float> shared_warp_w(8u);
            Shared<uint>  shared_warp_v(8u);

            Var<Reservoir> r;
            if constexpr (!kR3WarpReduceBoiling) {
                // Legacy path: OOB threads return early (safe only because
                // every bench dispatch is a multiple of 16).
                $if(!in_bounds) { $return(); };
                r = reservoir_buffer.read(pixel_index);
            } else {
                // OOB threads stay alive and contribute zeros (M=0 ⇒ W=0,
                // is_valid=false) so the reductions and the sync_block below
                // remain well-defined at screen edges — the GI_BoilingFilter
                // pattern since round-1 item 12.
                $if(in_bounds) {
                    r = reservoir_buffer.read(pixel_index);
                } $else {
                    r.light_idx    = 0u;
                    r.w_sum        = 0.0f;
                    r.target_pdf   = 0.0f;
                    r.light_bary_u = 0.0f;
                    r.light_bary_v = 0.0f;
                    r.packed_meta  = 0u;
                };
            }

            // Use w_sum/M (average RIS weight per sample) for outlier detection,
            // NOT weight() which divides by target_pdf and inflates when target_pdf
            // is small — causing false rejection of valid top-edge reservoirs.
            Float W     = ite(r->M() > 0u, r.w_sum / cast<float>(r->M()), 0.0f);
            Bool valid  = r->is_valid();

            UInt warp_id = local_id / 32u;
            UInt lane_id = local_id % 32u;

            // Warp-level reduction — no sync needed within a warp
            Float warp_total_w = warp_active_sum(W);
            // P1-10: count W > 0 (ref BoilingFilter.hlsli counts weight > 0), not
            // is_valid() — empty-but-alive reservoirs (w_sum = 0) must not dilute
            // the block average.
            UInt warp_total_v = warp_active_sum(ite(W > 0.0f, 1u, 0u));

            // Lane 0 writes per-warp result to shared memory (8 entries, 64 bytes)
            $if(lane_id == 0u) {
                shared_warp_w[warp_id] = warp_total_w;
                shared_warp_v[warp_id] = warp_total_v;
            };
            sync_block();

            Float total_w;
            UInt  total_v;
            if constexpr (!kR3WarpReduceBoiling) {
                // Legacy: lane 0 serially sums the 8 cross-warp partials,
                // then broadcasts via warp_read_first_active_lane.
                total_w = def(0.0f);
                total_v = def(0u);
                $if(lane_id == 0u) {
                    for (uint w = 0u; w < 8u; w++) {
                        total_w = total_w + shared_warp_w[w];
                        total_v = total_v + shared_warp_v[w];
                    };
                };
                total_w = warp_read_first_active_lane(total_w);
                total_v = warp_read_first_active_lane(total_v);
            } else {
                // 8 cross-warp results: one smem entry per lane + warp shuffle
                // reduction (round-1 item-12 fix, as shipped in GI_BoilingFilter).
                // Index clamped so non-reader lanes stay in bounds; ite masks
                // their contribution to zero.
                Bool sum_reader = lane_id < 8u;
                Float lane_w = ite(sum_reader, shared_warp_w[min(lane_id, 7u)], 0.0f);
                UInt  lane_v = ite(sum_reader, shared_warp_v[min(lane_id, 7u)], 0u);
                total_w = warp_active_sum(lane_w);
                total_v = warp_active_sum(lane_v);
            }

            // Block average weight (warp_active_sum broadcasts to every lane)
            Float avg_w = total_w / max(cast<float>(total_v), 1.0f);
            // P3-1: frame-global average of w_sum/M over W>0 reservoirs,
            // computed by the DI_BoilingStats single-block pre-pass (stats[0]
            // holds the average directly; 0 when no W>0 reservoirs exist,
            // which degenerates to the plain block average). The floor at
            // half the global average keeps far-below-average blocks from
            // producing an over-aggressive kill threshold.
            Float global_avg = boiling_stats.read(0u);
            Float eff_avg    = max(avg_w, 0.5f * global_avg);
            // Discard outliers: weight > eff_avg * multiplier
            // Skip outlier test near screen edges where block statistics are unreliable
            // (blocks at edges have a mix of geometry/sky that skews the average,
            //  causing false rejection of valid edge reservoirs)
            Bool near_edge = rsv.x < 8u | rsv.y < 8u
                | rsv.x + 8u >= rsv_res.x | rsv.y + 8u >= rsv_res.y;
            Float multiplier = 10.0f / max(strength, 1e-6f) - 9.0f;
            $if(valid & Expr{ !near_edge } & W > eff_avg * multiplier) {
                // Kill = M reset; after P1-5 the spatial pass can re-populate
                // killed centers from neighbors within the same frame.
                r->set_M(0u);
                r.w_sum      = 0.0f;
                r.target_pdf = 0.0f;
            };

            // OOB threads ran with a zeroed reservoir; only in-bounds pixels
            // write back (the legacy path returned before reaching here).
            $if(in_bounds) {
                reservoir_buffer.write(pixel_index, r);
            };
        });

        //==========================================================================
        // Boiling stats pre-pass (P3-1): frame-global average of w_sum/M over
        // W>0 reservoirs. PERF NOTE: the first version accumulated with float
        // buffer atomics (grid-wide fetch_add into stats[0..1]) — the DX
        // float-atomic emulation path cost ~0.45 ms/frame for ~4000 ops
        // (measured 2026-09-19, cornell perf harness: 82.6 vs 85.9 fps).
        // This single-block variant reduces in shared memory instead: one
        // dispatch, no atomics, no separate clear pass, thread 0 writes the
        // final average. Reads a 1/256 systematic subsample (~4K entries).
        //==========================================================================
        _boilingStatsDI = device.compile<1>([&](
            BufferVar<Reservoir> reservoir_buffer,
            BufferVar<float> stats_out,
            UInt total
        ) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("DI_BoilingStats");
            UInt tid = dispatch_id().x;
            // Subsample stride = block size (256 threads) × 1/256 — thread t
            // accumulates entries {it*kSubsample + t*256}, so the block covers
            // every 256th reservoir overall.
            constexpr uint kSubsample = 256u * 256u;
            UInt iters = (total + kSubsample - 1u) / kSubsample;
            Float w_acc = def(0.0f);
            UInt  v_acc = def(0u);
            $for(it, iters) {
                UInt idx = it * kSubsample + tid * 256u;
                $if(idx < total) {
                    Var<Reservoir> r = reservoir_buffer.read(idx);
                    Float W = ite(r->M() > 0u, r.w_sum / cast<float>(r->M()), 0.0f);
                    w_acc = w_acc + ite(W > 0.0f, W, 0.0f);
                    v_acc = v_acc + ite(W > 0.0f, 1u, 0u);
                };
            };

            // Block reduction (same shared/warp pattern as the boiling filter).
            Shared<float> shared_w(8u);
            Shared<uint>  shared_v(8u);
            UInt warp_id = tid / 32u;
            UInt lane_id = tid % 32u;
            Float warp_w = warp_active_sum(w_acc);
            UInt  warp_v = warp_active_sum(v_acc);
            $if(lane_id == 0u) {
                shared_w[warp_id] = warp_w;
                shared_v[warp_id] = warp_v;
            };
            sync_block();
            Float total_w;
            UInt  total_v;
            if constexpr (!kR3WarpReduceBoiling) {
                // Legacy: lane 0 serially sums the 8 cross-warp partials.
                total_w = def(0.0f);
                total_v = def(0u);
                $if(lane_id == 0u) {
                    for (uint w = 0u; w < 8u; w++) {
                        total_w = total_w + shared_w[w];
                        total_v = total_v + shared_v[w];
                    };
                };
                total_w = warp_read_first_active_lane(total_w);
                total_v = warp_read_first_active_lane(total_v);
            } else {
                // R3 (round-1 item-12 pattern, as shipped in GI_BoilingFilter):
                // one smem entry per lane + warp_active_sum instead of lane 0's
                // serial 8-iteration loop. ULP-level reorder of the float sum.
                Bool sum_reader = lane_id < 8u;
                Float lane_w = ite(sum_reader, shared_w[min(lane_id, 7u)], 0.0f);
                UInt  lane_v = ite(sum_reader, shared_v[min(lane_id, 7u)], 0u);
                total_w = warp_active_sum(lane_w);
                total_v = warp_active_sum(lane_v);
            }
            $if(tid == 0u) {
                stats_out->write(0u, total_w / max(cast<float>(total_v), 1.0f));  // global avg
                stats_out->write(1u, cast<float>(total_v));                       // subsampled W>0 count
            };
        });
        } // end resolverOnly gate (BoilingFilterDI)

        //==========================================================================
        // ReSTIR Spatial Reuse kernel
        //==========================================================================
        _spatialReuseShader = device.compile<2>([&](
            BufferVar<DIParams> params,
            BufferVar<Reservoir> reservoir_output,
            BufferVar<Reservoir> reservoir_input,
            ImageFloat gbuf_depth,
            ImageUInt  gbuf_vis,
            ImageFloat gbuf_bary_motion,
            UInt       frame_count,
            Var<SceneGeometryResources> scene,
            BindlessVar vertex_bindless,
            BindlessVar tex_bindless,
            Var<LightSamplingResources> lights,
            Var<util::CameraData> camera,
            Var<EnvLightResources> env,
            Float env_exp,
            UInt cbField,
            // Glass throughput for p_hat attenuation
            ImageFloat glass_throughput,
            // World normals + matID/roughness for CGNS candidate scoring
            // (prefilter writes this before the DI reuse passes; sky
            // sentinel packed_nr_w = 255.0)
            ImageFloat denoise_normal
#if NT_ENABLE_PROCEDURAL
	            , BindlessVar proc_bindless
#endif
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("DI_SpatialReuse");
            auto p = params.read(0u);

            // Compile-time-specialized flag (former UInt arg; baked by
            // compileImpl): DI BASIC piSum MIS in the neighbor merge. C++
            // const so DXC folds the dead branch away.
            const uint diBiasCorrectionEnabled = _bakedDiBiasCorrectionEnabled;
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

            Var<Reservoir> r = reservoir_input.read(pixel_index);
            UInt4 vis     = gbuf_vis.read(coord);
            UInt  inst_id = vis.x;
            //Bool is_point_sp = ((vis.y >> 30u) & 1u) > 0u;

            // P1-5: no is_valid() gate on the center — an empty-but-alive
            // reservoir (M>0, no accepted candidate) must receive neighbor
            // samples: its zero w_sum gives threshold w_neighbor/(0+w_neighbor)=1,
            // so any viable neighbor wins (RTXDI combines the center
            // unconditionally). Neighbor gates below still require a selected
            // sample; the winner fetches are guarded (P1-6).
            $if(inst_id != ~0u & !Expr{ ((vis.y >> 30u) & 1u) > 0u }) {
                Float depth = gbuf_depth.read(coord).x;
                auto   ray = camera->generate_ray(Expr{
                    (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f
                });
                Float3 wo           = -ray->direction();
                Bool   is_glass_sp  = (vis.y >> 31u) > 0u;
                UInt   prim_id_sp   = vis.y & 0x3FFFFFFFu;
                Float2 bary_sp      = gbuf_bary_motion.read(coord).xy();
                UInt4 inst_data     = scene.instance_buffer.read(inst_id);
                auto sp_xform       = scene.instance_transform_buffer.read(inst_id);

#if NT_ENABLE_PROCEDURAL
                Bool is_proc_sp  = ((vis.y >> 29u) & 1u) > 0u;
                // $if (not ite): ite evaluates both arms, making every non-glass
                // pixel pay reconstruct_object_position's bindless tri+vertex reads.
                Float3 world_pos = ray->origin() + ray->direction() * depth;
                $if(is_glass_sp & !is_proc_sp) {
                    world_pos = (sp_xform * make_float4(reconstruct_object_position(vertex_bindless, inst_data.z, inst_data.w, prim_id_sp, bary_sp), 1.0f)).xyz();
                };

                // Resolve surface - branch for procedural vs mesh. Declare `surface`
                // ONCE here and assign in both branches (see temporal pass for the
                // shadowing-bug rationale).
                SurfaceData surface;
                $if(is_proc_sp) {
                    // Recompute bary at unjittered pixel center for stable texture sampling.
                    auto sp_ray_unjit = camera->generate_ray(Expr{
                        (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter
                    });
                    $if(!is_glass_sp) {
                        bary_sp = reconstruct_unjittered_bary_procedural(
                            proc_bindless, inst_id, prim_id_sp, bary_sp,
                            sp_ray_unjit->origin(), sp_ray_unjit->direction());
                    };
                    surface = resolve_procedural_surface_textured(
                        resolver, proc_bindless, tex_bindless,
                        inst_id, prim_id_sp, world_pos, wo,
                        scene.material_buffer,
                        bary_sp,
                        0.0f,
                        Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
                        resolution.x, resolution.y);
                }
                $else {
                    surface = resolve_surface_from_instance(
                        resolver, vertex_bindless, tex_bindless,
                        inst_data, prim_id_sp, bary_sp,
                        scene.material_buffer, wo,
                        sp_xform,
                        0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) }, resolution.x, resolution.y,
                        inst_id);
                };
#else
                Float3 world_pos = ray->origin() + ray->direction() * depth;
                $if(is_glass_sp) {
                    world_pos = (sp_xform * make_float4(reconstruct_object_position(vertex_bindless, inst_data.z, inst_data.w, prim_id_sp, bary_sp), 1.0f)).xyz();
                };
                SurfaceData surface = resolve_surface_from_instance(
                    resolver, vertex_bindless, tex_bindless,
                    inst_data, prim_id_sp, bary_sp,
                    scene.material_buffer, wo,
                    sp_xform,
                    0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) }, resolution.x, resolution.y,
                    inst_id);
#endif

                // Blend-rolled-opaque reclass (see Shading.h).
                reclass_blend_rolled_opaque(surface, is_glass_sp);

                // Hoist bsdf_mis for spatial reuse p_hat (single construction per pixel).
                // Probe-covered SSS pixels zero the HK lobe (direct model = the
                // Burley probe; current pixel is point-gated above). Thin-wall
                // subsurface (diffuse_trans > 0) keeps the lobes.
                Bool sss_probe_direct = Expr{ surface.bsdf_type == 6u }
                                      & Expr{ surface.flatness > 0.0f }
                                      & Expr{ surface.diffuse_trans <= 0.0f };
                MaterialBSDF bsdf_mis = surface.make_bsdf(sss_probe_direct);

                Float3 ns       = surface.ns;
                Float3 tangent  = surface.tangent;
                Float  tangent_w= surface.tangent_w;

                // Glass attenuation for p_hat at current pixel
                Float4 glass_tp = glass_throughput.read(coord);
                Float glass_att = luminance(glass_tp.xyz());

                //==================================================================
                // CGNS — RTXDI 3.1 compatibility-guided neighbor selection
                // (docs/rtxdi31_cgns_decorrelation_plan.md; ref
                // Rtxdi/PT/SpatialNeighborSelection.hlsli). Score disk-sampled
                // candidates by pow(dot(N0,Nc), 8) * exp(-|P0-Pc|/sigma),
                // sigma = sqrt(0.05 * depth0^2 / pi), and keep K picks via K
                // parallel weighted-reservoir samplers over the same stream.
                // The picks replace the golden-angle spiral as the neighbor
                // offset source for BOTH the merge loop and the piSum pass,
                // so the two-pass BASIC correction stays consistent by
                // construction (same in-register array, no re-walk).
                // Inline selection instead of RTXDI's separate pass + packed
                // buffer: scoring needs only texture reads, and the register
                // handoff costs no VRAM, dispatch, or barrier.
                // Candidate normals come from the prefilter's world-normal
                // buffer (written before the DI reuse passes each frame;
                // w=255 sky sentinel). Engine extension: a material-sim key
                // gate (buffer already bound, previously read only by PassGI)
                // rejects dissimilar materials before they enter the WRS
                // stream, so picks never need re-vetting at merge time.
                // Compiled out entirely when _bakedCgnsEnabled == 0.
                //==================================================================
                const uint cgnsEnabled = _bakedCgnsEnabled;
                Local<int>  cgns_sel_dx{kCgnsNeighborCount};
                Local<int>  cgns_sel_dy{kCgnsNeighborCount};
                Local<uint> cgns_sel_valid{kCgnsNeighborCount};
                Local<float> cgns_sel_wsum{kCgnsNeighborCount};
                // RTXDI: neighborCount = min(numSpatialSamples, MAX). The
                // existing count knobs still control how many of the K slots
                // merge; the disocclusion-boost conditional does not apply
                // (scored picks already serve its purpose).
                UInt cgns_neighbor_count = def(cast<UInt>(kCgnsNeighborCount));
                if (cgnsEnabled != 0u) {
                    cgns_neighbor_count = min(
                        p.spatialNeighborCount + p.disocclusionBoostSamples,
                        cast<UInt>(kCgnsNeighborCount));

                    Float3 cgns_center_key = scene.sim_key_buffer.read(inst_data.y & 0xFFu);
                    for (uint k = 0u; k < kCgnsNeighborCount; ++k) {
                        cgns_sel_valid[k] = 0u;
                        cgns_sel_wsum[k]  = 0.0f;
                        cgns_sel_dx[k]    = 0;
                        cgns_sel_dy[k]    = 0;
                    }
                    UInt cgns_good = def(0u);

                    // Candidate bound from the params buffer (runtime) so DXC
                    // cannot unroll this loop — same rule as the merge loop.
                    $for(c, 0u, p.cgnsCandidateCount) {
                        $if(cgns_good < cgns_neighbor_count) {
                            Float cg_u1 = util::uniform_uint_to_float(
                                util::xxhash32(make_uint3(pixel_index, frame_count, c * 2u + 900u)));
                            Float cg_u2 = util::uniform_uint_to_float(
                                util::xxhash32(make_uint3(pixel_index, frame_count, c * 2u + 901u)));
                            // Uniform disk sample (ref lines 81-85).
                            Float cg_ang = cg_u1 * 6.28318530718f;
                            Float cg_rad = luisa::compute::sqrt(cg_u2) * p.cgnsRadius;
                            Int cg_cx = cast<Int>(coord.x) + cast<int>(luisa::compute::cos(cg_ang) * cg_rad);
                            Int cg_cy = cast<Int>(coord.y) + cast<int>(luisa::compute::sin(cg_ang) * cg_rad);
                            if (_checkerboard) {
                                // Snap to the active field's x parity (active
                                // pixels satisfy (x + y + cbField) even) so the
                                // compacted reservoir index and the G-buffer
                                // reads describe the same pixel.
                                cg_cx = cg_cx + ((cg_cx + cg_cy + cast<Int>(cbField)) & 1);
                            }

                            $if(Expr{ cg_cx >= 0 & cg_cx < cast<Int>(resolution.x)
                                    & cg_cy >= 0 & cg_cy < cast<Int>(resolution.y) }) {
                                $if(!Expr{ cast<UInt>(cg_cx) == coord.x & cast<UInt>(cg_cy) == coord.y }) {
                                    UInt2  cg_nc  = make_uint2(cast<uint>(cg_cx), cast<uint>(cg_cy));
                                    Float4 cg_nr  = denoise_normal.read(cg_nc);
                                    // w packs matID (floor) + roughness (fract);
                                    // 255.0 = prefilter's sky sentinel.
                                    $if(Expr{ cg_nr.w != 255.0f }) {
                                        Float  cg_nd  = gbuf_depth.read(cg_nc).x;
                                        auto   cg_ray = camera->generate_ray(Expr{
                                            (make_float2(cg_nc) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f });
                                        Float3 cg_pos = cg_ray->origin() + cg_ray->direction() * cg_nd;
                                        Float3 cg_n   = normalize(cg_nr.xyz());

                                        // RTXDI NeighborCompatibilityScore (ref lines 39-46).
                                        Float cg_nsim = luisa::compute::pow(
                                            luisa::compute::max(luisa::compute::dot(ns, cg_n), 0.0f),
                                            p.cgnsNormalExponent);
                                        Float cg_sigma = luisa::compute::sqrt(
                                            p.cgnsPosOmega * depth * depth * 0.31830988618f);
                                        Float cg_psim = luisa::compute::exp(
                                            -luisa::compute::distance(world_pos, cg_pos)
                                            / luisa::compute::max(cg_sigma, 1e-6f));
                                        Float cg_score = cg_nsim * cg_psim;

                                        // Material gate (engine extension): the
                                        // inst/sim-key reads sit behind the score
                                        // test so only viable candidates pay them.
                                        Bool cg_mat_ok = def(true);
                                        $if(p.cgnsMaterialGate != 0u) {
                                            UInt4  cg_vis_tp  = gbuf_vis.read(cg_nc);
                                            UInt4  cg_inst_tp = scene.instance_buffer.read(cg_vis_tp.x);
                                            Float3 cg_key     = scene.sim_key_buffer.read(cg_inst_tp.y & 0xFFu);
                                            cg_mat_ok = render::are_materials_similar_keys(
                                                cgns_center_key, cg_key,
                                                p.cgnsMatSimRoughness, p.cgnsMatSimF0, p.cgnsMatSimAlbedo);
                                        };

                                        $if(Expr{ cg_score > 0.0f } & cg_mat_ok) {
                                            $if(cg_score >= p.cgnsGoodScore) {
                                                cgns_good = cgns_good + 1u;
                                            };
                                            // K parallel WRS over the same stream
                                            // (ref lines 107-114). Static k loop —
                                            // no dynamic array indexing here.
                                            for (uint k = 0u; k < kCgnsNeighborCount; ++k) {
                                                cgns_sel_wsum[k] = cgns_sel_wsum[k] + cg_score;
                                                Float cg_uk = util::uniform_uint_to_float(
                                                    util::xxhash32(make_uint3(pixel_index, frame_count, c * 8u + 910u + k)));
                                                $if(cg_uk < cg_score / cgns_sel_wsum[k]) {
                                                    cgns_sel_dx[k]    = cg_cx - cast<Int>(coord.x);
                                                    cgns_sel_dy[k]    = cg_cy - cast<Int>(coord.y);
                                                    cgns_sel_valid[k] = 1u;
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
                }

                // BASIC bias correction state (RTXDI SpatialResampling.hlsli:241-303;
                // GI PassGI.cpp:1298-1380 template). input_M_s captured BEFORE the loop
                // = pre-merge canonical M (RTXDI curSample.M). cached_mask_s bit n marks
                // neighbors that passed all gates + merged in the 1st pass — drives 2nd
                // pass piSum walk.
                UInt  input_M_s           = r->M();
                UInt  cached_mask_s       = def(0u);
                UInt  merged_M_s          = input_M_s;  // accumulates merged neighbor M (pre-cap)
                Int   selected_neighbor_s = def(-1);   // -1 = canonical won

                auto try_spatial = [&](Int dx, Int dy, UInt neighbor_idx) noexcept {
                    Int nc_x = cast<Int>(coord.x) + dx;
                    Int nc_y = cast<Int>(coord.y) + dy;

                    $if(Expr{ nc_x >= 0 & nc_x < cast<Int>(resolution.x)
                            & nc_y >= 0 & nc_y < cast<Int>(resolution.y) }) {
                        UInt2 nc_uint = make_uint2(cast<uint>(nc_x), cast<uint>(nc_y));
                        UInt  n_pixel = cast<uint>(nc_y) * resolution.x + cast<uint>(nc_x);
                        if (_checkerboard)
                            n_pixel = cast<uint>(nc_y) * rsv_res.x + (cast<uint>(nc_x) >> 1u);

                        Float n_depth = gbuf_depth.read(nc_uint).x;
                        UInt4 n_vis = gbuf_vis.read(nc_uint);

                        $if(Expr{ (n_vis.x == inst_id) & (abs(n_depth - depth) / max(depth, 0.01f) < p.spatialDepthThresh) }) {
                            UInt4  n_inst_data  = scene.instance_buffer.read(n_vis.x);
                            Float2 n_bary       = gbuf_bary_motion.read(nc_uint).xy();
                            Float3 n_ns;
#if NT_ENABLE_PROCEDURAL
                            Bool n_is_proc      = ((n_vis.y >> 29u) & 1u) > 0u;
                            $if(n_is_proc) {
                                auto n_ray_np = camera->generate_ray(Expr{
                                    (make_float2(nc_uint) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f
                                });
                                Float3 n_pos_np = n_ray_np->origin() + n_ray_np->direction() * n_depth;
                                n_ns = reconstruct_procedural_normal(
                                    proc_bindless,
                                    n_vis.x, n_vis.y & 0x3FFFFFFFu, n_pos_np, make_float2(0.5f));
                            }
                            $else {
#endif
                                n_ns = transform_normal(scene.instance_transform_buffer.read(n_vis.x),
                                    reconstruct_normal(vertex_bindless,
                                    n_inst_data.z, n_inst_data.w, n_vis.y & 0x3FFFFFFFu, n_bary));
#if NT_ENABLE_PROCEDURAL
                            };
#endif
                            Float normal_dot    = luisa::compute::dot(ns, n_ns);

                            $if(normal_dot > p.spatialNormalThresh) {
                                Var<Reservoir> nr = reservoir_input.read(n_pixel);

                                // RTXDI ref (SpatialResampling.hlsli:98) only requires M > 0;
                                // the previous M > 2u gate over-culled low-M neighbors on narrow-lobe
                                // metal where temporal MFactor rejection keeps M in the 1-2 range.
                                $if(nr->is_valid() & nr->light_idx != ~0u & nr->target_pdf > p.targetPdfFloor & nr->M() > 0u) {
                                    // Fetch neighbor's sample once (different per neighbor —
                                    // cannot hoist). Both p_hat_new below and downstream logic
                                    // reuse this single fetch instead of duplicating reads.
                                    LightSample neighbor_sample = fetch_light_sample(
                                        Expr{ nr->light_idx == kEnvLightSentinel },
                                        nr->light_idx, nr->light_bary_u, nr->light_bary_v,
                                        lights.triangle_lights, lights.triangle_vertices,
                                        env.envmap, env.env_width, env.env_height,
                                        env.env_rotation, env_exp);

                                    // p_hat: neighbor's sample at current surface (no bindless reads)
                                    Float p_hat_new = evaluate_p_hat_from_sample(
                                        bsdf_mis,
                                        neighbor_sample,
                                        world_pos, ns, wo);
                                    p_hat_new = p_hat_new * glass_att;

                                    $if(p_hat_new > 0.0f) {
                                        // RTXDI MFactor (ref: Rtxdi/Utils/Math.hlsli:117, PairwiseStreaming.hlsli:58).
                                        // Replaces the soft log2(jac)/4 fade (~pow(jac, 1.44)) that admitted
                                        // ~128x more leakage than RTXDI's pow(min(q1/q0,1), N). On rotation,
                                        // the BRDF lobe sweeps across bright env features and stale neighbor
                                        // reservoirs fall in the 0.3-0.7 PDF-ratio band where the old fade
                                        // was too soft. Single-direction: q0 = neighbor PDF at neighbor
                                        // surface, q1 = neighbor PDF at canonical surface. Symmetric
                                        // MFactor(w_c@n, w_c@c) is Phase 2 (needs full neighbor surface).
                                        // Skip MFactor when canonical's own target_pdf is at the floor
                                        // (degenerate surface, e.g. glass with ~0 throughput) — ratio
                                        // would saturate min(·,1)=1 and falsely accept.
                                        Float m_factor = ite(r->target_pdf > p.targetPdfFloor * 1000.0f,
                                            rtxdi_mfactor(nr->target_pdf, p_hat_new, p.mFactorExponent),
                                            1.0f);

                                        $if(m_factor > p.mFactorThreshold) {
                                            // Drop the lower clamp (1/3): MFactor already zero-outs bad ratios.
                                            // Keep the upper clamp: MFactor saturates at 1.0 for jac > 1 but
                                            // jacobian itself can still explode a neighbor's w_sum contribution.
                                            Float jacobian     = min(p_hat_new / luisa::compute::max(nr->target_pdf, p.targetPdfFloor), 3.0f);
                                            Float w_neighbor   = nr->w_sum * jacobian * m_factor;
                                            Float w_sum_merged = r->w_sum + w_neighbor;
                                            Float threshold    = ite(w_sum_merged > 0.0f,
                                                w_neighbor / w_sum_merged, 0.0f);
                                            Float u = util::uniform_uint_to_float(
                                                util::xxhash32(make_uint3(pixel_index, frame_count, neighbor_idx)));

                                            // Track BASIC state: this neighbor passed all gates + merged.
                                            cached_mask_s = cached_mask_s | (1u << neighbor_idx);
                                            merged_M_s    = merged_M_s + nr->M();

                                            $if(u < threshold) {
                                                r->light_idx    = nr->light_idx;
                                                r->target_pdf   = p_hat_new;
                                                r->light_bary_u = nr->light_bary_u;
                                                r->light_bary_v = nr->light_bary_v;
                                                // Do NOT inherit neighbor's visibility — shadow
                                                // confirmation was for a different receiver point.
                                                // High-curvature silhouettes (glass sphere) and
                                                // opaque occluder boundaries let instance/depth/
                                                // normal checks pass while actual shadow geometry
                                                // differs, leaking vis=1 into shadowed pixels and
                                                // making the shade pass skip the trace. Force a
                                                // fresh trace (matches GI spatial reuse behavior).
                                                r->set_visibility(0u);
                                                r->set_vis_age(0u);
                                                r->set_spatial_dist_x(0);
                                                r->set_spatial_dist_y(0);
                                                selected_neighbor_s = cast<Int>(neighbor_idx);
                                            };
                                            UInt M_new    = r->M() + nr->M();
                                            UInt M_capped = min(M_new, p.spatialMaxM);
                                            Float w_scale = cast<float>(M_capped) / cast<float>(max(M_new, 1u));
                                            r->w_sum = luisa::compute::min(w_sum_merged * w_scale, p.wSumCap);
                                            r->set_M(M_capped);
                                        };
                                    };
                                };
                            };
                        };
                    };
                };

                // Neighbor counts come from the params buffer (runtime) so DXC
                // can't statically determine the loop bound below — preventing
                // the unrolling that blew this shader's compile time up to 55s.
                // The spiral-offset Constant is sized for the compile-time max
                // (kSpatialNeighborCount + kDisocclusionBoostSamples), so any
                // runtime value <= that sum reads valid offsets.
                const UInt spatial_count     = p.spatialNeighborCount;
                const uint spatial_radius    = _spatialRadius;
                const UInt disocclusion_boost = p.disocclusionBoostSamples;
                const UInt total_neighbors   = spatial_count + disocclusion_boost;

                // Per-pixel random rotation of the golden-angle spiral. Without this,
                // every pixel samples the same 16 deterministic (dx, dy) offsets — when
                // the central reservoir's selection is similar across a region (which
                // happens whenever the BRDF/preferred direction is similar, e.g. on a
                // flat floor), streaming RIS propagates the same directional shift
                // across the screen and produces a coherent "flowing water" pattern in
                // DI shadows. RTXDI randomizes the starting index into a precomputed
                // neighbor-offset buffer for the same reason (SpatialResampling.hlsli:57).
                // Rotating the spiral preserves the radial distribution (each ring
                // still has the same radius) while decorrelating the angular positions
                // between adjacent pixels.
                // Precompute golden-angle spiral offsets at kernel-trace time.
                // The base (dx, dy) per neighbor depends only on n and the max
                // neighbor count — fixed for this kernel compilation — so baking
                // them into a Constant eliminates N × (sqrt + cos + sin) per
                // pixel. Only the per-pixel rotation (cos_r, sin_r) stays
                // runtime. Sized for the compile-time max so any runtime
                // total_neighbors <= max reads valid offsets.
                constexpr uint max_neighbors_c = kSpatialNeighborCount + kDisocclusionBoostSamples;
                constexpr auto golden_angle_c = 2.39996322972865f;
                luisa::vector<luisa::float2> spiral_offsets_host;
                spiral_offsets_host.reserve(max_neighbors_c);
                for (uint host_n = 0u; host_n < max_neighbors_c; ++host_n) {
                    float r_frac = std::sqrt((static_cast<float>(host_n) + 1.0f) /
                                             static_cast<float>(max_neighbors_c));
                    float theta  = static_cast<float>(host_n) * golden_angle_c;
                    float r_disk = r_frac * static_cast<float>(spatial_radius);
                    spiral_offsets_host.emplace_back(
                        r_disk * std::cos(theta), r_disk * std::sin(theta));
                }
                Constant<float2> spiral_offsets = spiral_offsets_host;

                UInt rot_seed = util::xxhash32(make_uint2(pixel_index, frame_count));
                Float rot_angle = util::uniform_uint_to_float(rot_seed) * 6.28318530718f;
                Float cos_r = luisa::compute::cos(rot_angle);
                Float sin_r = luisa::compute::sin(rot_angle);

                // Neighbor loop. CGNS mode iterates the WRS pick array (each
                // slot already compatibility-scored; the disocclusion-boost
                // conditional does not apply — see the CGNS block above);
                // the legacy path is bit-identical to the former spiral-only
                // loop (merge_bound folds to total_neighbors).
                UInt merge_bound = def(total_neighbors);
                if (cgnsEnabled != 0u) {
                    merge_bound = cgns_neighbor_count;
                }

                $for(n, 0u, merge_bound) {
                    if (cgnsEnabled != 0u) {
                        $if(cgns_sel_valid[n] == 1u) {
                            try_spatial(cgns_sel_dx[n], cgns_sel_dy[n], cast<UInt>(n));
                        };
                    } else {
                    // Disocclusion boost: extra neighbors only when M < target history length
                    //Bool is_boost = cast<UInt>(n) >= spatial_count;
                    //Bool below_target = r->M() < _temporalMaxM;
                    Bool should_process = ite(Expr{ n >= spatial_count },
                        Expr{ r->M() < p.temporalMaxM }, true);

                    $if(should_process) {
                        Float2 base = spiral_offsets.read(n);
                        Int dx = cast<Int>(base.x * cos_r - base.y * sin_r);
                        Int dy = cast<Int>(base.x * sin_r + base.y * cos_r);

                        try_spatial(dx, dy, cast<UInt>(n));
                    };
                    };
                };

                // BASIC bias correction (RTXDI SpatialResampling.hlsli:241-303;
                // GI PassGI.cpp:1298-1380 template). 2nd-pass piSum: re-resolve each
                // cached merged neighbor's surface, eval winner's pdf at each, sum.
                // r->target_pdf post-merge = winner @ center surface (state.targetPdf).
                // Gate: cached_mask_s != 0 (any neighbor merged, regardless of who won).
                if (diBiasCorrectionEnabled != 0u) {
                // P1-6: light_idx != ~0u added — by construction a committed
                // merge implies adoption (threshold 1 for an empty center), but
                // a degenerate w_sum_merged == 0 leaves the center sampleless
                // while cached_mask_s is set; the fetch below would read
                // triangle_lights[~0u] (mirrors ref SpatialResampling.hlsli:239).
                $if(cached_mask_s != 0u & r->light_idx != ~0u) {
                    Float piSum_s = r->target_pdf * cast<Float>(input_M_s);
                    Float pi_s    = r->target_pdf;  // default: canonical won

                    // Cache winner's sample once (post-merge r = winner).
                    Bool winner_is_env = Expr{ r->light_idx == kEnvLightSentinel };
                    LightSample winner_sample = fetch_light_sample(
                        winner_is_env,
                        r->light_idx, r->light_bary_u, r->light_bary_v,
                        lights.triangle_lights, lights.triangle_vertices,
                        env.envmap, env.env_width, env.env_height,
                        env.env_rotation, env_exp);

                    $for(n, 0u, merge_bound) {
                        UInt bit_n = 1u << cast<UInt>(n);
                        $if((cached_mask_s & bit_n) != 0u) {
                            // Neighbor offset for slot n: the CGNS pick array
                            // (slots identical to the merge loop — no re-walk
                            // needed), or the SAME spiral coord computation as
                            // the legacy 1st pass.
                            Int dx_n = def(0);
                            Int dy_n = def(0);
                            if (cgnsEnabled != 0u) {
                                dx_n = cgns_sel_dx[n];
                                dy_n = cgns_sel_dy[n];
                            } else {
                                Float2 base_n = spiral_offsets.read(n);
                                dx_n = cast<Int>(base_n.x * cos_r - base_n.y * sin_r);
                                dy_n = cast<Int>(base_n.x * sin_r + base_n.y * cos_r);
                            }
                            Int nc_x_n = cast<Int>(coord.x) + dx_n;
                            Int nc_y_n = cast<Int>(coord.y) + dy_n;
                            UInt2 nc_uint_n = make_uint2(cast<uint>(nc_x_n), cast<uint>(nc_y_n));
                            UInt  n_pixel_n = cast<uint>(nc_y_n) * resolution.x + cast<uint>(nc_x_n);
                            if (_checkerboard)
                                n_pixel_n = cast<uint>(nc_y_n) * rsv_res.x + (cast<uint>(nc_x_n) >> 1u);

                            // Resolve neighbor n surface (mirror GI PassGI.cpp:1333-1346).
                            Float  n_depth_n = gbuf_depth.read(nc_uint_n).x;
                            UInt4  n_vis_n = gbuf_vis.read(nc_uint_n);
                            Float2 n_bary_n = gbuf_bary_motion.read(nc_uint_n).xy();
                            Float2 n_ndc_n = (make_float2(nc_uint_n) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
                            auto   n_ray_n = camera->generate_ray(n_ndc_n);
                            Float3 n_world_pos_n = n_ray_n->origin() + n_ray_n->direction() * n_depth_n;
                            Float3 n_wo_n = -n_ray_n->direction();
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
                                    0.0f, n_screen_uv_n, resolution.x, resolution.y,
                                    n_vis_n.x);
#if NT_ENABLE_PROCEDURAL
                            };
#endif
                            // Blend-rolled-opaque reclass (see Shading.h) — neighbor glass bit.
                            reclass_blend_rolled_opaque(
                                n_surface_n, Expr{ (n_vis_n.y >> 31u) > 0u });

                            // Winner's light at this neighbor's surface.
                            // Probe-covered SSS zeroing applies too; a
                            // point-primitive neighbor keeps the HK lobe.
                            // Thin-wall neighbor pixels keep the lobes.
                            Bool n_sss_probe_n = Expr{ n_surface_n.bsdf_type == 6u }
                                               & Expr{ n_surface_n.flatness > 0.0f }
                                               & Expr{ n_surface_n.diffuse_trans <= 0.0f }
                                               & !Expr{ ((n_vis_n.y >> 30u) & 1u) > 0u };
                            MaterialBSDF n_bsdf_n = n_surface_n.make_bsdf(n_sss_probe_n);
                            Float ps_n = evaluate_p_hat_from_sample(
                                n_bsdf_n,
                                winner_sample,
                                n_world_pos_n, n_surface_n.ns, n_wo_n);

                            Var<Reservoir> nr_n = reservoir_input.read(n_pixel_n);
                            piSum_s = piSum_s + ps_n * cast<Float>(nr_n->M());
                            $if(selected_neighbor_s == cast<Int>(n)) {
                                pi_s = ps_n;
                            };
                        };
                    };

                    $if(piSum_s > 1e-10f) {
                        // DI streaming-form BASIC multiplier: pi · M_merged / piSum
                        // (see temporal BASIC block for derivation). merged_M_s is
                        // pre-cap M across input + all merged neighbors.
                        r->w_sum = luisa::compute::min(
                            r->w_sum * pi_s * cast<Float>(merged_M_s) / piSum_s, p.wSumCap);
                    };
                };
                } // if (diBiasCorrectionEnabled) — compile-time
            };

            // P1-7: unconditional pass-through write. The write used to sit
            // inside the validity branch, so centers that were invalid (sky,
            // point primitives, empty reservoirs) never wrote their output
            // slot — ping-pong consumers could read a 2-frame-stale
            // reservoir there. Every non-OOB pixel now writes its (possibly
            // unchanged, possibly empty) r every frame.
            reservoir_output.write(pixel_index, r);
        });
    }

    //==========================================================================
    // Image Creation
    //==========================================================================
    void PassDI::createImages(luisa::compute::Device& device, uint width, uint height) {
        // When checkerboard is enabled, reservoirs store only active pixels at half width
        uint rsv_w = _checkerboard ? (width + 1u) / 2u : width;
        uint pixel_count = rsv_w * height;
        _resBuf[0] = device.create_buffer<Reservoir>(pixel_count);
        _resBuf[1] = device.create_buffer<Reservoir>(pixel_count);

        _diParamsBuf         = device.create_buffer<DIParams>(1u);
        // Presample pools are NOT reallocated here: fixed-size and
        // resolution-independent since Phase 2 (allocated once in
        // compileImpl, survive resizes — their contents stay valid samples
        // of an unchanged light distribution).
    }

    void PassDI::zeroInitBuffers() {
        // Both ping-pong slots: recreated buffers contain recycled-heap bytes,
        // and is_valid()==M()>0 treats garbage as a live reservoir with a wild
        // light index / sample position (resize-TDR root cause 2026-09-16).
        if (!_resBuf[0] || !_zeroReservoirShader) return;
        auto& stream = Renderer::stream();
        auto cl = luisa::compute::CommandList::create();
        for (auto& buf : _resBuf) {
            auto count = static_cast<luisa::uint>(buf.size());
            cl << _zeroReservoirShader(buf, count)
                  .dispatch((count + 255u) / 256u);
        }
        stream << cl.commit();
    }

    //==========================================================================
    // PassDI::_populateDiParams
    //==========================================================================
    void PassDI::_populateDiParams(luisa::compute::CommandList& cmdlist) noexcept {
        // FPS-aware: when auto-derivation is enabled, derive visMaxAge from
        // _visAgeAccumTimeSec. Visibility reuse is much shorter than the GI
        // accumulation it tracks by design (RTXDI: ~1/15 ratio), with a floor of 1.
        if (_visAgeAutoDerived) {
            float fps = util::clampSmoothedFps(
                ci::app::getWindow()->getApp()->getAverageFps());
            uint32_t accumFrames = util::computeAccumulatedFrames(_visAgeAccumTimeSec, fps);
            // Floor of 4 = RTXDI finalVisibilityMaxAge: below that, shade
            // re-traces visibility almost every frame (with 0.5s accumulation
            // the raw accumFrames/15 ratio yields 2) — pure ray cost, no gain.
            _visMaxAge = std::clamp(accumFrames / 15u, 4u, 15u);
        }
        // vis_age is a 4-bit field (0-15): visMaxAge = 16 wraps to 0 in
        // set_vis_age and the retest gate `vis_age() < visMaxAge` never trips.
        _visMaxAge = std::min(_visMaxAge, 15u);
        _diParamsCpu.temporalMaxM                = _temporalMaxM;
        _diParamsCpu.spatialMaxM                 = _spatialMaxM;
        _diParamsCpu.visMaxAge                   = _visMaxAge;
        _diParamsCpu.spatialRadius               = _spatialRadius;
        _diParamsCpu.spatialNormalThresh         = _spatialNormalThresh;
        _diParamsCpu.spatialDepthThresh          = _spatialDepthThresh;
        _diParamsCpu.wSumCap                     = _wSumCap;
        _diParamsCpu.targetPdfFloor              = _targetPdfFloor;
        _diParamsCpu.brdfCutoff                  = _brdfCutoff;
        _diParamsCpu.brdfCandidateRoughnessCutoff = _brdfCandidateRoughnessCutoff;
        _diParamsCpu.mFactorExponent             = _mFactorExponent;
        _diParamsCpu.mFactorThreshold            = _mFactorThreshold;
        // Populate runtime neighbor counts from the compile-time constants.
        // Keeps DXC from unrolling the spatial loops while preserving the
        // configured neighbor count at runtime.
        _diParamsCpu.spatialNeighborCount        = kSpatialNeighborCount;
        _diParamsCpu.disocclusionBoostSamples    = kDisocclusionBoostSamples;
        _diParamsCpu.localLightCandidateCount    = _localLightCandidateCount;
        // CGNS (selection compiled out when _bakedCgnsEnabled == 0; the
        // params stay populated so a UI toggle takes effect on recompile).
        _diParamsCpu.cgnsCandidateCount          = _cgnsCandidateCount;
        _diParamsCpu.cgnsRadius                  = _cgnsRadius;
        _diParamsCpu.cgnsNormalExponent          = _cgnsNormalExponent;
        _diParamsCpu.cgnsPosOmega                = _cgnsPosOmega;
        _diParamsCpu.cgnsGoodScore               = _cgnsGoodScore;
        _diParamsCpu.cgnsMaterialGate            = _cgnsMaterialGate ? 1u : 0u;
        _diParamsCpu.cgnsMatSimRoughness         = _cgnsMatSimRoughness;
        _diParamsCpu.cgnsMatSimF0                = _cgnsMatSimF0;
        _diParamsCpu.cgnsMatSimAlbedo            = _cgnsMatSimAlbedo;
        cmdlist << _diParamsBuf.copy_from(&_diParamsCpu);
    }

    //==========================================================================
    // Render Methods
    //==========================================================================
    void PassDI::renderGBuffer(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        cmdlist << _gbufShader(
            ctx.gbufDepth,
            ctx.gbufVis,
            ctx.gbufBaryMotion,
            ctx.glassThroughput,
            ctx.gbufVelocity,
            ctx.gbufDepthUpscale,
            ctx.camera,
            _geom->tlas(),
            SceneGeometryResources{ _geom->instance_buffer(), _geom->instance_transform_buffer(), _geom->instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() },
            _geom->vertex_bindless(),
            ctx.materialPool.textures(),
            kMaxGlassBounces   // 11: runtime PSR loop bound (perf review R2 item 8)
#if NT_ENABLE_PROCEDURAL
            , *_procBindlessPtr
#endif
#if NT_ENABLE_SHARC
            ,
            ctx.roughGlassInfo
#endif
        ).dispatch(ctx.width, ctx.height);
    }

    void PassDI::renderPresampleLocal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        auto& ls = ctx.lightSampler;
        uint emissive_count = ls.emissive_triangle_count();
        // Dirty-gated: the tiles stay valid samples of an unchanged light
        // distribution (alias table, emissive count, triangle areas). Pipeline
        // marks dirty on rebuild/update_weights/light-transform changes.
        if (emissive_count == 0u || !_presampleLocalDirty) return;

        cmdlist << _presampleLocalShader(
            _presampleLocalTiles,       // 0
            ctx.frameCount,             // 1
            LightSamplingResources{ ls.triangle_buffer(), ls.vertex_buffer(),
                ls.alias_table(), ls.emissive_triangle_count(),
                ls.total_power_inv(), ls.emissive_count_inv(),
                ls.instance_to_light_base() }   // 2
        ).dispatch(kPresampleTileSize, kPresamplePoolTileCount);
        _presampleLocalDirty = false;
    }

    void PassDI::renderPresampleEnv(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        auto& ls = ctx.lightSampler;
        if (!ls.has_environment()) return;
        // Dirty-gated: candidates store rotation/CDF-dependent uv directions
        // and pdfs. Pipeline marks dirty on env rotation uploads and
        // envmap/CDF regeneration (procedural sky, rebuild).
        if (!_presampleEnvDirty) return;

        cmdlist << _presampleEnvShader(
            _presampleEnvTiles,         // 0
            ctx.frameCount,             // 1
            EnvLightResources{ ls.envmap_image(), ls.env_cdf_marginal(),
                ls.env_cdf_conditional(), ls.env_integral(),
                ls.env_width(), ls.env_height(),
                ls.env_rotation_matrix() }  // 2
        ).dispatch(kPresampleEnvTileSize, kPresamplePoolTileCount);
        _presampleEnvDirty = false;
    }

    void PassDI::renderCandidate(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        _populateDiParams(cmdlist);
        auto& ls = ctx.lightSampler;
        auto& geom = ctx.geometry;

        cmdlist << _candidateShader(
            _diParamsBuf,                   // 0: di_params
            reservoirBuffer(),              // 1: reservoir_buffer
            ctx.gbufDepth,                  // 1
            ctx.gbufVis,                    // 2
            ctx.gbufBaryMotion,             // 3
            ctx.seedImage,                  // 4
            geom.tlas(),                    // 5
            ctx.camera,                     // 6
            SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() }, // 7
            geom.vertex_bindless(),         // 8
            ctx.materialPool.textures(),    // 9: tex_bindless
            LightSamplingResources{ ls.triangle_buffer(), ls.vertex_buffer(),
                ls.alias_table(), ls.emissive_triangle_count(),
                ls.total_power_inv(), ls.emissive_count_inv(),
                ls.instance_to_light_base() },  // 10
            EnvLightResources{ ls.envmap_image(), ls.env_cdf_marginal(),
                ls.env_cdf_conditional(), ls.env_integral(),
                ls.env_width(), ls.env_height(),
                ls.env_rotation_matrix() },  // 11
            ls.env_exposure(),              // 12
            kEnvCandidateCount,             // 13
            _presampleLocalTiles,           // 14
            _presampleEnvTiles,             // 15
            ctx.frameCount,                 // 16: per-frame window re-roll (Phase 2)
            ctx.cbField,                    // 17
            ctx.glassThroughput             // 18 (hasTransparentShadowCasters is compile-time baked)
#if NT_ENABLE_PROCEDURAL
            , *_procBindlessPtr
#endif
        ).dispatch(_checkerboard ? (ctx.width + 1u) / 2u : ctx.width, ctx.height);
    }

    void PassDI::renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        auto& ls = ctx.lightSampler;
        auto& geom = ctx.geometry;

        cmdlist << _temporalReuseShader(
            _diParamsBuf,                   // 0: di_params
            reservoirBuffer(),              // 1: reservoir_buffer
            reservoirPrevBuffer(),          // 2: reservoir_prev
            ctx.gbufDepth,                  // 2
            ctx.gbufVis,                    // 3
            ctx.gbufBaryMotion,             // 4
            ctx.frameCount,                 // 5
            ctx.camera,                     // 6
            SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() }, // 7
            geom.vertex_bindless(),         // 8
            ctx.materialPool.textures(),    // 9: tex_bindless
            LightSamplingResources{ ls.triangle_buffer(), ls.vertex_buffer(),
                ls.alias_table(), ls.emissive_triangle_count(),
                ls.total_power_inv(), ls.emissive_count_inv(),
                ls.instance_to_light_base() },  // 10
            EnvLightResources{ ls.envmap_image(), ls.env_cdf_marginal(),
                ls.env_cdf_conditional(), ls.env_integral(),
                ls.env_width(), ls.env_height(),
                ls.env_rotation_matrix() },  // 11
            ls.env_exposure(),              // 12
            ctx.cbField,                    // 13
            ctx.glassThroughput,            // 14
            ctx.gbufDepthPrev,              // 15
            ctx.gbufVisPrev,                // 16
            ctx.denoiseNormalPrev,          // 17: prev-frame world normals
            _diTemporalBiasRayTraced ? 1u : 0u, // 18b: RAY_TRACED temporal correction
            geom.tlas()                     // 18c: TLAS for the visibility re-trace
            // diBiasCorrectionEnabled (18) and hasTransparentShadowCasters
            // (18d) are compile-time baked (see compileImpl).
#if NT_ENABLE_PROCEDURAL
            , *_procBindlessPtr
#endif
        ).dispatch(_checkerboard ? (ctx.width + 1u) / 2u : ctx.width, ctx.height);
    }

    void PassDI::renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength) {
        // P3-1 stats pre-pass: clear, accumulate the frame-global w_sum/M
        // average (1/256 subsample), then run the filter. Same CommandList ⇒
        // the DX backend orders them with UAV barriers.
        uint rsv_w  = _checkerboard ? (ctx.width + 1u) / 2u : ctx.width;
        uint count  = rsv_w * ctx.height;
        // P3-1 stats pre-pass: single-block reduction (no atomics — see the
        // kernel's PERF NOTE), then the filter. Same CommandList ⇒ the DX
        // backend orders them with UAV barriers.
        cmdlist << _boilingStatsDI(
            reservoirBuffer(), _boilingStatsBuf, count
        ).dispatch(1u);
        cmdlist << _boilingFilterDI(
            reservoirBuffer(),
            strength,
            ctx.cbField,
            _boilingStatsBuf
        ).dispatch(_checkerboard ? (ctx.width + 1u) / 2u : ctx.width, ctx.height);
    }

    void PassDI::renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
        auto& ls = ctx.lightSampler;
        auto& geom = ctx.geometry;

        // Spatial reads from current slot, writes to the other slot (no copy needed)
        cmdlist << _spatialReuseShader(
            _diParamsBuf,                   // 0: di_params
            reservoirPrevBuffer(),          // 1: output (write to the free slot)
            reservoirBuffer(),              // 2: input (read from current slot)
            ctx.gbufDepth,                  // 2
            ctx.gbufVis,                    // 3
            ctx.gbufBaryMotion,             // 4
            ctx.frameCount,                 // 5
            SceneGeometryResources{ geom.instance_buffer(), geom.instance_transform_buffer(), geom.instance_transform_prev_buffer(), ctx.materialPool.buffer(), ctx.materialPool.simKeyBuffer() }, // 6
            geom.vertex_bindless(),         // 7
            ctx.materialPool.textures(),    // 8: tex_bindless
            LightSamplingResources{ ls.triangle_buffer(), ls.vertex_buffer(),
                ls.alias_table(), ls.emissive_triangle_count(),
                ls.total_power_inv(), ls.emissive_count_inv(),
                ls.instance_to_light_base() },  // 9
            ctx.camera,                     // 10
            EnvLightResources{ ls.envmap_image(), ls.env_cdf_marginal(),
                ls.env_cdf_conditional(), ls.env_integral(),
                ls.env_width(), ls.env_height(),
                ls.env_rotation_matrix() },  // 11
            ls.env_exposure(),              // 12
            ctx.cbField,                    // 13
            ctx.glassThroughput,            // 14
            ctx.denoiseNormal               // 15: CGNS candidate scoring normals
#if NT_ENABLE_PROCEDURAL
            , *_procBindlessPtr
#endif
        ).dispatch(_checkerboard ? (ctx.width + 1u) / 2u : ctx.width, ctx.height);
    }

    // PassDI::flipReservoir is inline in PassDI.h

    //==========================================================================
    // UI
    //==========================================================================
    void PassDI::drawUi() {
        if (ImGui::CollapsingHeader("DI")) {
            ImGui::BeginDisabled(_visAgeAutoDerived);
            ImGui::SliderInt("Vis Reuse Age", reinterpret_cast<int*>(&_visMaxAge), 0, 15);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip(_visAgeAutoDerived
                ? "Driven by PassGI's giAccumulationTime (visMaxAge = accumFrames / 15). Disable FPS-aware GI accumulation to edit."
                : "Higher = fewer shadow rays but more lag when shadows move. 0 disables reuse.");
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = fewer shadow rays but more lag when shadows move. 0 disables reuse.");
            ImGui::SliderFloat("Vis Max Distance", &_visMaxDistance, 1.0f, 64.0f, "%.1f px");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = reuse visibility across larger pixel motion. Lower = more fresh shadow rays.");
            ImGui::SliderFloat("Env Vis Max Distance", &_envVisMaxDistance, 0.0f, 16.0f, "%.1f px");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = reuse env-light visibility across motion (risky - env occlusion flips easily). 0 = disabled.");
            ImGui::SliderInt("Disp Shadow Interfaces", reinterpret_cast<int*>(&_dispShadowInterfaces), 1, 4);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Dispersive shadow sub-walk budget: glass crossings per RGB channel after the fan-out. 2 covers a sphere entry+exit (and two stacked glasses); lower = cheaper, deeper glass stacks truncate slightly brighter.");
            ImGui::SliderFloat("Disp Shadow Split", &_dispShadowSplit, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Dispersive shadow RGB split. 1 = exact per-channel estimate (saturated primary fringes). Lower desaturates toward the d-line gray for softer hues - zero cost.");
            ImGui::SliderFloat("BRDF Cand Rough Cutoff", &_brdfCandidateRoughnessCutoff, 0.02f, 0.3f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = skip more glossy BRDF candidates (less metal noise, dimmer highlights). Lower = sharper highlights but noisier.");
            ImGui::SliderInt("Local Light Candidates", reinterpret_cast<int*>(&_localLightCandidateCount), 4, 31);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Presampled local-light candidates per pixel in the initial kernel (RTXDI Medium preset uses 8; default 15). Lower = cheaper candidate pass, noisier until temporal M growth catches up.");
            ImGui::SliderFloat("MFactor Exponent", &_mFactorExponent, 2.0f, 12.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = faster M-factor falloff (less rotation speckle on metals, possibly dimmer). Lower = more reuse (brighter, noisier).");
            ImGui::SliderFloat("MFactor Threshold", &_mFactorThreshold, 1e-4f, 1e-2f, "%.4f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = reject more low-contribution neighbors (cleaner, possibly dimmer). Lower = more reuse (brighter, noisier).");
            ImGui::Checkbox("DI Bias Correction", &_diBiasCorrectionEnabled);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = reject cross-surface fireflies via MIS normalization. OFF = brighter, more fireflies at material seams.");
            ImGui::Checkbox("Temporal Vis Re-trace", &_diTemporalBiasRayTraced);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("RAY_TRACED temporal bias correction (RTXDI Medium): re-trace the reused reservoir's visibility from the current surface. Kills occlusion-onset light leaks; costs ~1 conservative ray per stale-history pixel.");
            ImGui::SeparatorText("CGNS Spatial (RTXDI 3.1)");
            ImGui::Checkbox("Enable CGNS", &_cgnsEnabled);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Compatibility-guided neighbor selection: spatial reuse merges compatibility-scored picks (normal^8 x exp(-|dP|/sigma) WRS over a disk) instead of blind spiral offsets. Targets reuse banding on glossy surfaces. Toggling recompiles the spatial kernel at a safe point.");
            ImGui::SliderInt("CGNS Candidates", reinterpret_cast<int*>(&_cgnsCandidateCount), 8, 48);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Disk candidates scored per pixel (RTXDI 32). Early-out once enough good candidates are found; higher = better picks on busy geometry, more scoring reads.");
            ImGui::SliderFloat("CGNS Radius", &_cgnsRadius, 8.0f, 64.0f, "%.0f px");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Candidate disk radius (RTXDI 50 px). Larger = wider compatibility search, weaker guarantees at depth discontinuities.");
            ImGui::SliderFloat("CGNS Normal Exp", &_cgnsNormalExponent, 1.0f, 16.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("pow(dot(N0,Nc), exp) normal-similarity falloff (RTXDI 8). Higher = only near-coplanar candidates score.");
            ImGui::SliderFloat("CGNS Pos Omega", &_cgnsPosOmega, 0.01f, 0.2f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("World-position tolerance: sigma = sqrt(omega*depth^2/pi) (RTXDI 0.05). Higher = candidates may sit farther off-plane.");
            ImGui::SliderFloat("CGNS Good Score", &_cgnsGoodScore, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Score counted as 'good' for the early-out (RTXDI 0.5). Lower = candidates accepted sooner, less exhaustive search.");
            ImGui::Checkbox("CGNS Material Gate", &_cgnsMaterialGate);
            if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Engine extension: reject dissimilar materials (sim-key roughness/F0/albedo, same gate as GI reuse) before the selection reservoir, so picks are BRDF-compatible too.");
            if (_cgnsMaterialGate) {
                ImGui::SliderFloat("CGNS Mat Rough", &_cgnsMatSimRoughness, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("CGNS Mat F0", &_cgnsMatSimF0, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("CGNS Mat Albedo", &_cgnsMatSimAlbedo, 0.0f, 1.0f, "%.2f");
            }
        }
    }

    //==========================================================================
    // Config serialization
    //==========================================================================
    void PassDI::toJson(ci::Json& j) const {
        j = ci::Json{
            {"temporalMaxM",          _temporalMaxM},
            {"spatialMaxM",           _spatialMaxM},
            {"spatialRadius",         _spatialRadius},
            {"spatialNormalThresh",   _spatialNormalThresh},
            {"spatialDepthThresh",    _spatialDepthThresh},
            {"wSumCap",               _wSumCap},
            {"targetPdfFloor",        _targetPdfFloor},
            {"brdfCutoff",            _brdfCutoff},
            {"visMaxAge",             _visMaxAge},
            {"visMaxDistance",        _visMaxDistance},
            {"envVisMaxDistance",     _envVisMaxDistance},
            {"brdfCandidateRoughnessCutoff", _brdfCandidateRoughnessCutoff},
            {"localLightCandidateCount", _localLightCandidateCount},
            {"mFactorExponent",       _mFactorExponent},
            {"mFactorThreshold",      _mFactorThreshold},
            {"diBiasCorrectionEnabled", _diBiasCorrectionEnabled},
            {"diTemporalBiasRayTraced", _diTemporalBiasRayTraced},
            {"dispShadowInterfaces", _dispShadowInterfaces},
            {"dispShadowSplit", _dispShadowSplit},
            {"cgnsEnabled", _cgnsEnabled},
            {"cgnsCandidateCount", _cgnsCandidateCount},
            {"cgnsRadius", _cgnsRadius},
            {"cgnsNormalExponent", _cgnsNormalExponent},
            {"cgnsPosOmega", _cgnsPosOmega},
            {"cgnsGoodScore", _cgnsGoodScore},
            {"cgnsMaterialGate", _cgnsMaterialGate},
            {"cgnsMatSimRoughness", _cgnsMatSimRoughness},
            {"cgnsMatSimF0", _cgnsMatSimF0},
            {"cgnsMatSimAlbedo", _cgnsMatSimAlbedo},
        };
    }

    void PassDI::fromJson(const ci::Json& j) {
        if (!j.is_object()) return;
        _temporalMaxM          = j.value("temporalMaxM",          _temporalMaxM);
        _spatialMaxM           = j.value("spatialMaxM",           _spatialMaxM);
        _spatialRadius         = j.value("spatialRadius",         _spatialRadius);
        _spatialNormalThresh   = j.value("spatialNormalThresh",   _spatialNormalThresh);
        _spatialDepthThresh    = j.value("spatialDepthThresh",    _spatialDepthThresh);
        _wSumCap               = j.value("wSumCap",               _wSumCap);
        _targetPdfFloor        = j.value("targetPdfFloor",        _targetPdfFloor);
        _brdfCutoff            = j.value("brdfCutoff",            _brdfCutoff);
        _visMaxAge             = std::min(j.value("visMaxAge", _visMaxAge), 15u);
        _visMaxDistance        = j.value("visMaxDistance",        _visMaxDistance);
        _envVisMaxDistance     = j.value("envVisMaxDistance",     _envVisMaxDistance);
        _brdfCandidateRoughnessCutoff = j.value("brdfCandidateRoughnessCutoff", _brdfCandidateRoughnessCutoff);
        _localLightCandidateCount = std::clamp(j.value("localLightCandidateCount", _localLightCandidateCount), 4u, 31u);
        _mFactorExponent      = j.value("mFactorExponent",       _mFactorExponent);
        _mFactorThreshold     = j.value("mFactorThreshold",       _mFactorThreshold);
        _diBiasCorrectionEnabled = j.value("diBiasCorrectionEnabled", _diBiasCorrectionEnabled);
        _diTemporalBiasRayTraced = j.value("diTemporalBiasRayTraced", _diTemporalBiasRayTraced);
        _dispShadowInterfaces = std::clamp(j.value("dispShadowInterfaces", _dispShadowInterfaces), 1u, 4u);
        _dispShadowSplit      = std::clamp(j.value("dispShadowSplit", _dispShadowSplit), 0.0f, 1.0f);
        _cgnsEnabled          = j.value("cgnsEnabled", _cgnsEnabled);
        _cgnsCandidateCount   = std::clamp(j.value("cgnsCandidateCount", _cgnsCandidateCount), 8u, 48u);
        _cgnsRadius           = std::clamp(j.value("cgnsRadius", _cgnsRadius), 4.0f, 128.0f);
        _cgnsNormalExponent   = std::clamp(j.value("cgnsNormalExponent", _cgnsNormalExponent), 1.0f, 16.0f);
        _cgnsPosOmega         = std::clamp(j.value("cgnsPosOmega", _cgnsPosOmega), 0.005f, 0.5f);
        _cgnsGoodScore        = std::clamp(j.value("cgnsGoodScore", _cgnsGoodScore), 0.0f, 1.0f);
        _cgnsMaterialGate     = j.value("cgnsMaterialGate", _cgnsMaterialGate);
        _cgnsMatSimRoughness  = std::clamp(j.value("cgnsMatSimRoughness", _cgnsMatSimRoughness), 0.0f, 1.0f);
        _cgnsMatSimF0         = std::clamp(j.value("cgnsMatSimF0", _cgnsMatSimF0), 0.0f, 1.0f);
        _cgnsMatSimAlbedo     = std::clamp(j.value("cgnsMatSimAlbedo", _cgnsMatSimAlbedo), 0.0f, 1.0f);
    }

}
