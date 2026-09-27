#include "newtype/render/PassSharc.h"
#include "newtype/render/Shading.h"
#include "newtype/render/ProceduralTrace.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"

#include <algorithm>

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
// Kernel-side cache bundle helper — selects the Config-driven insert route
// (Sharc.h: SharcEngineCache). BufferVar members are move-only, so the bundle
// aggregates from the kernel arguments directly (harness pattern).
//==============================================================================
namespace {

// Template: if constexpr only discards the dead route inside a template —
// in a plain function both branches would need to compile.
template<bool UseLock = SharcEngineCache::uses_lock_buffer>
[[nodiscard]] inline auto make_sharc_cache(
    BufferVar<SharcKeyHost> &&entries, BufferVar<uint> &&locks,
    BufferVar<uint4> &&accum, BufferVar<SharcPackedData> &&resolved,
    UInt capacity, Float radiance_scale) noexcept {
    if constexpr (UseLock) {
        return SharcLockCache<SharcEngineLayout>{std::move(entries), std::move(locks),
                                                  std::move(accum), std::move(resolved),
                                                  capacity, radiance_scale};
    } else {
        return SharcCache<SharcEngineLayout>{std::move(entries), std::move(accum),
                                             std::move(resolved), capacity, radiance_scale};
    }
}

}// anonymous namespace

//==============================================================================
// compile
//==============================================================================
void PassSharc::compile(luisa::compute::Device& device,
                        scene::Geometry& geom,
                        const SurfaceResolverPoly& resolver) {
    _geom = &geom;

    //==========================================================================
    // Update kernel: sparse (1 pixel per downscaleFactor^2 tile) multi-bounce
    // camera paths with NEE feed SharcUpdateHit/UpdateMiss/SetThroughput.
    //==========================================================================
    _updateShader = device.compile<2>([&](
        BufferVar<SharcParams> params,
        BufferVar<SharcKeyHost> entries,
        BufferVar<uint>         locks,
        BufferVar<uint4>        accum,
        BufferVar<SharcPackedData> resolved,
        ImageUInt               seed_image,
        UInt                    frame_count,
        AccelVar                accel,
        Var<util::CameraData>   camera,
        Var<SceneGeometryResources> scene,
        BindlessVar             vertex_bindless,
        BindlessVar             tex_bindless,
        Var<LightSamplingResources> lights,
        Var<EnvLightResources>  env,
        Float                   env_exposure_val,
        BufferVar<render::PresampledCandidate> presample_env_tiles,
        UInt                    presample_env_total_entries,
        UInt                    hasTransparentShadowCasters,
        UInt                    res_x,
        UInt                    res_y
#if NT_ENABLE_PROCEDURAL
        ,
        BindlessVar             proc_bindless
#endif
    ) noexcept {
        set_name("Sharc_Update");
        set_block_size(16u, 16u, 1u);
        set_warp_size(32u); // SM6.6 for the native 64-bit CAS insert route

        auto p = params.read(0u);
        SharcGridParams grid{camera->position, kSharcGridLogarithmBase,
                             p.sceneScale, p.levelBias};
        SharcEngineCache cache = make_sharc_cache(std::move(entries), std::move(locks), std::move(accum), std::move(resolved), p.capacity, p.radianceScale);

        // Tile-sparse dispatch: one thread per NxN screen tile, mapped to the
        // same frame-varied hash-selected pixel the former full-screen W×H
        // dispatch activated (full-frustum coverage over time at ~1/N^2 pixel
        // rate, N=5 => ~4%). The other N^2-1 threads of the old dispatch
        // early-outed without side effects, so this is bit-exact with a 25×
        // smaller launch.
        UInt d = max(p.downscaleFactor, 1u);
        UInt2 tile = dispatch_id().xy();
        UInt sel = util::xxhash32(make_uint3(tile.x, tile.y, frame_count));
        UInt2 offs = make_uint2(sel % d, (sel / d) % d);
        UInt2 coord = tile * d + offs;
        UInt2 resolution = make_uint2(res_x, res_y);
        $if(any(coord >= resolution)) { $return(); };

        UInt base_seed = seed_image.read(coord).x;

        // Jittered camera ray within the pixel (AA + coverage over time).
        UInt sj = util::xxhash32(make_uint3(coord.x, coord.y, frame_count ^ 0x51ed270bu));
        Float jx = util::uniform_uint_to_float(sj);
        Float jy = util::uniform_uint_to_float(util::xxhash32(make_uint2(sj, 0x9e3779b9u)));
        Float2 ndc = ((make_float2(coord) + make_float2(jx, jy)) /
                      make_float2(resolution)) * 2.0f - 1.0f;
        auto ray = camera->generate_ray(ndc);

        Float3 origin = ray->origin();
        Float3 dir = ray->direction();
        Float3 wo = -dir;

        SharcState state;
        sharc_init(state);

        $for(b, 0u, p.updateBounces) {
            auto hit = render::trace_closest(accel, make_ray(origin, dir, 0.001f, 1e10f)
#if NT_ENABLE_PROCEDURAL
                , proc_bindless
#endif
            );
            $if(hit->miss()) {
                sharc_update_miss(cache, state,
                    eval_envmap_radiance(dir, env.envmap, env.env_width, env.env_height,
                                         env.env_rotation, env_exposure_val));
                $break;
            };

            // Procedural hits are out of Phase-1 scope (NT_ENABLE_PROCEDURAL=0).
#if NT_ENABLE_PROCEDURAL
            $if(hit.is_procedural) { $break; };
#endif

            UInt hit_inst = hit.inst;
            UInt hit_prim = hit.prim;
            Float hit_t = hit.committed_ray_t;

            UInt4 inst_data = scene.instance_buffer.read(hit_inst);
            Var<MaterialData> mat = scene.material_buffer.read(Expr{ inst_data.y & 0xFFu });

            SurfaceData surface = resolve_surface_from_instance(
                resolver, vertex_bindless, tex_bindless,
                inst_data, hit_prim, hit.bary,
                scene.material_buffer, wo,
                scene.instance_transform_buffer.read(hit_inst),
                                0.0f, make_float2(0.0f), 0u, 0u, hit_inst);

            // SHARC gate: roughness floored at update-path hits so the cache
            // only ever sees non-delta lobes (delta transport is not cacheable).
            Float roughness = max(surface.roughness, p.roughnessMin);

            // ---- NEE at the hit (single candidate: alias-table triangle, else
            // presampled env tile; one shadow ray) ----
            Float3 direct = def(make_float3(0.0f));
            $if(Expr{ mat.type == 12u } & Expr{ mat.meta < 0.5f }) {
                // Unlit/emissive: radiance is the emissive albedo.
                direct = surface.albedo;
            } $elif(surface.bsdf_type != 3u) {
                // Hoisted NEE BSDF (bit-identical to the PassGI gi_bounce
                // construction: bsdf_type pinned to 0, layered params zeroed).
                MaterialBSDF nee_bsdf = make_material_bsdf(
                    surface.albedo, roughness, surface.metallic, surface.ior,
                    0.f, 0.f,
                    0.f, 0.5f,
                    0.f, 1.3f, 0.f,
                    0.f, 0.f,
                    luisa::compute::make_float3(1.f, 0.f, 0.f), 1.f,
                    surface.attenuation, surface.conductor_k,
                    surface.ns);

                Float3 sel_dir = def(make_float3(0.0f, 1.0f, 0.0f));
                Float sel_dist = def(0.0f); // 0 = env (infinite)
                Float3 sel_mc = def(make_float3(0.0f));
                Bool have = def(false);

                UInt seed = util::xxhash32(
                    make_uint3(base_seed, hit_prim, frame_count ^ cast<UInt>(b)));
                seed = util::lcg_ui(seed);
                Float u_select = util::uniform_uint_to_float(seed);
                seed = util::lcg_ui(seed);
                Float u_tri_x = util::uniform_uint_to_float(seed);
                seed = util::lcg_ui(seed);
                Float u_tri_y = util::uniform_uint_to_float(seed);

                $if(lights.emissive_count > 0u) {
                    Float u_scaled = u_select * cast<Float>(lights.emissive_count);
                    UInt idx = cast<uint>(u_scaled);
                    idx = min(idx, lights.emissive_count - 1u);
                    Var<AliasEntry> entry = lights.alias_table.read(idx);
                    UInt light_idx = ite(u_scaled - cast<Float>(idx) < entry.pdf,
                                         entry.triangle_index, entry.alias_index);
                    light_idx = min(light_idx, lights.emissive_count - 1u);
                    auto verts = lights.triangle_vertices.read(light_idx);
                    Float su = sqrt(u_tri_x);
                    Float2 bary_light = make_float2(1.0f - su, u_tri_y * su);
                    Float3 light_point = bary_light.x * verts.v0
                                       + bary_light.y * verts.v1
                                       + (1.0f - bary_light.x - bary_light.y) * verts.v2;
                    Float3 to_light = light_point - surface.position;
                    Float light_dist = length(to_light);
                    Float3 light_dir = to_light * (1.0f / max(light_dist, 1e-6f));
                    Float cos_hit = max(0.0f, dot(surface.ns, light_dir));
                    $if(cos_hit > 0.0f & Expr{ light_dist > 0.05f }) {
                        auto tri_light = lights.triangle_lights.read(light_idx);
                        Float source_pdf = tri_light.pdf / tri_light.area;
                        source_pdf = max(source_pdf, 1e-10f);
                        Float3 light_normal = tri_light->normal();
                        Float cos_light = max(0.0f, dot(light_normal, -light_dir));
                        Float dist_sq = light_dist * light_dist;
                        Float3 illum = evaluate_direct_illuminance_with_geometry(
                            nee_bsdf, tri_light->emission(), light_dir, dist_sq,
                            cos_hit, cos_light, surface.ns, wo);
                        sel_mc = illum / source_pdf;
                        sel_dir = light_dir;
                        sel_dist = light_dist;
                        have = true;
                    };
                };

                $if(!have & Expr{ env.env_integral > 0.0f } &
                    Expr{ presample_env_total_entries > 0u }) {
                    UInt eseed = util::xxhash32(make_uint3(seed, 0xabcdefu, 0x1234u));
                    UInt entry_idx = eseed % presample_env_total_entries;
                    auto pc = presample_env_tiles.read(entry_idx);
                    Float3 env_dir = uv_to_direction(pc.bary_u, pc.bary_v,
                        env.env_width, env.env_height, env.env_rotation);
                    Float3 env_rad = eval_envmap_from_uv(pc.bary_u, pc.bary_v,
                        env.envmap, env.env_width, env.env_height, env_exposure_val);
                    Float cos_hit = max(0.0f, dot(surface.ns, env_dir));
                    $if(cos_hit > 0.0f) {
                        Float3 contrib = evaluate_env_contribution(
                            nee_bsdf, env_rad, env_dir, surface.ns, wo);
                        sel_mc = contrib * pc.inv_source_pdf;
                        sel_dir = env_dir;
                        sel_dist = 0.0f;
                        have = true;
                    };
                };

                $if(have) {
                    Float s_offset = max(0.001f * hit_t, 1e-4f);
                    Float shadow_tmax = ite(sel_dist > 0.0f, sel_dist - s_offset, 1e10f);
                    auto shadow_ray = make_ray(
                        surface.position + surface.ns * s_offset + sel_dir * (0.25f * s_offset),
                        sel_dir, s_offset, shadow_tmax);
                    Bool visible = def(true);
                    $if(hasTransparentShadowCasters == 0u) {
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
#if NT_ENABLE_PROCEDURAL
                            // Procedural blocker: inst is the shared proc TLAS slot
                            // (out of range for instance_buffer) — classify via the
                            // AABB's own material layers. Emissive passes; no
                            // alpha-cutout on procedural geometry today.
                            $if(shadow_hit.is_procedural) {
                                Var<scene::ProcInstanceData> s_proc = proc_bindless
                                    .buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                                    .read(shadow_hit.prim);
                                Var<MaterialData> s_mat =
                                    scene.material_buffer.read(Expr{ s_proc.material_layers & 0xFFu });
                                visible = (s_mat.type == 5u);
                            } $else {
#endif
                                UInt4 s_inst_data = scene.instance_buffer.read(shadow_hit.inst);
                                Var<MaterialData> s_mat =
                                    scene.material_buffer.read(Expr{ s_inst_data.y & 0xFFu });
                                visible = (s_mat.type == 5u) |
                                    is_alpha_cutout(s_mat, vertex_bindless, tex_bindless,
                                                    s_inst_data, shadow_hit.prim, shadow_hit.bary);
#if NT_ENABLE_PROCEDURAL
                            };
#endif
                        };
                    };
                    $if(visible) { direct = sel_mc; };
                };
            };

            // Firefly clamp — quantized uint accumulation has no headroom for
            // spikes (fp16 storage clamps at 65504 on the resolve side anyway).
            Float dl = luminance(direct);
            direct = ite(dl > p.maxRadiance, direct * (p.maxRadiance / max(dl, 1e-8f)), direct);

            // Resampling coin for SharcUpdateHit (cache resampling on).
            UInt resample_seed = util::xxhash32(
                make_uint3(base_seed ^ cast<UInt>(b), hit_prim, frame_count));
            Float u_resample = util::uniform_uint_to_float(resample_seed);

            Bool cont = sharc_update_hit(cache, grid, state, surface.position,
                                         surface.ns, direct, u_resample);
            $if(!cont) { $break; };

            // ---- Sample the next segment (roughness floored => non-delta) ----
            MaterialBSDF bsdf = surface.make_bsdf_roughened(p.roughnessMin);
            UInt bseed = util::xxhash32(make_uint3(base_seed, cast<UInt>(b) + 7919u, hit_prim));
            auto pcg = util::pcg2d(make_uint2(bseed, bseed ^ 0x68bc21ebu));
            Float2 u_brdf = make_float2(util::uniform_uint_to_float(pcg.x),
                                        util::uniform_uint_to_float(pcg.y));
            Float brdf_pdf = def(0.0f);
            Float3 wi = bsdf.sample(wo, surface.ns, u_brdf, brdf_pdf);
            Float cos_wi = dot(wi, surface.ns);
            $if(brdf_pdf <= 1e-6f | Expr{ cos_wi } <= 1e-4f) { $break; };

            Float3 seg_throughput = bsdf.evaluate(wo, wi, surface.ns) *
                                    max(cos_wi, 0.0f) / brdf_pdf;
            sharc_set_throughput(state, seg_throughput);

            Float offset = max(0.001f * hit_t, 1e-4f);
            origin = surface.position + surface.ns * offset;
            dir = wi;
            wo = -wi;
        };

        // Advance the per-pixel RNG stream (GI's seed-rotation pattern).
        seed_image.write(coord, make_uint4(
            util::xxhash32(make_uint2(coord.x * 1973u + coord.y * 9277u, base_seed))));
    });

    //==========================================================================
    // Resolve kernel: 1 thread/entry — EWMA + eviction + probe-window recovery
    // + adjacent-level blending (ported sharc_resolve_entry).
    //==========================================================================
    _resolveShader = device.compile<1>([&](
        BufferVar<SharcParams> params,
        BufferVar<SharcKeyHost> entries,
        BufferVar<uint>         locks,
        BufferVar<uint4>        accum,
        BufferVar<SharcPackedData> resolved,
        Var<util::CameraData>   camera
    ) noexcept {
        set_name("Sharc_Resolve");
        set_block_size(256u, 1u, 1u);
        set_warp_size(32u);

        auto p = params.read(0u);
        SharcGridParams grid{camera->position, kSharcGridLogarithmBase,
                             p.sceneScale, p.levelBias};
        SharcResolveParams resolve{camera->prev_position,
                                   p.accumulationFrameNum, p.staleFrameNumMax};
        SharcEngineCache cache = make_sharc_cache(std::move(entries), std::move(locks), std::move(accum), std::move(resolved), p.capacity, p.radianceScale);

        sharc_resolve_entry(cache, grid, resolve, dispatch_id().x);
    });

    //==========================================================================
    // Clear kernel: reset all buffers to the empty-entry state (scene edits,
    // toggles, first frame).
    //==========================================================================
    _clearShader = device.compile<1>([&](
        BufferVar<SharcKeyHost> entries,
        BufferVar<uint>         locks,
        BufferVar<uint4>        accum,
        BufferVar<SharcPackedData> resolved,
        UInt                    capacity
    ) noexcept {
        set_name("Sharc_Clear");
        set_block_size(256u, 1u, 1u);
        set_warp_size(32u);

        UInt i = dispatch_id().x;
        $if(i < capacity) {
            entries.write(i, def<SharcKeyHost>(static_cast<SharcKeyHost>(0)));
            locks.write(i, 0u);
            accum.write(i, make_uint4(0u));
            resolved.write(i, sharc_zero_packed_data());
        };
    });

    //==========================================================================
    // Occupancy counter kernel: atomic count of valid entries.
    //==========================================================================
    _countShader = device.compile<1>([&](
        BufferVar<SharcKeyHost> entries,
        BufferVar<uint>         stats,
        UInt                    capacity
    ) noexcept {
        set_name("Sharc_Count");
        set_block_size(256u, 1u, 1u);
        set_warp_size(32u);

        UInt i = dispatch_id().x;
        $if(i < capacity) {
            auto key = entries.read(i);
            $if(key != def<SharcKeyHost>(static_cast<SharcKeyHost>(0))) {
                stats.atomic(0u).fetch_add(1u);
            };
        };
    });

    //==========================================================================
    // Debug view: colored hash of the primary surface's cache entry, or the
    // bucket occupancy heat at that surface.
    //==========================================================================
#if NT_DEBUG_VIZ
    _debugShader = device.compile<2>([&](
        BufferVar<SharcParams> params,
        BufferVar<SharcKeyHost> entries,
        ImageFloat              target,
        ImageFloat              gbuf_depth,
        ImageUInt               gbuf_vis,
        ImageFloat              gbuf_bary_motion,
        Var<util::CameraData>   camera,
        Var<SceneGeometryResources> scene,
        BindlessVar             vertex_bindless,
        BindlessVar             tex_bindless,
        UInt                    mode
    ) noexcept {
        set_name("Sharc_Debug");
        set_block_size(16u, 16u, 1u);

        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(any(coord >= resolution)) { $return(); };

        auto p = params.read(0u);
        SharcGridParams grid{camera->position, kSharcGridLogarithmBase,
                             p.sceneScale, p.levelBias};

        Float depth = gbuf_depth.read(coord).x;
        UInt4 vis = gbuf_vis.read(coord);
        UInt inst_id = vis.x;
        Float3 color = def(make_float3(0.0f));

        $if(inst_id != ~0u) {
            UInt prim_id = vis.y & 0x3FFFFFFFu;
            Float2 bary = gbuf_bary_motion.read(coord).xy();

            Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
            auto ray = camera->generate_ray(ndc);
            Float3 wo = -normalize(ray->direction());

            UInt4 inst_data = scene.instance_buffer.read(inst_id);
            SurfaceData surface = resolve_surface_from_instance(
                resolver, vertex_bindless, tex_bindless,
                inst_data, prim_id, bary,
                scene.material_buffer, wo,
                scene.instance_transform_buffer.read(inst_id),
                0.0f, (make_float2(coord) + 0.5f) / make_float2(resolution),
                                resolution.x, resolution.y, inst_id);

            $if(mode == 1u) {
                // HashGrid_DebugColoredHash: entry identity flicker-check.
                color = sharc_debug_colored_hash<SharcEngineLayout>(
                    surface.position, surface.ns, grid);
            } $elif(mode == 2u) {
                // Bucket occupancy: fraction of the 16-slot probe bucket that
                // holds any entry (heat: blue 0 -> red 1).
                Float voxel_size;
                auto key = sharc_compute_spatial_hash<SharcEngineLayout>(
                    surface.position, surface.ns, grid, voxel_size);
                UInt base_slot = sharc_get_base_slot<SharcEngineLayout>(key, p.capacity);
                UInt used = 0u;
                $for(offset, 0u, kSharcBucketSize) {
                    auto stored = entries.read(base_slot + offset);
                    used = used + ite(stored != def<SharcKeyHost>(
                        static_cast<SharcKeyHost>(0)), 1u, 0u);
                };
                Float frac = cast<Float>(used) * (1.0f / static_cast<float>(kSharcBucketSize));
                color = make_float3(frac, frac * frac, 1.0f - frac) * frac;
            };
        };

        target.write(coord, make_float4(color, 1.0f));
    });
#endif // NT_DEBUG_VIZ
}

//==============================================================================
// createResources
//==============================================================================
void PassSharc::createResources(luisa::compute::Device& device, uint entriesNum) {
    _entriesNum = render::sharc_validate_capacity(entriesNum);
    if (_entriesNum == 0u) {
        CI_LOG_E("PassSharc: entriesNum must be a power of two >= "
                 << kSharcBucketSize << " (got " << entriesNum
                 << "); falling back to the default "
                 << kSharcDefaultEntriesNum << " entries.");
        _entriesNum = kSharcDefaultEntriesNum;
    }

    _entriesBuf  = device.create_buffer<SharcKeyHost>(_entriesNum);
    _locksBuf    = device.create_buffer<uint>(_entriesNum);
    _accumBuf    = device.create_buffer<uint4>(_entriesNum);
    _resolvedBuf = device.create_buffer<SharcPackedData>(_entriesNum);
    _statsBuf    = device.create_buffer<uint>(1u);
    _queryStatsBuf = device.create_buffer<uint>(5u);  // GI hit/miss, gather hit/miss, gather refl fallbacks
    _paramsBuf   = device.create_buffer<SharcParams>(1u);

    _entriesBuf.set_name("sharc_entries");
    _locksBuf.set_name("sharc_locks");
    _accumBuf.set_name("sharc_accumulation");
    _resolvedBuf.set_name("sharc_resolved");
    _statsBuf.set_name("sharc_stats");
    _queryStatsBuf.set_name("sharc_query_stats");
    _paramsBuf.set_name("sharc_params");

    // Poll completion event (submitPolls -> readPolls; see those headers).
    // PRESERVED across recreations: destroying/recreating a timeline event on
    // every resize wore on LuisaCompute's event/fence plumbing (stale
    // CommandQueue::Wait values around each scale change — see resize-TDR
    // debugging 2026-09-16) and served no purpose: the event has no size
    // dependence. Create only on the first call.
    if (!_pollEvent) {
        _pollEvent = device.create_timeline_event();
        _pollFence = 0u;
        _pollQueued = false;
    }

    // Zero the query counters once up front: unlike the occupancy counter
    // (zeroed in the same poll that dispatches it), these accumulate during
    // normal GI/gather frames, so the first poll must start from a known value.
    // Same stream as all SHARC traffic — FIFO ordering covers the racing
    // dispatches without a sync.
    {
        static const uint kQueryZero[5] = {0u, 0u, 0u, 0u, 0u};
        Renderer::stream() << _queryStatsBuf.copy_from(kQueryZero);
    }

    // Plan §7 defaults / §5 tuning
    _paramsCpu.sceneScale           = 10.0f;
    _paramsCpu.levelBias            = 0.0f;
    _paramsCpu.radianceScale        = 1e3f;
    _paramsCpu.accumulationFrameNum = 32u;
    _paramsCpu.staleFrameNumMax     = 64u;
    _paramsCpu.roughnessMin         = 0.4f;
    _paramsCpu.maxRadiance          = 1e4f;
    _paramsCpu.updateBounces        = 3u;
    _paramsCpu.downscaleFactor      = 5u;
    _paramsCpu.capacity             = _entriesNum;
    _paramDirty = true;
    _needsReset = true;
    _entryCountHost = 0u;
}

//==============================================================================
// release
//==============================================================================
void PassSharc::release() {
    _geom = nullptr;
    _entriesNum = 0u;
    (void)_entriesBuf.release();
    (void)_locksBuf.release();
    (void)_accumBuf.release();
    (void)_resolvedBuf.release();
    (void)_statsBuf.release();
    (void)_queryStatsBuf.release();
    (void)_paramsBuf.release();
    _pollEvent = {};
    _presampleEnvTilesPtr = nullptr;
    _presampleEnvTotalEntries = 0u;
    _queryHitsHost = 0u;
    _queryMissesHost = 0u;
    _gatherHitsHost = 0u;
    _gatherMissesHost = 0u;
    _gatherFallbackHost = 0u;
}

//==============================================================================
// _populateParams — copy CPU staging struct to GPU buffer (deferred: member)
//==============================================================================
void PassSharc::_populateParams(luisa::compute::CommandList& cmdlist) noexcept {
    cmdlist << _paramsBuf.copy_from(&_paramsCpu);
}

//==============================================================================
// renderUpdate
//==============================================================================
void PassSharc::renderUpdate(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
    if (_entriesNum == 0u || _geom == nullptr) return;

    // Cache reset: scene edits / accum resets invalidate every entry; the
    // query miss fallback (Phase 2) covers the refill window by design.
    if (ctx.accumReset || _needsReset) {
        cmdlist << _clearShader(_entriesBuf, _locksBuf, _accumBuf, _resolvedBuf,
                                _entriesNum).dispatch(_entriesNum);
        _needsReset = false;
    }
    if (_paramDirty) {
        _populateParams(cmdlist);
        _paramDirty = false;
    }

    auto& ls = ctx.lightSampler;
    auto& geom = ctx.geometry;

    SceneGeometryResources scene_resources {
        geom.instance_buffer(),
        geom.instance_transform_buffer(),
        geom.instance_transform_prev_buffer(),
        ctx.materialPool.buffer(),
        ctx.materialPool.simKeyBuffer()
    };
    LightSamplingResources light_resources {
        ls.triangle_buffer(), ls.vertex_buffer(),
        ls.alias_table(), ls.emissive_triangle_count(),
        ls.total_power_inv(), ls.emissive_count_inv(),
        ls.instance_to_light_base()
    };
    EnvLightResources env_resources {
        ls.envmap_image(), ls.env_cdf_marginal(),
        ls.env_cdf_conditional(), ls.env_integral(),
        ls.env_width(), ls.env_height(),
        ls.env_rotation_matrix()
    };

    // Tile-sparse grid: ceil(W/d) × ceil(H/d) threads, one per NxN tile (the
    // kernel re-derives the active pixel from the tile hash). _paramsCpu is
    // the same source the GPU params buffer was uploaded from.
    const uint d = std::max(_paramsCpu.downscaleFactor, 1u);
    cmdlist << _updateShader(
        _paramsBuf,
        _entriesBuf,
        _locksBuf,
        _accumBuf,
        _resolvedBuf,
        ctx.seedImage,
        ctx.frameCount,
        geom.tlas(),
        ctx.camera,
        scene_resources,
        geom.vertex_bindless(),
        ctx.materialPool.textures(),
        light_resources,
        env_resources,
        ls.env_exposure(),
        *_presampleEnvTilesPtr,
        _presampleEnvTotalEntries,
        geom.has_transparent_shadow_casters() ? 1u : 0u,
        ctx.width,
        ctx.height
#if NT_ENABLE_PROCEDURAL
        , *ctx.procBindless
#endif
    ).dispatch((ctx.width + d - 1u) / d,
               (ctx.height + d - 1u) / d);
}

//==============================================================================
// renderResolve
//==============================================================================
void PassSharc::renderResolve(luisa::compute::CommandList& cmdlist, const FrameContext& ctx) {
    if (_entriesNum == 0u) return;
    cmdlist << _resolveShader(
        _paramsBuf,
        _entriesBuf,
        _locksBuf,
        _accumBuf,
        _resolvedBuf,
        ctx.camera
    ).dispatch(_entriesNum);
}

//==============================================================================
// renderDebug
//==============================================================================
#if NT_DEBUG_VIZ
void PassSharc::renderDebug(luisa::compute::CommandList& cmdlist, const FrameContext& ctx,
                            luisa::compute::Image<float>& target) {
    if (_entriesNum == 0u || _geom == nullptr) return;
    if (_paramDirty) {
        _populateParams(cmdlist);
        _paramDirty = false;
    }

    auto& geom = ctx.geometry;
    SceneGeometryResources scene_resources {
        geom.instance_buffer(),
        geom.instance_transform_buffer(),
        geom.instance_transform_prev_buffer(),
        ctx.materialPool.buffer(),
        ctx.materialPool.simKeyBuffer()
    };

    cmdlist << _debugShader(
        _paramsBuf,
        _entriesBuf,
        target,
        ctx.gbufDepth,
        ctx.gbufVis,
        ctx.gbufBaryMotion,
        ctx.camera,
        scene_resources,
        geom.vertex_bindless(),
        ctx.materialPool.textures(),
        static_cast<uint>(_debugView)
    ).dispatch(ctx.width, ctx.height);
}
#endif // NT_DEBUG_VIZ

//==============================================================================
// submitPolls — enqueue the occupancy count + query-stat readbacks at the END
// of the frame (after the tail work), then signal a timeline event. No stream
// synchronize: results are consumed by readPolls() at the next cadence tick
// (>= 1 frame later), whose host-side event wait is already satisfied in
// practice. The zeroing of the query counters opens a fresh stats window.
//==============================================================================
void PassSharc::submitPolls(luisa::compute::Stream& stream) {
    if (_entriesNum == 0u) return;
    static const uint kZero = 0u;
    static const uint kQueryZero[5] = {0u, 0u, 0u, 0u, 0u};
    stream << _statsBuf.copy_from(&kZero)
           << _countShader(_entriesBuf, _statsBuf, _entriesNum).dispatch(_entriesNum)
           << _statsBuf.copy_to(&_entryCountHost)
           << _queryStatsBuf.copy_to(_queryStatsHost)
           << _queryStatsBuf.copy_from(kQueryZero);
    ++_pollFence;
    stream << _pollEvent.signal(_pollFence);
    _pollQueued = true;
}

//==============================================================================
// readPolls — consume the results of the last submitPolls. Host-side event
// wait; the poll was submitted at the previous cadence tick, so this does not
// block in practice.
//==============================================================================
void PassSharc::readPolls() {
    if (!_pollQueued || !_pollEvent) return;
    _pollEvent.synchronize(_pollFence);
    _pollQueued = false;
    _queryHitsHost = _queryStatsHost[0];
    _queryMissesHost = _queryStatsHost[1];
    _gatherHitsHost = _queryStatsHost[2];
    _gatherMissesHost = _queryStatsHost[3];
    _gatherFallbackHost = _queryStatsHost[4];
}

//==============================================================================
// drawUi
//==============================================================================
void PassSharc::drawUi() {
    if (ImGui::CollapsingHeader("SHARC Radiance Cache")) {
        ImGui::Checkbox("Enabled (Update+Resolve)", &_enabled);
        ImGui::SameLine();
        if (ImGui::Button("Reset Cache")) requestReset();

        if (ImGui::TreeNode("Gather Tuning")) {
            int transTaps = static_cast<int>(_gatherTransTaps);
            if (ImGui::SliderInt("Transmission Taps K", &transTaps, 1, 8))
                setGatherTransTaps(static_cast<uint>(transTaps));
            int reflTaps = static_cast<int>(_gatherReflTaps);
            if (ImGui::SliderInt("Reflection Taps K'", &reflTaps, 0, 8))
                setGatherReflTaps(static_cast<uint>(reflTaps));
            ImGui::SliderFloat("Temporal Alpha Min", &_gatherTemporalAlpha, 0.02f, 1.0f);
            ImGui::TreePop();
        }

        const char* debugNames[] = { "Off", "Colored Hash", "Bucket Occupancy" };
        int debugIdx = static_cast<int>(_debugView);
        if (ImGui::Combo("Debug View", &debugIdx, debugNames, IM_ARRAYSIZE(debugNames)))
            _debugView = static_cast<DebugView>(debugIdx);

        _paramDirty |= ImGui::DragFloat("Scene Scale", &_paramsCpu.sceneScale, 0.1f, 0.5f, 200.0f);
        _paramDirty |= ImGui::DragFloat("Level Bias", &_paramsCpu.levelBias, 0.01f, -2.0f, 2.0f);
        _paramDirty |= ImGui::DragFloat("Radiance Scale", &_paramsCpu.radianceScale, 1.0f, 1.0f, 1e5f);
        _paramDirty |= ImGui::DragInt("Accum Frames", reinterpret_cast<int*>(&_paramsCpu.accumulationFrameNum), 1.0f, 1, 1024);
        _paramDirty |= ImGui::DragInt("Stale Frames Max", reinterpret_cast<int*>(&_paramsCpu.staleFrameNumMax), 1.0f, 8, 1024);
        _paramDirty |= ImGui::DragFloat("Roughness Min", &_paramsCpu.roughnessMin, 0.005f, 0.05f, 1.0f);
        _paramDirty |= ImGui::DragFloat("Max Radiance", &_paramsCpu.maxRadiance, 1.0f, 10.0f, 1e6f);
        _paramDirty |= ImGui::DragInt("Update Bounces", reinterpret_cast<int*>(&_paramsCpu.updateBounces), 1.0f, 1, 4);
        _paramDirty |= ImGui::DragInt("Downscale Factor", reinterpret_cast<int*>(&_paramsCpu.downscaleFactor), 1.0f, 2, 9);
        _paramsCpu.downscaleFactor = std::clamp(_paramsCpu.downscaleFactor, 2u, 9u);

        ImGui::Text("Occupancy: %.2f%% (%u / %u entries)",
                    static_cast<double>(occupancy() * 100.0f),
                    _entryCountHost, _entriesNum);
        ImGui::Text("GI Query: %.1f%% hit (%u hit / %u miss, last window)",
                    static_cast<double>(queryHitRate() * 100.0f),
                    _queryHitsHost, _queryMissesHost);
        ImGui::Text("Gather Query: %.1f%% hit (%u hit / %u miss, last window)",
                    static_cast<double>(gatherHitRate() * 100.0f),
                    _gatherHitsHost, _gatherMissesHost);
        // Speckle-fix diagnostic: reflection taps resolving to the fallback
        // without attempting a lookup never reached the hit/miss counters —
        // a high hit rate could hide a fallback-dominated pixel. Post-fix the
        // fallback value is stable, so watch the speckle, not the %.
        ImGui::Text("Gather Taps: %u (fallback %.1f%%)",
                    gatherAttempts() + gatherFallbackTaps(),
                    static_cast<double>(gatherFallbackShare() * 100.0f));
        ImGui::Text("Memory: %.1f MiB",
                    static_cast<double>(render::sharc_total_bytes(
                        _entriesNum, NT_SHARC_COMPACT != 0,
                        SharcEngineCache::uses_lock_buffer) / (1024.0 * 1024.0)));
    }
}

} // namespace core
} // namespace newtype
