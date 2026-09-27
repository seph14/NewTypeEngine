#include "newtype/core/Pipeline.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/render/Shading.h"
// Dispersion-rings debug tag traces the first glass interface (the 2-arg
// fallback overload serves non-procedural builds, same as Shading.h's guard).
#if NT_DEBUG_VIZ && NT_ENABLE_SHARC && NT_ENABLE_DISPERSION
#include "newtype/render/ProceduralTrace.h"
#endif
#include "cinder/Log.h"
#include "newtype/util/Profiler.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

//==============================================================================
// Render
//==============================================================================

void Pipeline::render(Renderer& renderer, const util::CameraData& data, DebugTag debugTag, float dt) noexcept {
    if (!isBuilt()) {
        CI_LOG_W("Pipeline::render called before buildScene() - TLAS is null "
            "and shaders are not compiled; skipping frame to avoid device "
            "removal. Call buildScene() once before the render loop.");
        return;
    }
    if (_frameSubmitted) {
        CI_LOG_W("Pipeline::render called twice without an intervening "
            "beginFrame() - the previous frame's GPU tail is still in flight; "
            "skipping this frame to avoid command-list races.");
        return;
    }
    // GPU-GPU ordering with this frame's compute-stream writes (TLAS/BLAS
    // builds, instance uploads, light-sampler rewrites, env rotation upload).
    // The CPU no longer stalls for those — the render queue fences on the
    // signal instead (replaces the former per-animated-frame CPU sync).
    if (_geomUpdatePending) {
        Renderer::stream() << _geomUpdateEvent.wait(_geomUpdateFence);
        _geomUpdatePending = false;
    }
    auto& frame = *_currentFrame;
    DebugTag debugTagResolved = (_debugTag != DebugTag::None) ? _debugTag : debugTag;

    // Check for custom material callable DLL hot-reload (Debug builds with
    // the DLL actually loaded)
#ifdef _DEBUG
    if (_callableDLL.isLoaded() && _callableDLL.checkAndReload()) {
        CI_LOG_I("Pipeline: Custom material callables reloaded, recompiling shaders...");
        _recompileAllShaders();
        requestAccumReset();
    }

    // Hot-reload any pending shader DLLs on the main thread. The watch thread
    // only sets a flag; performReload (which this calls) syncs the stream,
    // destroys the old shader, rebuilds, and recompiles. Running on the render
    // thread guarantees the render loop is paused and no in-flight stream
    // commands hold the shader's pointer when we delete it.
    ShaderManager::instance().processPendingReloads(_computeStream);
#endif

    // Checkerboard field: 0=off, 1 or 2 = active field (alternates each frame)
    uint cbField = _checkerboardEnabled ? ((_frameCount & 1u) ? 1u : 2u) : 0u;

    // Primary jitter gate (docs/primary_jitter_plan.md): when disabled, zero
    // the incoming jitter on a local copy so every consumer downstream of the
    // FrameContext sees a fully deterministic primary ray — generate_ray stops
    // offsetting, the `ndc - jitter` unjittered reconstructions become
    // identity, the MV jitter delta and the denoiser jitter consts go to 0.
    // prev_jitter is zeroed too so the delta terms are exactly 0 and the
    // on→off toggle costs at most one frame of swim. Zero shader changes.
    //
    // FSR/DLSS exception (perf R2 item 14): at renderScale < 1 the upscaler
    // is the jitter resolver and expects jittered input, so let the Halton
    // pattern through (it feeds generate_ray AND the upscaler's
    // jitterOffset below) while the full-res TAA-style jitter stays off.
    // DLSS-RR also expects jittered input at native res (guide: use a long
    // Halton sequence, >= 32 positions), so it engages the same exception.
    util::CameraData cam = data;
    const bool perspectiveProjection =
        data.projection == static_cast<uint32_t>(util::CameraProjection::Perspective);
    const bool rrJitterOn =
        _fsrJitterEnabled && _denoiserMode == DenoiserMode::DlssRR &&
        _denoiser.enabled() && _rrDenoiser.available() && perspectiveProjection;
    const bool upscalerJitterOn =
        (_fsrJitterEnabled && _renderScale < 1.0f && upscalerActive()) || rrJitterOn;
    if (!_primaryJitterEnabled && !upscalerJitterOn) {
        cam.jitter = luisa::float2(0.0f);
        cam.prev_jitter = luisa::float2(0.0f);
    }

    // FPS-aware coordination: sync PassGI's accumulation state to PassDI so its
    // visMaxAge derivation in _populateDiParams has up-to-date values (handles
    // user toggling the checkbox in PassGI's UI mid-session).
    _passDI.setVisAgeDerivation(_passGI.giAccumulationTimeEnabled(),
                                _passGI.giAccumulationTime());

#if NT_ENABLE_SHARC
    // The GI query hook follows the SHARC runtime state — cache maintenance
    // off means no fresh data, so queries stay off too (the Phase-2/3 A/B
    // toggles were removed 2026-09-09; the master toggle owns both).
    _passGI.setSharcQueryEnabled(_passSharc.enabled());
#endif

    // Compile-time specialization flip check (oneBounce / giScale /
    // sharcQueryOn / diBiasCorrectionEnabled / hasTransparentShadowCasters).
    // Runs at the same render-thread safe point as the DLL hot-reload above:
    // no in-flight stream commands can hold the old shader objects. A flip
    // (UI checkbox, geometry rebuild, SHARC toggle) pays one recompile of the
    // affected passes; revisited variants hit the shader disk cache.
    _refreshSpecialization();

    // Checkerboard flip (DLSS-RR engage/leave, config reload): the layout is
    // baked into kernels + reservoir image widths, so the change recompiles
    // and recreates here at the render-thread safe point — same contract as
    // _refreshSpecialization (the image recreation adds resize()'s sync
    // envelope inside).
    if (_checkerboardBaked != _checkerboardEnabled)
        _applyCheckerboardReconfigure();

    // Denoiser projection specialization (docs §9 perf investigation): the
    // ReLAX projection-sensitive kernels bake the active projection as a
    // compile-time constant — perspective keeps the pre-projection
    // instruction sequence. A camera-type flip recompiles those kernels
    // here, at the same render-thread safe point as _refreshSpecialization;
    // revisited projections hit the shader disk cache.
    if (_denoiser.enabled() && _denoiser.bakedProjection() != cam.projection) {
        CI_LOG_I("Denoiser: projection specialization flip "
            << _denoiser.bakedProjection() << " -> " << cam.projection
            << " - recompiling projection-sensitive kernels");
        _denoiser.setBakedProjection(cam.projection);
        _denoiser.recompileProjectionKernels(Renderer::device());
    }

    // Build shared frame context. displayWidth/Height carry the PRESENT
    // (display-target) dims — with a render-size override the display target
    // (tonemap output, recorder capture) runs at render resolution, and
    // AfterToneMap features must dispatch at that size, not the window's.
    FrameContext ctx = {
        _gbufDepth, _gbufVis, _gbufBaryMotion, _glassThroughput,
        _gbufVelocity, _gbufDepthUpscale,
        cam, *_geom, *_materialPool, *_lightSampler,
        _frameCount, _width, _height, renderer.presentWidth(), renderer.presentHeight(),
        cbField, _accumReset,
        dt,
        _seedImage,
        _accumBuffer, _specularBuffer,
        _denoiseAlbedo, _denoiseSpecFactor, _denoiseNormal,
        _gbufDepthPrev, _gbufVisPrev, _denoiseNormalPrev,
        _passSSS.sssRadiance(),
#if NT_ENABLE_SHARC
        _roughGlassInfo,
#endif
#if NT_ENABLE_PROCEDURAL
        &_procBindless,
#endif
        _solidBackgroundEnabled, _solidBackgroundColor
    };

    auto& profiler = util::Profiler::instance();
    
    //==========================================================================
    // Pass 1: G-Buffer (primary rays -> visibility buffer)
    //==========================================================================
    {
        auto cl = CommandList::create();
        profiler.set_pass("DI/G-Buffer");
        util::CpuScopedTimer _cpu_DI_G_Buffer("DI/G-Buffer");
        _passDI.renderGBuffer(cl, ctx);
        Renderer::stream() << cl.commit();
    }

    //==========================================================================
    // Pass 1.7: SSS Probe (Burley 2015 BSSRDF).
    // Full-res dispatch; reads G-Buffer + scene and writes per-channel
    // demodulated radiance to _passSSS.sssRadiance(). Runs before presample
    // so the probe's light sampling completes before _rasterStream touches
    // shared light buffers (matches Pass 1.5 rationale).
    // Skipped entirely when no visible instance uses a Subsurface material —
    // matches the glass-dispatch gating pattern.
    //==========================================================================
    if (_geom->has_active_subsurface()) {
        auto cl = CommandList::create();
        profiler.set_pass("SSS/Probe");
        util::CpuScopedTimer _cpu_SSS_Probe("SSS/Probe");
        _passSSS.renderProbe(cl, ctx);
        Renderer::stream() << cl.commit();
    }

    //==========================================================================
    // Pass 1.5: Presample light candidates (batched)
    // Moved before raster features so Renderer::stream() finishes accessing
    // shared buffers (env CDF, material) before _rasterStream touches them.
    //==========================================================================
    {
        util::CpuScopedTimer _cpu_DI_Presample("DI/Presample");
        auto cl = CommandList::create();
        profiler.set_pass("DI/Presample Local");
        _passDI.renderPresampleLocal(cl, ctx);
        profiler.set_pass("DI/Presample Env");
        _passDI.renderPresampleEnv(cl, ctx);
        if (!cl.empty())
            Renderer::stream() << cl.commit();
    }

    // Signal that Renderer::stream() is done with shared buffers.
    // _rasterStream waits for this before accessing env CDF, material, etc.
    ++_renderReadyFence;
    Renderer::stream() << _renderReadyEvent.signal(_renderReadyFence);

    //==========================================================================
    // Pass 1.55: SHARC Update + Resolve (docs/sharc_rough_glass_plan.md §7).
    // Sparse-path cache maintenance on one CommandList — the DX backend's
    // automatic UAV barriers order Update -> Resolve (and Resolve's writes
    // against GI initial's reads further down the same stream). Runs before
    // the DI/GI reuse chain; GI initial queries the cache at its x2 hits
    // (Phase 2, gated by the SHARC panel's A/B toggle).
    //==========================================================================
#if NT_ENABLE_SHARC
    if (_passSharc.enabled()) {
        auto cl = CommandList::create();
        profiler.set_pass("SHARC/Update");
        {
            util::CpuScopedTimer _cpu_SHARC_Update("SHARC/Update");
            _passSharc.renderUpdate(cl, ctx);
        }
        profiler.set_pass("SHARC/Resolve");
        {
            util::CpuScopedTimer _cpu_SHARC_Resolve("SHARC/Resolve");
            _passSharc.renderResolve(cl, ctx);
        }
        Renderer::stream() << cl.commit();

        // Debug view (colored hash / bucket occupancy): bypass the rest of the
        // frame like the Glass debug checkpoint.
#if NT_DEBUG_VIZ
        if (_passSharc.debugView() != PassSharc::DebugView::Off) {
            auto cl2 = CommandList::create();
            _passSharc.renderDebug(cl2, ctx, frame.display_target);
            Renderer::stream() << cl2.commit();
            Renderer::stream() << synchronize();
            _frameSubmitted = false;
            _accumReset = false;
            _frameCount++;
            return;
        }
#endif
    }
#endif

    // Clear voxel grid before raster features voxelize into it
    // Must use _computeStream (same queue as the update shader in NewTypeEngine::update)
    // to avoid D3D12 simultaneous-access errors on the positions buffer.
#if NT_ALLOW_RASTER_FEATURES
    if (_voxelGrid && _voxelGrid->created()) {
        profiler.set_pass("Voxel/Clear");
        util::CpuScopedTimer _cpu_Voxel_Clear("Voxel/Clear");
        _voxelGrid->clear(_computeStream);
    }
#endif
    _dispatchFeatures(_computeStream, FeaturePoint::AfterGBuffer, ctx, frame.render_target);
    // Voxel grid dilate + summary must run on _rasterStream (AFTER the
    // transparent pass) to avoid a cross-queue write-read race on voxel
    // buffers. FIFO ordering guarantees the transparent shader finishes
    // reading before we modify them. _featureGbufEvent is signaled after
    // so Renderer::stream() waits for the full update.
#if NT_ALLOW_RASTER_FEATURES
    if (_rasterContext && _featureGbufFence != 0u) {
        auto& rs = _rasterContext->rasterStream();
        if (_voxelGrid && _voxelGrid->created()) {
            luisa::uint dilate_iters = 0u;
            for (auto& feat : _features[static_cast<size_t>(FeaturePoint::AfterGBuffer)]) {
                auto* vgu = dynamic_cast<IVoxelGridUser*>(feat.get());
                if (vgu && feat->enabled())
                    dilate_iters = max(dilate_iters, vgu->shadowDilation());
            }
            if (dilate_iters > 0u) {
                profiler.set_pass("Voxel/Dilate");
                util::CpuScopedTimer _cpu_Voxel_Dilate("Voxel/Dilate");
                _voxelGrid->dilate(rs, dilate_iters);
            }
            {
                profiler.set_pass("Voxel/Summary");
                util::CpuScopedTimer _cpu_Voxel_Summary("Voxel/Summary");
                _voxelGrid->buildSummary(rs);
            }
        }
        rs << _featureGbufEvent.signal(_featureGbufFence);
    }
#endif

    // Glass debug: show PSR throughput + vis data immediately after G-buffer
#if NT_DEBUG_VIZ
    if (debugTagResolved == DebugTag::Glass) {
        static auto glassDebugShader = Renderer::device().compile<2>([&](
            ImageFloat out_frame,
            ImageFloat glass_throughput,
            ImageUInt  gbuf_vis,
            ImageFloat gbuf_depth,
            BufferVar<luisa::uint4> instance_buffer,
            BufferVar<MaterialData> material_buffer
#if NT_ENABLE_PROCEDURAL
            ,
            BindlessVar proc_bindless
#endif
        ) noexcept {
            UInt2 coord = dispatch_id().xy();
            Float4 glass = glass_throughput.read(coord);
            UInt4  vis   = gbuf_vis.read(coord);
            Float  depth = gbuf_depth.read(coord).x;
            UInt   inst  = vis.x;
            UInt   is_g  = vis.y >> 31u;
            Bool   is_point_viz = ((vis.y >> 30u) & 1u) > 0u;
#if NT_ENABLE_PROCEDURAL
            Bool   is_procedural_viz = ((vis.y >> 29u) & 1u) > 0u;
#endif
            Float cr = def(0.0f);
            Float cg = def(0.0f);
            Float cb = def(0.0f);

            $if(inst == ~0u) {
                // Miss - dark blue
                cb = 0.3f;
            } $elif(is_g > 0u) {
                // Glass pixel: R=throughput.r, G=throughput.g, B=Fresnel_accum
                cr = glass.x;
                cg = glass.y;
                cb = glass.w;
            } 
#if NT_ENABLE_PROCEDURAL
            $elif(is_procedural_viz) {
                // Procedural diagnostic: R=inst_id/10, G=material_idx/10, B=type/10
                // Bypasses denoiser — reads vis buffer directly from G-buffer
                Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(inst);
                cr = cast<float>(inst) * 0.1f;
                cg = cast<float>(proc_inst.material_layers & 0xFFu) * 0.1f;
                cb = cast<float>(proc_inst.type) * 0.1f;
            } 
#endif
            $elif(is_point_viz) {
                // Point pixel: vis.x = material_id, show albedo
                Var<MaterialData> pmat = material_buffer.read(Expr{ inst });
                Float3 albedo = pmat.albedo.xyz();
                cr = albedo.x; cg = albedo.y; cb = albedo.z;
            } $else {
                // Opaque: show material INDEX (inst_data.y & 0xFF) in R,
                //         material TYPE in G, depth in B
                UInt4 inst_data = instance_buffer.read(inst);
                UInt mat_idx = inst_data.y & 0xFFu;
                Var<MaterialData> mat = material_buffer.read(Expr{ mat_idx });
                Float mat_idx_f = cast<float>(mat_idx);
                Float mtype_f = cast<float>(mat.type);
                // R = index/16, G = type/12, B = depth
                cr = mat_idx_f* (1.0f / 16.0f);
                cg = mtype_f* (1.0f / 12.0f);
                cb = 0.f;//min(depth * 0.15f, 1.0f);
            };

            out_frame.write(coord, make_float4(cr, cg, cb, 1.0f));
        });
        _computeStream <<
            glassDebugShader(frame.display_target, _glassThroughput, _gbufVis, _gbufDepth,
                _geom->instance_buffer(), _materialPool->buffer()
#if NT_ENABLE_PROCEDURAL
                ,_procBindless
#endif
            ).dispatch(_width, _height);
        _computeStream << synchronize();
        _frameSubmitted = false;
        _accumReset = false;
        _frameCount++;
        return;
    }
#endif // NT_DEBUG_VIZ

    // Dispersion rings debug (speckle plan Phase 4): per-channel
    // first-interface refraction / TIR topology of dispersive glass.
    // Encoding: non-glass 0.05 gray; non-dispersive glass 0.5 (smooth) /
    // 0.25 (rough) gray; dispersive glass RGB = per-channel refraction
    // booleans at the first interface (channel TIRs → its primary goes
    // dark) — the concentric critical-angle bands (B TIRs first, then G,
    // then R) read as yellow → red rings toward the rim. Rough dispersive
    // pixels are dimmed ×0.35: their taps sample GGX half-vectors, so the
    // deterministic first-interface test shown is the envelope of where
    // per-tap TIR begins, exact only for smooth glass.
#if NT_DEBUG_VIZ && NT_ENABLE_SHARC && NT_ENABLE_DISPERSION
    if (debugTagResolved == DebugTag::Dispersion) {
        static auto dispRingsShader = Renderer::device().compile<2>([&](
            ImageFloat out_frame,
            ImageFloat rough_glass_info,
            ImageUInt  gbuf_vis,
            Var<util::CameraData> camera,
            Var<SceneGeometryResources> scene,
            BindlessVar vertex_bindless,
            AccelVar accel
#if NT_ENABLE_PROCEDURAL
            ,
            BindlessVar proc_bindless
#endif
        ) noexcept {
            set_name("disp_rings_debug");
            UInt2 coord = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();
            $if(any(coord >= resolution)) { $return(); };

            Float4 info = rough_glass_info.read(coord);
            Float rough = info.x;
            Float eta_d = max(info.y, 1e-3f);
            Float abbe  = info.w;
            Bool is_glass = (gbuf_vis.read(coord).y >> 31u) != 0u;

            Float3 out_col = def(make_float3(0.05f));
            $if(is_glass & (abbe > 0.0f)) {
                // Reconstruct the first interface via the unjittered camera
                // ray (same protocol as the tint pass's replay).
                auto cam_ray = camera->generate_ray(Expr{
                    (make_float2(coord) + 0.5f) / make_float2(resolution)
                        * 2.0f - 1.0f - camera.jitter });
                auto hit = render::trace_closest(accel, cam_ray
#if NT_ENABLE_PROCEDURAL
                    , proc_bindless
#endif
                );
                Float3 col = def(make_float3(0.1f));
                $if(hit->inst != ~0u) {
                    Float t_first = hit.committed_ray_t;
                    Float3 hit_pos = cam_ray->origin() + cam_ray->direction() * t_first;
                    Float3 wo = -normalize(cam_ray->direction());
                    Float3 glass_ns;
#if NT_ENABLE_PROCEDURAL
                    $if(hit.is_procedural) {
                        UInt packed_proc_prim = (1u << 29u) | hit.local_tri;
                        glass_ns = reconstruct_procedural_normal(
                            proc_bindless, hit->prim, packed_proc_prim,
                            hit_pos, hit.local_bary);
                    } $else {
#endif
                        UInt4 inst_data = scene.instance_buffer.read(hit.inst);
                        auto glass_xform = scene.instance_transform_buffer.read(hit.inst);
                        Float3 glass_obj_n = reconstruct_normal(vertex_bindless,
                            inst_data.z, inst_data.w, hit.prim, hit.bary);
                        glass_ns = transform_normal(glass_xform, glass_obj_n);
#if NT_ENABLE_PROCEDURAL
                    };
#endif
                    Float3 ns = ite(dot(wo, glass_ns) < 0.0f, -glass_ns, glass_ns);
                    Float glass_d = ite(eta_d < 1.0f, 1.0f / eta_d, eta_d);
                    Float cos_i = dot(wo, ns);
                    // Per-channel first-interface TIR test at the channel IOR.
                    Float3 refr = def(make_float3(0.0f));
                    $for(c, 3u) {
                        Float ior_c = dispersed_ior(glass_d, abbe, c);
                        Float eta_c = ite(eta_d < 1.0f, 1.0f / ior_c, ior_c);
                        Float sin2_t = eta_c * eta_c * (1.0f - cos_i * cos_i);
                        Float pass = ite(sin2_t < 1.0f, 1.0f, 0.0f);
                        refr = ite(c == 0u, make_float3(pass, refr.y, refr.z),
                             ite(c == 1u, make_float3(refr.x, pass, refr.z),
                                        make_float3(refr.x, refr.y, pass)));
                    };
                    col = refr;
                };
                out_col = ite(rough > kRoughGlassEpsilon, col * 0.35f, col);
            } $elif(is_glass) {
                out_col = ite(rough > kRoughGlassEpsilon,
                              make_float3(0.25f), make_float3(0.5f));
            };
            out_frame.write(coord, make_float4(out_col, 1.0f));
        });
        _computeStream <<
            dispRingsShader(frame.display_target, _roughGlassInfo, _gbufVis, cam,
                scene_gpu_resources(), _geom->vertex_bindless(), _geom->tlas()
#if NT_ENABLE_PROCEDURAL
                , _procBindless
#endif
            ).dispatch(_width, _height);
        _computeStream << synchronize();
        _frameSubmitted = false;
        _accumReset = false;
        _frameCount++;
        return;
    }
#endif // NT_DEBUG_VIZ && NT_ENABLE_SHARC && NT_ENABLE_DISPERSION

    // GPU wait: ensure async feature G-buffer modifications (e.g., PointCloud raster merge)
    // are complete before Candidate reads per-pixel G-buffer data
    if (_featureGbufFence != 0u) {
        Renderer::stream() << _featureGbufEvent.wait(_featureGbufFence);
        _featureGbufFence = 0u;
    }

    //==========================================================================
    // Pass 1.6 + Pass 2: ClassifyTiles + Denoiser Prefilter + DI Candidate
    // (single CommandList). Prefilter writes _denoiseAlbedo/_denoiseNormal;
    // Candidate writes reservoirs and reads G-Buffer. Disjoint outputs,
    // overlapping read-only inputs — safe to batch on the same stream. Saves
    // one stream submission per frame. ClassifyTiles leads the batch (perf R2
    // item 9): gbufDepth is final here (post raster-merge wait), so the
    // prefilter's sky-tile early-out and every later tile consumer see THIS
    // frame's classification instead of one-frame-stale data.
    //==========================================================================
    {
        auto cl = CommandList::create();
        {
            profiler.set_pass("ReLAX/classify");
            util::CpuScopedTimer _cpu_ReLAX_classify("ReLAX/classify");
            _denoiser.renderClassifyTiles(cl, ctx);
        }
        {
            profiler.set_pass("ReLAX/prefilter");
            util::CpuScopedTimer _cpu_ReLAX_prefilter("ReLAX/prefilter");
            _denoiser.renderPrefilter(cl, ctx);
        }
        {
            profiler.set_pass("DI/Candidate");
            util::CpuScopedTimer _cpu_DI_Candidate("DI/Candidate");
            _passDI.renderCandidate(cl, ctx);
        }
        Renderer::stream() << cl.commit();
    }

    // DI Debug checkpoint: AfterCandidate
#if NT_DEBUG_VIZ
    if (_diDebugPass == DIDebugPass::AfterCandidate) {
        Renderer::stream() <<
            _diDebugReservoirShader(frame.display_target, _passDI.reservoirBuffer(), _gbufDepth, _gbufVis, cbField)
            .dispatch(_checkerboardBaked ? (_width + 1u) / 2u : _width, _height);
        Renderer::stream() << synchronize();
        _frameSubmitted = false;
        _accumReset = false;
        _frameCount++;
        return;
    }
#endif // NT_DEBUG_VIZ

    //==========================================================================
    // DI Temporal + Boiling + Spatial + GI + Shade (single command list)
    // All shared resources are read-only on the same DX queue — no concurrent
    // access hazards. Batching eliminates N stream submissions per frame.
    // UAV barriers between passes are inserted automatically by the DX backend.
    //==========================================================================
    bool giEnabled = _passGI.enabled();

    {
        auto cl = CommandList::create();

        // DI Temporal + Boiling
        {
            util::CpuScopedTimer _cpu_DI_Temporal("DI/Temporal+Boiling");
            if (!_accumReset && _frameCount != 0) {
                profiler.set_pass("DI/Temporal");
                _passDI.renderTemporal(cl, ctx);
            }
            if (_frameCount != 0) {
                profiler.set_pass("DI/Boiling Filter");
                _passDI.renderBoiling(cl, ctx, _boilingFilterStrength);
            }
        }

        // DI Debug checkpoint: AfterTemporal / AfterBoiling
        // AfterBoiling collides with AfterTemporal in the current pipeline layout
        // because DI Temporal and DI Boiling share a single command list (no
        // dispatch boundary between them). Treat both selectors the same —
        // they both dump post-boiling (== post-temporal) state.
    #if NT_DEBUG_VIZ
        if (_diDebugPass == DIDebugPass::AfterTemporal ||
            _diDebugPass == DIDebugPass::AfterBoiling) {
            Renderer::stream() << cl.commit();
            Renderer::stream() <<
                _diDebugReservoirShader(frame.display_target, _passDI.reservoirBuffer(), _gbufDepth, _gbufVis, cbField)
                .dispatch(_checkerboardBaked ? (_width + 1u) / 2u : _width, _height);
            // Copy current→prev for temporal continuity (spatial won't run in this debug path)
            {
                auto cl2 = CommandList::create();
                cl2 << _passDI.reservoirBuffer().copy_to(_passDI.reservoirPrevBuffer());
                if (giEnabled) _passGI.copyReservoirs(cl2);
                Renderer::stream() << cl2.commit();
            }
            Renderer::stream() << synchronize();
            _frameSubmitted = false;
            _accumReset = false;
            _frameCount++;
            return;
        }
    #endif // NT_DEBUG_VIZ

        // DI Spatial + full GI pipeline
        {
            util::CpuScopedTimer _cpu_DI_Spatial("DI/Spatial Reuse");
            profiler.set_pass("DI/Spatial Reuse");
            _passDI.renderSpatial(cl, ctx);

            if (giEnabled) {
                util::CpuScopedTimer _cpu_GI_Pipeline("GI/Pipeline");
                profiler.set_pass("GI/Initial");
                _passGI.renderInitial(cl, ctx);

                // Snapshot frozen initial GI reservoirs for shade-time MIS.
                // Full-res only (half-res MIS deferred — see plan).
                if (!_passGI.giHalfRes()) {
                    profiler.set_pass("GI/SnapshotInitial");
                    _passGI.snapshotInitial(cl);
                }
                if (!_accumReset && _frameCount != 0) {
                    profiler.set_pass("GI/Temporal");
                    _passGI.renderTemporal(cl, ctx);
                }
                if (_frameCount != 0) {
                    profiler.set_pass("GI/Boiling Filter");
                    _passGI.renderBoiling(cl, ctx, _boilingFilterStrength);
                }
                // Half-res: upsample runs AFTER temporal+boiling so both operate
                // on the half-res pair (with the half-res history ping-pong)
                // instead of the full-res grid — saves two full-res pass
                // invocations per frame (checkerboard: ~2x, without: ~4x).
                if (_passGI.giHalfRes()) {
                    profiler.set_pass("GI/Upsample");
                    _passGI.renderUpsample(cl, _width, _height);
                }
                if (!_passGI.giHalfRes()) {
                    profiler.set_pass("GI/Spatial Reuse");
                    _passGI.renderSpatial(cl, ctx);
                }
            }
        }

        // Flip reservoir indices BEFORE constructing the shade dispatch below:
        // spatial wrote to reservoirPrevBuffer() (the free slot), and shade must
        // read that slot via reservoirBuffer() (post-flip). flipReservoir() is
        // CPU-only (swaps an index), so it can run before cl.commit() without
        // affecting GPU ordering. Keeping shade in the same CommandList as
        // DI Temporal+Boiling+Spatial+GI saves one stream submission per frame.
        // Half-res mode skips this flip: shade reads the upsampled Full buffer
        // and the single end-of-frame flip advances the half-res history.
        _passDI.flipReservoir();
        if (giEnabled && !_passGI.giHalfRes())
            _passGI.flipReservoir();

        // Stagnancy smoothing (RTXDI 3.1 decorrelation support): reads the
        // same final GI reservoir buffer shade reads (post-flip / post-
        // upsample), so it must run after the flip block above.
        if (giEnabled && _passGI.giDecorrelationMode() != 0u) {
            profiler.set_pass("GI/StagnancySmooth");
            _passGI.renderStagnancySmooth(cl, ctx);
        }

        // DI/GI Debug checkpoints: AfterSpatial / GIReservoir — must read
        // post-spatial reservoir state, so commit the partial CL (without shade)
        // before the debug blit, then early-return.
#if NT_DEBUG_VIZ
        if (_diDebugPass == DIDebugPass::AfterSpatial) {
            Renderer::stream() << cl.commit();
            Renderer::stream() <<
                _diDebugReservoirShader(frame.display_target, _passDI.reservoirBuffer(), _gbufDepth, _gbufVis, cbField)
                .dispatch(_checkerboardBaked ? (_width + 1u) / 2u : _width, _height);
            {
                auto cl2 = CommandList::create();
                cl2 << _passDI.reservoirBuffer().copy_to(_passDI.reservoirPrevBuffer());
                if (giEnabled)
                    _passGI.copyReservoirs(cl2);
                Renderer::stream() << cl2.commit();
            }
            Renderer::stream() << synchronize();
            _frameSubmitted = false;
            _accumReset = false;
            _frameCount++;
            return;
        }

        // GI Debug checkpoint: dumps the post-spatial GI reservoir state.
        // R = invalid flag (0.8 if invalid), G = W/20 (capped at 1), B = M/64, A = age/30
        if (debugTagResolved == DebugTag::GIReservoir && giEnabled) {
            Renderer::stream() << cl.commit();
            Renderer::stream() <<
                _giDebugReservoirShader(frame.display_target, _passGI.reservoirBuffer(), _gbufDepth, _gbufVis, cbField)
                .dispatch(_checkerboardBaked ? (_width + 1u) / 2u : _width, _height);
            {
                auto cl2 = CommandList::create();
                cl2 << _passDI.reservoirBuffer().copy_to(_passDI.reservoirPrevBuffer());
                _passGI.copyReservoirs(cl2);
                Renderer::stream() << cl2.commit();
            }
            Renderer::stream() << synchronize();
            _frameSubmitted = false;
            _accumReset = false;
            _frameCount++;
            return;
        }
#endif // NT_DEBUG_VIZ

        //==========================================================================
        // Pass 8: Combined Shade (DI + GI shadow rays -> raw output)
        // Appended to the same CommandList as DI Temporal+Boiling+Spatial+GI.
        // The DX backend's EnhancedBarrierTracker inserts an automatic UAV
        // barrier between spatial's write to the new current slot and shade's
        // read from it, since both are bound as UAVs in the same CL.
        //==========================================================================
#if NT_DEBUG_VIZ
        // Debug views write render_target (NOT display_target): with a tag
        // active ReLAX skips its composite into render_target but still
        // reports rendered=true, and the frame-end tone map blits
        // render_target -> display_target, which would overwrite anything
        // written to display_target here with stale data (frozen image).
        if (debugTagResolved == DebugTag::Visibility) {
            cl << _debugVisShader(frame.render_target, _gbufVis)
                .dispatch(_width, _height);
        } else if (debugTagResolved == DebugTag::Motion) {
            cl << _debugMotionShader(frame.render_target, _gbufBaryMotion)
                .dispatch(_width, _height);
        } else
#endif
        {
            profiler.set_pass("Shade/Compose");
            util::CpuScopedTimer _cpu_Shade_Compose("Shade/Compose");
            cl << _shadeShader(
                    _accumBuffer,                       // 0: raw output
                    _passDI.reservoirBuffer(),          // 1: DI reservoir buffer
                    _gbufDepth,                         // 2: gbuf_depth
                    _gbufVis,                           // 3: gbuf_vis
                    _gbufBaryMotion,                    // 4: gbuf_bary_motion
                    _frameCount,                        // 5: frame_count
                    _geom->tlas(),                      // 6: TLAS (shadow rays)
                    cam,                                // 7: camera
                    scene_gpu_resources(),              // 8: scene geometry binding group
                    _geom->vertex_bindless(),           // 9: vertex_bindless
                    _materialPool->textures(),          // 10: tex_bindless (material textures)
                    light_gpu_resources(),              // 11: light sampling binding group
                    _passGI.reservoirBuffer(),          // 12: GI reservoir buffer
                    env_gpu_resources(),                // 13: env light binding group
                    _lightSampler->env_exposure(),      // 14: env_exposure
                    cbField,                            // 15: cbField
                    static_cast<uint>(_shadeDebugVizMode), // 16: debug viz mode
                    _specularBuffer,                    // 17: specular output
                    _glassThroughput,                   // 18: glass throughput
                    _denoiseAlbedo,                     // 19: denoise albedo output
                    _denoiseSpecFactor,                 // 19b: NRD spec demod factor output
                    _passDI.visMaxAge(),                // 20: visibility reuse max age
                    _passDI.visMaxDistance(),           // 21: visibility max spatial distance (local lights)
                    _passDI.envVisMaxDistance()         // 22: visibility max spatial distance (env lights)
                    // 23 (hasTransparentShadowCasters) is compile-time baked
                    // into the shade shader (see _compileShadeShader).
#if NT_ALLOW_RASTER_FEATURES
                    ,
                    _voxelGrid->gpu_resources(),         // 24: voxel grid binding group
                    _shadowCachePrev,                     // 25: shadow cache prev (read)
                    _shadowCache                          // 26: shadow cache (write)
#endif
                    ,
                    _passGI.initialSnapshotBuffer(),     // 24/27: GI initial-reservoir snapshot
                    _passGI.giMISRoughness(),  // 25/28: MIS roughness (0=off)
                    _passGI.deltaBranchNEEEnabled() ? 1u : 0u,  // 26/28: delta-branch NEE for mirror metals
                    _passDI.dispShadowInterfaces(),             // 27/29: dispersive shadow sub-walk budget (dial (a))
                    _passDI.dispShadowSplit(),                  // 28/30: dispersive shadow RGB-split saturation (1 = exact)
                    _passSSS.sssRadiance(),
                    // Stagnancy decorrelation (RTXDI 3.1): mode != None gates
                    // the swap; firefly honors the boiling flag only where it
                    // was written (full-res GI mode — mirrors renderBoiling).
                    _passGI.stagnancyTexture(),
                    _passGI.fireflyFlagTexture(),
                    _passGI.giDecorrelationMode(),
                    _passGI.giDecorrelationFactor(),
                    _passGI.giDecorrelationStagnancyExponent(),
                    (!_passGI.giHalfRes() && _passGI.giFireflyReplaceActive()) ? 1u : 0u,
                    _passGI.giDecorrelationMultiplyBound()
#if NT_ENABLE_PROCEDURAL
                    , _procBindless
#endif
                )
                .dispatch(_checkerboardBaked ? (_width + 1u) / 2u : _width, _height);
        }

        Renderer::stream() << cl.commit();
    }

    // Shade pass is the last reader of _triangle_lights / _alias_table on
    // Renderer::stream(). Signal so next frame's matLightsDirty path can
    // safely rewrite them on _computeStream without a write-while-read race.
    ++_lightSamplerReadyFence;
    Renderer::stream() << _lightSamplerReadyEvent.signal(_lightSamplerReadyFence);

    _dispatchFeatures(Renderer::stream(), FeaturePoint::AfterShade, ctx, frame.render_target);

    // DI Debug checkpoint: AfterShade (show raw shade output, skip denoiser)
#if NT_DEBUG_VIZ
    if (_diDebugPass == DIDebugPass::AfterShade && debugTagResolved == DebugTag::None) {
        Renderer::stream() <<
                _compositeBlitShader(
                    frame.display_target,
                    _accumBuffer,
                    _gbufDepth,
                    env_gpu_resources(),
                    cam,
                _lightSampler->env_exposure(),
                _denoiseAlbedo,
                _denoiseSpecFactor,
                _specularBuffer,
                _solidBackgroundEnabled ? 1u : 0u,
                _solidBackgroundColor
            )
            .dispatch(_width, _height);

        // Copy reservoirs to prev for next frame's temporal reuse. (The prev
        // G-buffer set needs no copy — ping-pong handle rotation in
        // beginFrame(), perf R2 item 10.)
        {
            auto cl = CommandList::create();
            cl << _passDI.reservoirBuffer().copy_to(_passDI.reservoirPrevBuffer());
            if (_passGI.enabled())
                _passGI.copyReservoirs(cl);
            Renderer::stream() << cl.commit();
        }

        Renderer::stream() << synchronize();
        _frameSubmitted = false;
        _accumReset = false;
        _frameCount++;
        return;
    }
#endif // NT_DEBUG_VIZ

    //==========================================================================
    // Pass 9: Denoiser (ReLAX) — or DLSS Ray Reconstruction when engaged.
    // RR replaces the denoiser 1:1 at render res: compose the noisy HDR
    // (same remodulation the no-denoiser fallback uses), format the guide
    // buffers, evaluate NGX, and write denoised HDR back into the render
    // target so AfterDenoiser / OIT / glass / tonemap continue unchanged.
    //==========================================================================
    // History-reset trigger shared by RR and the upscaler stage below (the
    // consume is one-shot; hoisting it here lets both features see it).
    const bool upscalerReset = _consumeUpscalerReset(ctx);
    _rrDenoiserRan = false;
    {
        const bool rrActive = rrDenoiserActive() &&
                              perspectiveProjection &&
                              !_progressiveAccum &&
                              debugTagResolved == DebugTag::None;
        bool denoiserRendered = false;
        if (rrActive) {
            {
                profiler.set_pass("Shade/DLSS-RR compose");
                util::CpuScopedTimer _cpu_RR_compose("Shade/DLSS-RR compose");
                // Noisy composited HDR: remodulated shade + emission + sky.
                Renderer::stream() <<
                    _compositeBlitShader(
                        _rrColor,
                        _accumBuffer,
                        _gbufDepth,
                        env_gpu_resources(),
                        cam,
                        _lightSampler->env_exposure(),
                        _denoiseAlbedo,
                        _denoiseSpecFactor,
                        _specularBuffer,
                        _solidBackgroundEnabled ? 1u : 0u,
                        _solidBackgroundColor
                    )
                    .dispatch(_width, _height);
                // Guide buffers: float albedo / F0 / packed normal+roughness.
                profiler.set_pass("Shade/DLSS-RR format");
                util::CpuScopedTimer _cpu_RR_format("Shade/DLSS-RR format");
                Renderer::stream() <<
                    _rrInputFormatShader(
                        _rrAlbedo, _rrF0, _rrNormalRoughness,
                        _denoiseAlbedo, _denoiseSpecFactor, _denoiseNormal
                    )
                    .dispatch(_width, _height);
            }
            {
                profiler.set_pass("Shade/DLSS-RR eval");
                util::CpuScopedTimer _cpu_RR_eval("Shade/DLSS-RR eval");
                ngx::RrFrameParams rrParams;
                // Jitter in RENDER pixels, same conversion as the upscaler.
                // Mirrors the jitter actually applied this frame (the zeroing
                // block above decided cam.jitter) — a stale-zero offset
                // against jittered rays desyncs RR's history and blurs the
                // reconstruction.
                rrParams.jitterOffset =
                    luisa::make_float2(cam.jitter.x * 0.5f * static_cast<float>(_width),
                                       cam.jitter.y * 0.5f * static_cast<float>(_height));
                rrParams.reset = upscalerReset;
                rrParams.width = _width;
                rrParams.height = _height;
                _rrDenoiser.dispatch(Renderer::stream(),
                                     _rrColor, _gbufDepthUpscale, _gbufVelocity,
                                     _rrAlbedo, _rrF0, _rrNormalRoughness,
                                     frame.render_target, rrParams);
            }
            denoiserRendered = true;
            _rrDenoiserRan = true;
        } else {
            const Image<float>* rasterDepthPtr = nullptr;
#if NT_ALLOW_RASTER_FEATURES
            if (_rasterContext && _rasterContext->hasResources()) rasterDepthPtr = &_rasterContext->rasterDepth();
#endif
            denoiserRendered = _denoiser.render(Renderer::stream(), ctx,
                frame.render_target, debugTagResolved == DebugTag::None, rasterDepthPtr);

            // No-denoiser fallback: composite raw shade output with envmap for sky
            if (!denoiserRendered && debugTagResolved == DebugTag::None) {
                Renderer::stream() <<
                    _compositeBlitShader(
                        frame.render_target,
                        _accumBuffer,
                        _gbufDepth,
                        env_gpu_resources(),
                        cam,
                        _lightSampler->env_exposure(),
                        _denoiseAlbedo,
                        _denoiseSpecFactor,
                        _specularBuffer,
                        _solidBackgroundEnabled ? 1u : 0u,
                        _solidBackgroundColor
                    )
                    .dispatch(_width, _height);
            }

            if (denoiserRendered)
                _denoiser.advanceFrame();
        }

        // AfterDenoiser features run on raw denoised HDR — before OIT particles
        // and glass tint are baked in, so transparent effects composite cleanly.
        if (denoiserRendered)
            _dispatchFeatures(Renderer::stream(), FeaturePoint::AfterDenoiser, ctx, frame.render_target);

        // OIT composite: blend transparent particles over denoised output
#if NT_ALLOW_RASTER_FEATURES
        if (denoiserRendered && _rasterContext && _rasterContext->hasResources() && _rasterContext->requireOITBlit()) {
            profiler.set_pass("Shade/OIT composite");
            util::CpuScopedTimer _cpu_OIT("Shade/OIT composite");
            Renderer::stream() <<
                _oitCompositeShader(
                    frame.render_target,
                    _rasterContext->oitAccum(),
                    _rasterContext->oitLogReveal()
                ).dispatch(_width, _height);
        }
#endif

        // Glass tint pass: apply glass attenuation + Fresnel reflection on denoised output
        // For non-glass pixels, this is identity passthrough (throughput = 1,1,1,0)
        if (denoiserRendered && _geom->has_visible_glass() && debugTagResolved == DebugTag::None) {
#if NT_ENABLE_SHARC
            // Phase 3 rough-glass gather + smooth dispersive replay follow the
            // SHARC runtime state (cache maintenance must be on — the gather
            // queries fresh data; the replay shares the toggle, its history
            // images and its reset logic). The history reset covers accum
            // resets, the first frame, and any gap since the last gather
            // frame (toggles, resize, skipped tint frames): both history
            // slots are zero-filled so the .w validity flag can never read
            // garbage as valid.
            bool sharcGather = _passSharc.enabled();
            bool gatherReset = _accumReset || _frameCount == 0u ||
                               _roughGlassLastFrame == ~0u ||
                               (_frameCount - _roughGlassLastFrame) > 1u;
            if (sharcGather && gatherReset) {
                Renderer::stream() <<
                    _roughGlassClearShader(_roughGlassHist[0]).dispatch(_width, _height) <<
                    _roughGlassClearShader(_roughGlassHist[1]).dispatch(_width, _height);
            }
#endif
            profiler.set_pass("Shade/glass tint");
            util::CpuScopedTimer _cpu_Shade_glass_tint("Shade/glass tint");
            Renderer::stream() <<
                _glassTintShader(
                    frame.render_target,
                    frame.render_target,    // read + write same target (in-place)
                    _glassThroughput,
                    _gbufDepth,
                    _gbufVis,
                    _gbufBaryMotion,
                    cam,
                    env_gpu_resources(),
                    _lightSampler->env_exposure(),
                    scene_gpu_resources(),
                    _geom->vertex_bindless(),
                    _geom->tlas()
#if NT_ENABLE_PROCEDURAL
                    , _procBindless
#endif
#if NT_ENABLE_SHARC
                    ,
                    _materialPool->textures(),          // tap surface resolves
                    _roughGlassInfo,
                    _passSharc.paramsBuffer(),
                    _passSharc.entriesBuffer(),
                    _passSharc.resolvedBuffer(),
                    _passSharc.queryStatsBuffer(),
                    sharcGather ? 1u : 0u,
                    _passSharc.gatherTransTaps(),
                    _passSharc.gatherReflTaps(),
                    _roughGlassHist[_roughGlassHistIdx],       // read slot
                    _roughGlassHist[_roughGlassHistIdx ^ 1u],  // write slot
                    _frameCount,
                    _passSharc.gatherTemporalAlpha(),
                    gatherReset ? 1u : 0u
#endif
	            )
                .dispatch(_width, _height);
#if NT_ENABLE_SHARC
            if (sharcGather) {
                _roughGlassLastFrame = _frameCount;
                _roughGlassHistIdx ^= 1u;  // next frame reads what this frame wrote
            }
#endif
        }
    }

    _dispatchFeatures(Renderer::stream(), FeaturePoint::AfterGlassTint, ctx, frame.render_target);

    //==========================================================================
    // Progressive (offline) accumulation: running average of the composed HDR
    // frame (denoiser path is bypassed while active). The tonemap below reads
    // this frame's accumulator instead of the raw render target.
    //==========================================================================
    Image<float>* progressiveOutput = nullptr;
    if (_progressiveAccum) {
        if (_progressiveResetPending) {
            _progressiveFrame = 0u;
            _progressiveResetPending = false;
        }
        // Create both pool entries first, THEN take references: the second
        // push_back can reallocate _tempImages and would dangle a reference
        // taken before it (use-after-free at the dispatch below).
        const char* idDst  = _progressiveAccumIdx == 0u ? "prog_accum_a" : "prog_accum_b";
        const char* idPrev = _progressiveAccumIdx == 0u ? "prog_accum_b" : "prog_accum_a";
        requestTempImage(PixelStorage::FLOAT4, _width, _height, idDst);
        requestTempImage(PixelStorage::FLOAT4, _width, _height, idPrev);
        auto& dst  = requestTempImage(PixelStorage::FLOAT4, _width, _height, idDst);
        auto& prev = requestTempImage(PixelStorage::FLOAT4, _width, _height, idPrev);
        profiler.set_pass("Shade/ProgressiveAccum");
        util::CpuScopedTimer _cpu_Shade_ProgAccum("Shade/ProgressiveAccum");
        Renderer::stream() << _progressiveAccumShader(
            dst, frame.render_target, prev, _progressiveFrame
        ).dispatch(_width, _height);
        progressiveOutput = &dst;
        _progressiveFrame++;
        _progressiveAccumIdx ^= 1u; // next frame swaps dst/prev (true ping-pong)
    }

    //==========================================================================
    // Upscaler stage: HDR render-res -> display-res transition, pre-tonemap.
    // Runs after the frame is fully composed (denoised + OIT + glass tint +
    // AfterGlassTint features). Jitter stays off in Phase 1 (plumbing only —
    // docs/upscaling_feasibility_report.md); the upscaler becomes the jitter
    // resolver when TAA returns.
    //==========================================================================
    _lastFrameUpscaled = false;
    const bool nonPerspectiveProjection =
        cam.projection != static_cast<uint32_t>(util::CameraProjection::Perspective);
    if (nonPerspectiveProjection && upscalerActive()) {
        // FSR's fov + NDC-depth contract only holds for the plain perspective
        // camera (docs/non_perspective_camera_report.md §3); setRenderResolution
        // Override normally keeps the upscaler off — this is the belt to that
        // suspenders for perspective-fov-config paths.
        static bool sWarnedUpscalerProjection = false;
        if (!sWarnedUpscalerProjection) {
            sWarnedUpscalerProjection = true;
            CI_LOG_W("Upscaler: skipped - non-perspective projection active "
                     "(fov/NDC-depth contract does not hold)");
        }
    }
    if (_upscalerMode != upscal::UpscalerMode::None && !upscalerActive() &&
        !_upscalerUnavailableLogged) {
        CI_LOG_W("Upscaler: mode is set but the backend is inactive (RGBA8 "
                 "display + loaded DLLs + valid dims required) - upscaling "
                 "falls back to a bilinear stretch until reconfigured.");
        _upscalerUnavailableLogged = true;
    }
    if (_upscalerMode != upscal::UpscalerMode::None && !nonPerspectiveProjection &&
        !_progressiveAccum && renderer.isRGBA8()) {
        const bool reset = upscalerReset; // consumed once above (shared with RR)
        profiler.set_pass("Shade/Upscale");
        util::CpuScopedTimer _cpu_Shade_Upscale("Shade/Upscale");
        upscal::UpscaleFrameParams p;
        // Jitter in RENDER pixels (UpscalerBackend.h contract), converted
        // from the NDC units Camera produces — the mirror of the MV math in
        // the DI velocity write (ndc * 0.5 * render_size). Halton-0.5 lands
        // in [-0.5, 0.5) px, FSR's required range. Mirrors the jitter
        // actually applied this frame (cam.jitter after the zeroing decision
        // above), so the upscaler's history stays phase-locked to the rays
        // even when only the full-res TAA jitter toggle is on.
        p.jitterOffset       =
            luisa::make_float2(cam.jitter.x * 0.5f * static_cast<float>(_width),
                               cam.jitter.y * 0.5f * static_cast<float>(_height));
        p.preExposure        = 1.0f;
        p.deltaTimeSeconds   = dt;
        p.reset              = reset;
        p.cameraNear         = data.near_clip;
        p.cameraFar          = data.far_clip;
        p.cameraFovVertical  = luisa::radians(data.fov);
        p.sharpness          = _upscalerSharpness;
        p.renderWidth        = _width;
        p.renderHeight       = _height;
        p.displayWidth       = _displayWidth;
        p.displayHeight      = _displayHeight;
        if (upscalerActive()) {
            _upscalerBackend->dispatch(Renderer::stream(),
                                       frame.render_target, _gbufDepthUpscale,
                                       _gbufVelocity, _upscaledHdr, p);
        } else {
            // Backend cannot run this frame (feature creation failed — e.g.
            // DLSS rejected the current dims). Keep the view whole: tonemap
            // still reads _upscaledHdr, so fill it with a bilinear stretch of
            // the render target instead of leaving it stale. The backend
            // retries on the next size change (resize clears the latch).
            Renderer::stream() <<
                _stretchBlitShader(_upscaledHdr, frame.render_target,
                                   _width, _height)
                    .dispatch(_displayWidth, _displayHeight);
        }
        _lastFrameUpscaled = true;
    }

    // Tonemap + copy-prev batched into one CommandList to reduce stream submission
    // overhead. Each stream << creates a CommandAllocator with 3MB pre-allocated
    // GPU buffers; batching saves 1-2 submissions per frame.
    {
        auto cl = CommandList::create();
        // Record the tonemap decision for AfterToneMap features (FXAA runs
        // only when the display target actually holds tone-mapped LDR).
        _lastFrameTonemapped = renderer.isRGBA8();
        // Tone-map HDR -> RGBA8 if renderer is in Int8 mode. Input is the
        // upscaled display-res HDR when the upscaler ran this frame, the
        // progressive accumulator when offline accumulation is active, the
        // render-res render target otherwise (identical sizes when scale=1).
        auto& tonemapSource = _lastFrameUpscaled ? _upscaledHdr
                            : progressiveOutput  ? *progressiveOutput
                                                 : frame.render_target;
        const uint tonemapW = _lastFrameUpscaled ? _displayWidth  : _width;
        const uint tonemapH = _lastFrameUpscaled ? _displayHeight : _height;

        // FXAA input-copy removal (perf R2 item 13): when the sole enabled
        // AfterToneMap feature will consume the tonemap output this frame,
        // route tonemap into the persistent "fxaa_temp" BYTE4 image — the
        // feature reads the temp and writes the display target directly, so
        // the full-screen copy_to in FxaaFeature::onExecute is skipped.
        // Requires the feature's own readiness (wantsTonemapTempRoute:
        // compiled + enabled) so a non-running feature can never leave the
        // display target unwritten, and exactly one feature so no earlier
        // chain member reads a stale display target. Tonemap dims equal the
        // AfterToneMap point's dims (display res when upscaled).
        const auto& atmFeatures = _features[static_cast<size_t>(FeaturePoint::AfterToneMap)];
        _tonemapRoutesToTemp = renderer.isRGBA8() && atmFeatures.size() == 1u &&
                               atmFeatures.front()->wantsTonemapTempRoute(*this);
        auto& tonemapOut = _tonemapRoutesToTemp
            ? requestTempImage(PixelStorage::BYTE4, tonemapW, tonemapH, "fxaa_temp")
            : frame.display_target;

        if (renderer.isRGBA8()) {
            if (_toneMapMode == ToneMapMode::LUT) {
                profiler.set_pass("Shade/LUT");
                cl << _toneMapLutShader(
                        tonemapOut,
                        tonemapSource,
                        _toneMapExposure,
                        _toneMapGamma,
                        _toneMapLut
                    ).dispatch(tonemapW, tonemapH);
            } else {
                profiler.set_pass("Shade/Tonemap");
                cl << _toneMapBlitShader(
                        tonemapOut,
                        tonemapSource,
                        static_cast<uint>(_toneMapMode),
                        _toneMapExposure,
                        _toneMapGamma
                    ).dispatch(tonemapW, tonemapH);
            }
        }

        // Copy current G-Buffer + normals to prev: NO LONGER DONE — the three
        // buffer pairs ping-pong by handle swap in beginFrame() (perf R2 item
        // 10), removing ~16 B/px read+write and 3 dispatches per frame. The
        // shadow-cache copy below remains (its prev is read mid-frame by the
        // SHARC shadow path, which does not rotate).

#if NT_ALLOW_RASTER_FEATURES
        // Shadow cache temporal copy (skip on accum reset to invalidate stale cache)
        if (!_accumReset) {
            if (_requireExplicitBlit) {
                profiler.set_pass("Shade/shadow reset");
                cl << _blitFltShader(_shadowCachePrev, _shadowCache).dispatch(_width, _height);
            } else {
                cl << _shadowCache.copy_to(_shadowCachePrev);
            }
        }
#endif

        // Flip reservoir indices so prev points to this frame's output for next frame's temporal.
        // Full-res: after spatial, current = spatial output. A second flip makes prev = spatial
        // output, and current = the free slot that next frame's candidate will write into.
        // Half-res: this is the SINGLE flip of the frame — it advances the half-res
        // history so next frame's temporal merges against the temporal+boiling output.
        _passDI.flipReservoir();
        if (_passGI.enabled())
            _passGI.flipReservoir();
        Renderer::stream() << cl.commit();
    }

    _dispatchFeatures(Renderer::stream(), FeaturePoint::AfterToneMap, ctx,
                      renderer.isRGBA8() ? frame.display_target : frame.render_target);

#if NT_ENABLE_SHARC
    // SHARC stats polls ride the tail of the render stream (event-signaled).
    // The former mid-frame polls synchronized this stream from drawUi() — a
    // full CPU stall that also blocked submission of the current frame's
    // work. drawUi() reads the results at the next cadence tick via a
    // host-side event wait that is long satisfied by then.
    if (_passSharc.enabled() && (_frameCount % 30u) == 0u) {
        _passSharc.submitPolls(Renderer::stream());
    }
#endif

    // Signal frame-tail event so compute stream can start deformable mesh updates
    // for the next frame as soon as the render stream finishes its tail work.
    ++_frameTailFence;
    Renderer::stream() << _frameTailEvent.signal(_frameTailFence);

    // Don't synchronize here — defer to next beginFrame() for CPU/GPU overlap
    _frameSubmitted = true;
    _accumReset = false;
    _frameCount++;
}

} // namespace newtype::core
