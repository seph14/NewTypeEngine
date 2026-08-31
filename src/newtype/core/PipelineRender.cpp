#include "newtype/core/Pipeline.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/render/Shading.h"
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
        CI_LOG_W("Pipeline::render called before buildScene() — TLAS is null "
            "and shaders are not compiled; skipping frame to avoid device "
            "removal. Call buildScene() once before the render loop.");
        return;
    }
    if (_frameSubmitted) {
        CI_LOG_W("Pipeline::render called twice without an intervening "
            "beginFrame() — the previous frame's GPU tail is still in flight; "
            "skipping this frame to avoid command-list races.");
        return;
    }
    auto& frame = *_currentFrame;
    DebugTag debugTagResolved = (_debugTag != DebugTag::None) ? _debugTag : debugTag;

    // Check for custom material callable DLL hot-reload (Debug_Runtime only)
#ifdef RT_RUNTIME
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

    // FPS-aware coordination: sync PassGI's accumulation state to PassDI so its
    // visMaxAge derivation in _populateDiParams has up-to-date values (handles
    // user toggling the checkbox in PassGI's UI mid-session).
    _passDI.setVisAgeDerivation(_passGI.giAccumulationTimeEnabled(),
                                _passGI.giAccumulationTime());

    // Build shared frame context
    FrameContext ctx = {
        _gbufDepth, _gbufVis, _gbufBaryMotion, _glassThroughput,
        data, *_geom, *_materialPool, *_lightSampler,
        _frameCount, _width, _height, cbField, _accumReset,
        dt,
        _seedImage,
        _accumBuffer, _specularBuffer,
        _denoiseAlbedo, _denoiseSpecFactor, _denoiseNormal,
        _gbufDepthPrev, _gbufVisPrev, _denoiseNormalPrev,
#if NT_ENABLE_BSSRDF
        _passSSS.sssRadiance(),
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
    // Pass 1.7: SSS Probe (Burley 2015 BSSRDF — gated by NT_ENABLE_BSSRDF).
    // Full-res dispatch; reads G-Buffer + scene and writes per-channel
    // demodulated radiance to _passSSS.sssRadiance(). Runs before presample
    // so the probe's light sampling completes before _rasterStream touches
    // shared light buffers (matches Pass 1.5 rationale).
    // Skipped entirely when no visible instance uses a Subsurface material —
    // matches the glass-dispatch gating pattern.
    //==========================================================================
#if NT_ENABLE_BSSRDF
    if (_geom->has_active_subsurface()) {
        auto cl = CommandList::create();
        profiler.set_pass("SSS/Probe");
        util::CpuScopedTimer _cpu_SSS_Probe("SSS/Probe");
        _passSSS.renderProbe(cl, ctx);
        Renderer::stream() << cl.commit();
    }
#endif

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

    // GPU wait: ensure async feature G-buffer modifications (e.g., PointCloud raster merge)
    // are complete before Candidate reads per-pixel G-buffer data
    if (_featureGbufFence != 0u) {
        Renderer::stream() << _featureGbufEvent.wait(_featureGbufFence);
        _featureGbufFence = 0u;
    }

    //==========================================================================
    // Pass 1.6 + Pass 2: Denoiser Prefilter + DI Candidate (single CommandList)
    // Prefilter writes _denoiseAlbedo/_denoiseNormal; Candidate writes reservoirs
    // and reads G-Buffer. Disjoint outputs, overlapping read-only inputs — safe
    // to batch on the same stream. Saves one stream submission per frame.
    //==========================================================================
    {
        auto cl = CommandList::create();
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
            .dispatch(_checkerboardEnabled ? (_width + 1u) / 2u : _width, _height);
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
                .dispatch(_checkerboardEnabled ? (_width + 1u) / 2u : _width, _height);
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

        // DI/GI Debug checkpoints: AfterSpatial / GIReservoir — must read
        // post-spatial reservoir state, so commit the partial CL (without shade)
        // before the debug blit, then early-return.
#if NT_DEBUG_VIZ
        if (_diDebugPass == DIDebugPass::AfterSpatial) {
            Renderer::stream() << cl.commit();
            Renderer::stream() <<
                _diDebugReservoirShader(frame.display_target, _passDI.reservoirBuffer(), _gbufDepth, _gbufVis, cbField)
                .dispatch(_checkerboardEnabled ? (_width + 1u) / 2u : _width, _height);
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
                .dispatch(_checkerboardEnabled ? (_width + 1u) / 2u : _width, _height);
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
        if (debugTagResolved == DebugTag::Visibility) {
            cl << _debugVisShader(frame.display_target, _gbufVis)
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
                    data,                               // 7: camera
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
                    _passDI.envVisMaxDistance(),        // 22: visibility max spatial distance (env lights)
                    _geom->has_transparent_shadow_casters() ? 1u : 0u  // 23: any-hit fast path gate
#if NT_ALLOW_RASTER_FEATURES
                    ,
                    _voxelGrid->gpu_resources(),         // 24: voxel grid binding group
                    _shadowCachePrev,                     // 25: shadow cache prev (read)
                    _shadowCache                          // 26: shadow cache (write)
#endif
                    ,
                    _passGI.initialSnapshotBuffer(),     // 24/27: GI initial-reservoir snapshot
                    _passGI.giMISRoughness(),  // 25/28: MIS roughness (0=off)
                    _passGI.deltaBranchNEEEnabled() ? 1u : 0u  // 26/28: delta-branch NEE for mirror metals
#if NT_ENABLE_BSSRDF
                    , _passSSS.sssRadiance()
#endif
#if NT_ENABLE_PROCEDURAL
                    , _procBindless
#endif
                )
                .dispatch(_checkerboardEnabled ? (_width + 1u) / 2u : _width, _height);
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
                data,
                _lightSampler->env_exposure(),
                _denoiseAlbedo,
                _denoiseSpecFactor,
                _specularBuffer,
                _solidBackgroundEnabled ? 1u : 0u,
                _solidBackgroundColor
            )
            .dispatch(_width, _height);

        // Copy prev buffers for temporal reuse continuity (batched)
        {
            auto cl = CommandList::create();
            if (_requireExplicitBlit) {
                cl << _blitFltShader(_gbufDepthPrev, _gbufDepth).dispatch(_width, _height);
                cl << _blitUIntShader(_gbufVisPrev, _gbufVis).dispatch(_width, _height);
                cl << _blitFltShader(_denoiseNormalPrev, _denoiseNormal).dispatch(_width, _height);
            } else {
                cl << _gbufDepth.copy_to(_gbufDepthPrev);
                cl << _gbufVis.copy_to(_gbufVisPrev);
                cl << _denoiseNormal.copy_to(_denoiseNormalPrev);
            }
            // Copy reservoirs to prev for next frame's temporal reuse
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
    // Pass 9: Denoiser (ReLAX)
    //==========================================================================
    {
        const Image<float>* rasterDepthPtr = nullptr;
#if NT_ALLOW_RASTER_FEATURES
        if (_rasterContext && _rasterContext->hasResources()) rasterDepthPtr = &_rasterContext->rasterDepth();
#endif
        bool denoiserRendered = _denoiser.render(Renderer::stream(), ctx,
            frame.render_target, debugTagResolved == DebugTag::None, rasterDepthPtr);

        // No-denoiser fallback: composite raw shade output with envmap for sky
        if (!denoiserRendered && debugTagResolved == DebugTag::None) {
            Renderer::stream() <<
                _compositeBlitShader(
                    frame.render_target,
                    _accumBuffer,
                    _gbufDepth,
                    env_gpu_resources(),
                    data,
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
                    data,
                    env_gpu_resources(),
                    _lightSampler->env_exposure(),
                    scene_gpu_resources(),
                    _geom->vertex_bindless(),
                    _geom->tlas()
#if NT_ENABLE_PROCEDURAL
                    , _procBindless
#endif
	            )
                .dispatch(_width, _height);
        }
    }

    _dispatchFeatures(Renderer::stream(), FeaturePoint::AfterGlassTint, ctx, frame.render_target);

    // Tonemap + copy-prev batched into one CommandList to reduce stream submission
    // overhead. Each stream << creates a CommandAllocator with 3MB pre-allocated
    // GPU buffers; batching saves 1-2 submissions per frame.
    {
        auto cl = CommandList::create();
        // Tone-map HDR -> RGBA8 if renderer is in Int8 mode
        if (renderer.isRGBA8()) {
            if (_toneMapMode == ToneMapMode::LUT) {
                profiler.set_pass("Shade/LUT");
                cl << _toneMapLutShader(
                        frame.display_target,
                        frame.render_target,
                        _toneMapExposure,
                        _toneMapGamma,
                        _toneMapLut
                    ).dispatch(_width, _height);
            } else {
                profiler.set_pass("Shade/Tonemap");
                cl << _toneMapBlitShader(
                        frame.display_target,
                        frame.render_target,
                        static_cast<uint>(_toneMapMode),
                        _toneMapExposure,
                        _toneMapGamma
                    ).dispatch(_width, _height);
            }
        }

        // Copy current G-Buffer + normals + reservoirs to prev for next frame
        if (_requireExplicitBlit) {
            profiler.set_pass("Shade/copy");
            cl << _blitFltShader(_gbufDepthPrev, _gbufDepth).dispatch(_width, _height)
               << _blitUIntShader(_gbufVisPrev, _gbufVis).dispatch(_width, _height)
               << _blitFltShader(_denoiseNormalPrev, _denoiseNormal).dispatch(_width, _height);
        } else {
            cl << _gbufDepth.copy_to(_gbufDepthPrev)
               << _gbufVis.copy_to(_gbufVisPrev)
               << _denoiseNormal.copy_to(_denoiseNormalPrev);
        }

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
