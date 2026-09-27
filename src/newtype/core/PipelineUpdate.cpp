#include "newtype/core/Pipeline.h"
#include "newtype/util/Camera.h"
#include "newtype/upscal/Fsr31Backend.h"
#include "newtype/upscal/DlssSrBackend.h"
#include "cinder/Log.h"
#include "cinder/Json.h"
#include "newtype/util/UiHelper.h"
#include "newtype/util/JsonBackup.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include <algorithm>
#include <utility>
#include "cinder/Utilities.h"

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;


//==============================================================================
// beginFrame — sync previous GPU frame + frame buffer swap + material upload
//==============================================================================

void Pipeline::beginFrame(Renderer& renderer) noexcept {
    // 1. Synchronize previous frame's GPU tail (denoiser + tonemap + copy-prev)
    if (_frameSubmitted) {
        Renderer::stream() << synchronize();
        _frameSubmitted = false;
        renderer.markFrameReady();
    }
    // Per-frame reset of the beginUpdate() wait latch (beginFrame always
    // runs between two update phases; resize()'s full synchronize makes a
    // stale-true flag harmless there).
    _updateTailWaited = false;

    // 1b. Video publish window: the GPU is momentarily drained here and this
    // frame's kernels are not yet dispatched — the safe, cheap moment for the
    // media player to write the D3D12-shared video texture (no-op without a
    // hook or when nothing is due).
    if (_videoPublishHook) _videoPublishHook();

    // 1c. Ping-pong rotation of the prev G-buffer set (perf R2 item 10).
    // Replaces the three end-of-frame full-screen copies (gbufDepth/gbufVis/
    // denoiseNormal -> Prev, ~16 B/px read+write + 3 dispatches). Current and
    // prev stay two distinct live images at every instant, so the same-frame
    // current+prev reads in DI/GI temporal reuse are unaffected; nothing
    // reads either set after the old copy point, so "last frame's current"
    // is byte-for-byte what the copy produced. Must run before any
    // FrameContext binds references for the frame (preCtx below, main ctx in
    // render()) — FrameContext holds Image& members. Resize recreates both
    // pair members together, so swap state needs no explicit reset; the
    // _accumReset/_frameCount temporal gates mask garbage on the first frame
    // exactly as they did for un-cleared prev images before.
    std::swap(_gbufDepth, _gbufDepthPrev);
    std::swap(_gbufVis, _gbufVisPrev);
    std::swap(_denoiseNormal, _denoiseNormalPrev);

    // 2. Advance to next frame buffer
    _currentFrame = &renderer.beginFrame();

    // 3. Upload material changes from UI edits
    _materialPool->update(_bufferStream);

    // 4. Upload envmap rotation if changed (yaw/elevation from UI)
    if (_lightSampler->update_env_rotation(_computeStream)) {
        // The rotation buffer is read this frame by Renderer::stream()
        // (presample env / shade / composite). Order it behind the upload and
        // refresh env-presampled candidates — their stored directions and
        // pdf-in-solid-angle depend on the rotation.
        _signalGeomUpdate();
        _passDI.markPresampleEnvDirty();
    }

    // 5. Dispatch PreUpdate features on compute stream
    //    (user GPU sim: VATMesh interpolation, particles, fluids)
    {
        auto& preFeatures = _features[static_cast<size_t>(FeaturePoint::PreUpdate)];
        if (!preFeatures.empty()) {
            util::CameraData dummyCam{};
            FrameContext preCtx = {
                _gbufDepth, _gbufVis, _gbufBaryMotion, _glassThroughput,
                _gbufVelocity, _gbufDepthUpscale,
                dummyCam, *_geom, *_materialPool, *_lightSampler,
                _frameCount, _width, _height, _displayWidth, _displayHeight,
                0u, _accumReset,
                0.0f,
                _seedImage,
                _accumBuffer, _specularBuffer,
                _denoiseAlbedo, _denoiseSpecFactor, _denoiseNormal,
                _gbufDepthPrev, _gbufVisPrev, _denoiseNormalPrev,
                _passSSS.sssRadiance(),
#if NT_ENABLE_SHARC
                _roughGlassInfo,
#endif
            };
            _dispatchFeatures(_computeStream, FeaturePoint::PreUpdate, preCtx, _dummyTarget);
        }
    }
}

//==============================================================================
// Update — deformable mesh BLAS rebuild, animated transforms, light sampler
// Must be called AFTER VATMesh/PointCloud updates on compute stream.
//==============================================================================

void Pipeline::beginUpdate() noexcept {
    // GPU-GPU sync for the SCENE phase: wait for the previous frame's render
    // tail before any scene-update compute dispatch. Render-stream shaders
    // hold in-flight reads of the instance-transform buffers (TetSolve rows)
    // and deformable vertex buffers (update_cpu BLAS rebuilds); dispatching
    // those writes ahead of this wait races the reads (D3D12 debug layer
    // OBJECT_ACCESSED_WHILE_STILL_IN_USE; a mid-trace BLAS rewrite can take
    // the device down). Stream order means the wait must precede the writes
    // in the queue — hence a separate entry point called before scene
    // update(), not the wait that used to sit at the top of update().
    if (_frameSubmitted && !_updateTailWaited) {
        _computeStream << _frameTailEvent.wait(_frameTailFence);
        _updateTailWaited = true;
    }
}

void Pipeline::update(float time, float dt) noexcept {
    // GPU-GPU sync: wait for previous frame's render tail to complete before
    // touching geometry resources (BLAS, vertex buffers, TLAS). Fallback for
    // callers that don't pair beginUpdate() with update() — beginUpdate()
    // already enqueued the wait ahead of the scene-phase dispatches.
    if (_frameSubmitted && !_updateTailWaited) {
        _computeStream << _frameTailEvent.wait(_frameTailFence);
        _updateTailWaited = true;
    }

    // Flush InstancedMesh batch transforms (set_transforms /
    // apply_gpu_transforms) into Geometry before its update consumes them.
    for (auto *mesh : _instancedMeshes)
        mesh->flush_transform_updates(*_geom);

    bool proc_changed = false;
#if NT_ENABLE_PROCEDURAL
    // Per-frame procedural geometry update (animation + AABB refit)
    if (_procGeom)
        proc_changed = _procGeom->update(_computeStream, time);
#endif

    // Bridge material structure changes (type/override/alphacut) to Geometry's
    // deferred material-flag recompute. UI/JSON edits to MaterialData bypass
    // Geometry's own setters, so without this the flags never flip.
    if (_materialPool->structureDirty()) {
        _geom->mark_has_visible_glass_dirty();
        _geom->mark_has_active_subsurface_dirty();
        _materialPool->clearStructureDirty();
    }

    // Update geometry: deformable meshes, animated transforms, pending BLAS builds
    bool requireSync = _geom->update(_computeStream, proc_changed, time);

    // Re-upload instance buffer: props (uint4) only when materials/bindless changed,
    // transforms (float4x4) only when transforms changed — avoids O(N) full upload
    if (_geom->props_dirty()) {
        requireSync = true;
        _geom->upload_instance_props(_computeStream);
    }
    // Per-instance custom data rows (track B2) follow the same lifecycle,
    // but flush independently — a value edit alone doesn't dirty the uint4
    // props rows. Runs AFTER upload_instance_props so a structure change's
    // row reshuffle is folded into the same upload.
    if (_geom->instance_params_dirty()) {
        requireSync = true;
        _geom->upload_instance_params(*_materialPool, _computeStream);
    }
    if (_geom->transforms_dirty() || _geom->transform_prev_stale()) {
        requireSync = true;
        _geom->upload_dirty_transforms(_computeStream);
    }

    // Rebuild light sampler if emissive shapes changed visibility/were removed
    // or if emissive material properties were modified at runtime
    bool geomLightsDirty = _geom->lights_need_rebuild();
    bool geomLightsVisibilityDirty = _geom->lights_visibility_dirty();
    bool matLightsDirty  = _materialPool->lightsNeedRebuild();

    if (geomLightsDirty) {
        // Topology changed (light added/removed): full rebuild — rescan all
        // geometry, recompute areas. Reset accumulation so ReSTIR reservoirs
        // don't hold stale samples referencing removed lights.
        Renderer::stream() << synchronize();
        _lightSampler->rebuild(_computeStream, *_geom, *_materialPool);
        requestAccumReset();
        _geom->clear_lights_dirty();
        _geom->clear_lights_visibility_dirty();
        _materialPool->clearLightsNeedRebuild();
        // Alias table / triangle data rewritten: presampled local candidates
        // carry stale light indices and pdfs until regenerated. Env CDF is
        // rebuilt by rebuild() when the envmap changed with topology.
        _passDI.markPresampleLocalDirty();
        _passDI.markPresampleEnvDirty();
        // Make this frame's rewrites visible to this frame's render (GPU-GPU;
        // was a trailing _computeStream << synchronize()).
        _signalGeomUpdate();
    } else if (matLightsDirty || geomLightsVisibilityDirty) {
        // Emission value OR light visibility toggled: cheap weight refresh,
        // no geometry rescan. _triangle_lights / _alias_table are read by
        // shade shaders on Renderer::stream() (a separate DX12 compute queue);
        // wait for the previous frame's last read to complete before rewriting
        // them on _computeStream. The trailing signal orders this frame's
        // writes against this frame's render (render() waits on the event) —
        // cross-queue visibility via the timeline fence instead of the former
        // CPU synchronize().
        if (_lightSamplerReadyFence != 0u) {
            _computeStream << _lightSamplerReadyEvent.wait(_lightSamplerReadyFence);
        }
        _lightSampler->update_weights(_computeStream, *_materialPool, *_geom);
        _materialPool->clearLightsNeedRebuild();
        _geom->clear_lights_visibility_dirty();
        // Stored pdfs in presampled local candidates reflect the old alias
        // table weights — regenerate.
        _passDI.markPresampleLocalDirty();
        _signalGeomUpdate();
        // No requestAccumReset() here: the shade shader guards stale-reservoir
        // references to hidden lights via `tri_light.pdf > 0.0f` (see
        // PipelineInit.cpp DI reservoir shade). Reset would cause flicker;
        // the shader check gives a correct zero-contribution without it.
        // Emission-only changes (matLightsDirty && !visibility) also converge
        // smoothly — reservoir samples stay valid, intensity refreshes via
        // update_weights.
    } else if (_geom->light_transforms_dirty()) {
        // Light shape transform changed: update vertex positions (and areas if scaled)
        bool scaleChanged = _geom->light_scale_dirty();
        _lightSampler->update_transforms(_computeStream, *_geom, *_materialPool, scaleChanged);
        _geom->clear_light_transforms_dirty();
        //requestAccumReset();
        // Area changes on scale rewrite pdf/area; be conservative for pure
        // translation too (regeneration is cheap and rare).
        _passDI.markPresampleLocalDirty();
        // The requireSync signal from the transform upload path below orders
        // these writes against this frame's render.
    }

    if (requireSync && !geomLightsDirty && !matLightsDirty && !geomLightsVisibilityDirty) {
        // GPU-GPU ordering only: render() waits on _geomUpdateEvent before its
        // first dispatch, so BLAS/TLAS builds and instance-buffer uploads can
        // still be in flight on the GPU without stalling the CPU here (this
        // used to be _computeStream << synchronize() every animated frame).
        _signalGeomUpdate();
    }
}

//==============================================================================
// Runtime Shape Manipulation
//==============================================================================

void Pipeline::setShapeTransform(scene::ShapeId id, const float4x4 &matrix,
                                  scene::Change changeHint) noexcept {
    _geom->set_instance_transform(id, matrix, changeHint);
}

void Pipeline::setShapeVisibility(scene::ShapeId id, bool visible) noexcept {
    _geom->set_visibility(id, visible);
}

void Pipeline::setShapeCameraVisibility(scene::ShapeId id, bool camera_visible) noexcept {
    _geom->set_camera_visibility(id, camera_visible);
}

void Pipeline::setShapeMaterial(scene::ShapeId id, uint32_t material_layers) noexcept {
    _geom->set_material_layers(id, material_layers);
}

void Pipeline::setInstanceUserData(scene::ShapeId id, uint slot, luisa::float4 value) noexcept {
    _geom->set_instance_user_param(id, slot, value);
}

luisa::float4 Pipeline::instanceUserData(scene::ShapeId id, uint slot) const noexcept {
    return _geom->instance_user_param(id, slot);
}

void Pipeline::unloadStaticMeshCPUData() noexcept {
    _geom->unload_static_cpu_data();
}

bool Pipeline::removeShape(scene::ShapeId id) noexcept {
    return _geom->remove_shape(id);
}

MeshShape* Pipeline::getShape(scene::ShapeId id) noexcept {
    return _geom->get_shape(id);
}

Transform* Pipeline::getShapeTransform(scene::ShapeId id) noexcept {
    return _geom->get_transform(id);
}

DeformableMesh* Pipeline::getDeformable(scene::ShapeId id) noexcept {
    return _geom->get_deformable(id);
}

uint Pipeline::geometryTlasRow(scene::ShapeId id) const noexcept {
    return _geom->tlas_index_of(id);
}

bool Pipeline::registerGpuTransformRows(luisa::span<const scene::ShapeId> ids) noexcept {
    return _geom->register_gpu_transform_rows(ids);
}

void Pipeline::notifyGpuTransformsDirty() noexcept {
    _geom->notify_gpu_transforms_dirty();
}

uint Pipeline::topologyGeneration() const noexcept {
    return _geom->topology_generation();
}

//==============================================================================
// Resize
//==============================================================================

void Pipeline::resize(uint width, uint height) {
    // Cinder fires an unconditional resize() right after setup()
    // (AppImplMswBasic::run), and WM_SIZE can repeat the last size — skip
    // the full image recreate when nothing changed. Startup allocation
    // happens in the constructor plus addFeature's onResize call. Render
    // dims are compared too: setRenderScale() calls resize() with unchanged
    // display dims after the renderer's render dims changed, and that call
    // must still run.
    if (width == _displayWidth && height == _displayHeight &&
        (_renderer == nullptr ||
         (_renderer->renderWidth() == _width && _renderer->renderHeight() == _height)))
        return;
    // Mid-frame resize: beginFrame() was called but render() hasn't completed
    // yet. _currentFrame holds a live frame resource; recreating images now
    // invalidates the in-flight command list.
    if (_currentFrame != nullptr && !_frameSubmitted) {
        CI_LOG_W("Pipeline::resize called between beginFrame() and render() - "
            "recreating images mid-frame invalidates the live command list. "
            "Call resize() outside the frame loop.");
    }
    // Drain ALL streams before destroying images. Renderer::stream() alone is
    // not enough: deformable-mesh / sim updates ride _computeStream into the
    // next frame by design (frame-tail event), and uploads ride _bufferStream.
    // Destroying images under in-flight compute work caused GPU page faults →
    // TDR → DXGI_ERROR_DEVICE_REMOVED on later texture creation (crash log
    // 2026-09-16, docs/upscaling_feasibility_report.md implementation record).
    Renderer::stream() << synchronize();
    _computeStream       << synchronize();
    _bufferStream        << synchronize();
    // The raster feature stream was missing from this envelope: point-cloud /
    // trail merges and the voxel passes ride it, and their buffers/textures
    // are destroyed+recreated right below (feature->onResize) — destroying
    // raster resources under in-flight raster work is the same
    // destroy-while-in-flight TDR family the copy-queue fix addressed.
#if NT_ALLOW_RASTER_FEATURES
    if (_rasterContext) _rasterContext->rasterStream() << synchronize();
#endif
    _frameSubmitted = false;

    // Args are DISPLAY dimensions; render dims derive from the render scale
    // (Renderer owns the rounding — it was resized first with the same
    // display size, or the scale changed under an unchanged display size).
    _displayWidth  = width;
    _displayHeight = height;
    if (_renderer != nullptr) {
        _width  = _renderer->renderWidth();
        _height = _renderer->renderHeight();
    } else {
        _width  = width;
        _height = height;
    }
    _createImages    (_width, _height);
    _initSeedImage   ();
    _denoiser.resetFrameIdx();

    // Upscaler context sizes are baked in at creation — recreate. The stream
    // sync above covers the destroy half; the upscaler history restarts from
    // the reset below.
    if (_upscalerBackend != nullptr)
        _upscalerBackend->resize(_width, _height, _displayWidth, _displayHeight);
    requestUpscalerReset();

    // RR feature sizes are baked at creation too — release here (streams are
    // drained above) and let the next dispatch lazily recreate at the new size.
    if (_denoiserMode == DenoiserMode::DlssRR)
        _rrDenoiser.resize(_width, _height);

    // Resize all registered features
    auto& device = Renderer::device();
    for (auto& featureList : _features)
        for (auto& feature : featureList)
            feature->onResize(device, _width, _height);

    // Recreate temp image pool at new dimensions
    for (auto& entry : _tempImages) {
        entry.width = _width;
        entry.height = _height;
        entry.image = device.create_image<float>(entry.format, _width, _height);
    }

    requestAccumReset();
    // Resize recreated the progressive accumulator pool images — restart the
    // running average.
    requestProgressiveReset();
}

//==============================================================================
// Upscaler control (docs/upscaling_feasibility_report.md — Phase 0/1)
//==============================================================================

bool Pipeline::upscalerActive() const noexcept {
    if (_upscalerMode == upscal::UpscalerMode::None || _upscalerBackend == nullptr)
        return false;
    // The upscale output path (upscaled HDR -> tonemap -> BYTE4 display)
    // only exists in RGBA8 mode; float display modes render native.
    if (_renderer != nullptr && !_renderer->isRGBA8())
        return false;
    return _upscalerBackend->available();
}

void Pipeline::setUpscalerMode(upscal::UpscalerMode mode) {
    if (mode == _upscalerMode && (mode == upscal::UpscalerMode::None || upscalerActive()))
        return;

    // Same safety envelope as resize(): no in-flight GPU work may touch the
    // context being destroyed.
    // ffxDestroyContext (shutdown of the previous backend) requires a fully
    // idle GPU — drain all streams, not just the render stream (see resize()).
    Renderer::stream() << synchronize();
    _computeStream       << synchronize();
    _bufferStream        << synchronize();
#if NT_ALLOW_RASTER_FEATURES
    if (_rasterContext) _rasterContext->rasterStream() << synchronize();
#endif
    _frameSubmitted = false;

    if (mode == upscal::UpscalerMode::Fsr3) {
        auto backend = std::make_unique<upscal::Fsr31Backend>();
        if (!backend->init(Renderer::device(), _width, _height, _displayWidth, _displayHeight)) {
            CI_LOG_E("Upscaler: FSR 3.1 backend unavailable (DLLs missing or context "
                     "creation failed) - staying at None. Deploy "
                     "external/FidelityFX/bin/*.dll next to the executable.");
            mode = upscal::UpscalerMode::None;
        } else {
            _upscalerBackend = std::move(backend);
        }
    } else if (mode == upscal::UpscalerMode::Dlss) {
        auto backend = std::make_unique<upscal::DlssSrBackend>();
        if (!backend->init(Renderer::device(), _width, _height, _displayWidth, _displayHeight)) {
            CI_LOG_E("Upscaler: DLSS backend unavailable (no RTX GPU / NGX-capable "
                     "driver, or DLSS-SR feature unsupported). Staying at None; "
                     "FSR 3.1 remains available.");
            mode = upscal::UpscalerMode::None;
        } else {
            _upscalerBackend = std::move(backend);
            // DLSS is designed for jittered input — let the Halton pattern
            // through by default (the checkbox below still overrides; resets
            // are already requested at the end of this function).
            if (!_fsrJitterEnabled) {
                _fsrJitterEnabled = true;
                CI_LOG_I("Upscaler: DLSS selected - input jitter enabled by "
                         "default (DLSS expects jittered input).");
            }
        }
    } else {
        _upscalerBackend.reset();
    }

    _upscalerMode = mode;
    _upscalerUnavailableLogged = false;

    // Only the display-res output image depends on the mode — allocate /
    // release it directly instead of a full _createImages pass.
    auto& device = Renderer::device();
    if (_upscalerMode != upscal::UpscalerMode::None) {
        _upscaledHdr = device.create_image<float>(PixelStorage::FLOAT4,
                                                  _displayWidth, _displayHeight);
        _upscaledHdr.set_name("upscaled_hdr");
    } else {
        _upscaledHdr = {};
    }
    requestUpscalerReset();
    requestAccumReset();
}

//==============================================================================
// Denoiser mode control (DLSS-RR replace-ReLAX seam — the feasibility
// report's "resolve-stage integration": RR consumes the noisy AfterShade
// signals at Pass 9 and writes denoised HDR to the render target, so OIT /
// glass / tonemap / upscaler are unaffected.)
//==============================================================================

bool Pipeline::rrDenoiserActive() const noexcept {
    if (_denoiserMode != DenoiserMode::DlssRR || !_denoiser.enabled())
        return false;
    return _rrDenoiser.available();
}

void Pipeline::setDenoiserMode(DenoiserMode mode) {
    if (mode == _denoiserMode &&
        (mode == DenoiserMode::Relax || rrDenoiserActive()))
        return;

    // Same envelope as setUpscalerMode: feature release requires idle GPU.
    Renderer::stream() << synchronize();
    _computeStream       << synchronize();
    _bufferStream        << synchronize();
#if NT_ALLOW_RASTER_FEATURES
    if (_rasterContext) _rasterContext->rasterStream() << synchronize();
#endif
    _frameSubmitted = false;

    // Leave the previous mode's GI decorrelation state behind cleanly.
    if (_denoiserMode == DenoiserMode::DlssRR && _giDecorrelationPreRr.has_value()) {
        _passGI.applyGiDecorrelationSettings(*_giDecorrelationPreRr);
        _giDecorrelationPreRr.reset();
        CI_LOG_I("Denoiser: leaving DLSS-RR - restored the prior GI "
                 "decorrelation settings.");
    }
    // ... and the checkerboard state (RR contract: dense per-pixel input —
    // the DLSS-RR Integration Guide §3.5 lists checkerboard rendering among
    // the practices to avoid; half the pixels would otherwise carry
    // one-frame-stale samples from the previous jitter phase, which RR
    // smooths into blur and structured artifacts). The flip is applied at
    // the render-thread safe point (see _applyCheckerboardReconfigure).
    if (_denoiserMode == DenoiserMode::DlssRR && _checkerboardPreRr.has_value()) {
        _checkerboardEnabled = *_checkerboardPreRr;
        _checkerboardPreRr.reset();
        CI_LOG_I("Denoiser: leaving DLSS-RR - restoring checkerboard rendering.");
    }

    if (mode == DenoiserMode::DlssRR) {
        if (!_rrDenoiser.init(Renderer::device(), _width, _height)) {
            CI_LOG_E("Denoiser: DLSS-RR unavailable (no RTX GPU / NGX-capable "
                     "driver, or RR feature unsupported) - staying on ReLAX.");
            mode = DenoiserMode::Relax;
        } else {
            // RTXDI 3.1 compatibility preset: decorrelate the temporally
            // reused GI samples so RR sees fresh noise (RTXDI's own DLSS-RR
            // preset semantics); remember the prior values for restore.
            _giDecorrelationPreRr = _passGI.giDecorrelationSettings();
            _passGI.applyRrCompatPreset();
            CI_LOG_I("Denoiser: DLSS-RR engaged - applied the GI RR-style "
                     "decorrelation preset (prior settings restorable by "
                     "switching back).");
            if (_rrDenoiser.available()) {
                // Only when the feature can actually run (dims validated in
                // init) — otherwise ReLAX keeps its checkerboard input.
                _checkerboardPreRr = _checkerboardEnabled;
                if (_checkerboardEnabled) {
                    _checkerboardEnabled = false;
                    CI_LOG_I("Denoiser: DLSS-RR engaged - checkerboard "
                             "rendering disabled for dense per-pixel input "
                             "(recompile applies on the next frame; restored "
                             "when switching back).");
                }
            }
        }
    } else {
        _rrDenoiser.shutdown();
    }

    _denoiserMode = mode;
    _denoiserModeUnavailableLogged = false;

    // RR working images only exist while the mode is active.
    auto& device = Renderer::device();
    if (_denoiserMode == DenoiserMode::DlssRR) {
        _rrColor           = device.create_image<float>(PixelStorage::HALF4, _width, _height);
        _rrAlbedo          = device.create_image<float>(PixelStorage::HALF4, _width, _height);
        _rrF0              = device.create_image<float>(PixelStorage::HALF4, _width, _height);
        _rrNormalRoughness = device.create_image<float>(PixelStorage::HALF4, _width, _height);
        _rrColor.set_name("rr_color");
        _rrAlbedo.set_name("rr_albedo");
        _rrF0.set_name("rr_f0");
        _rrNormalRoughness.set_name("rr_normal_roughness");
    } else {
        _rrColor = {};
        _rrAlbedo = {};
        _rrF0 = {};
        _rrNormalRoughness = {};
    }
    requestUpscalerReset();
    requestAccumReset();
}

void Pipeline::setRenderScale(float scale) {
    scale = std::clamp(scale, 1.0f / 3.0f, 1.0f);
    if (scale < 1.0f) {
        if (_renderer != nullptr && _renderer->renderSizeOverride().has_value()) {
            CI_LOG_W("renderScale < 1 is unavailable while a render-size override is "
                     "active (non-perspective projection) - clamping to 1.0");
            scale = 1.0f;
        } else if (_renderer != nullptr && !_renderer->isRGBA8()) {
            CI_LOG_W("renderScale < 1 requires RGBA8 display mode - clamping to 1.0");
            scale = 1.0f;
        } else if (_upscalerMode == upscal::UpscalerMode::None) {
            CI_LOG_W("renderScale < 1 requires an active upscaler - clamping to 1.0");
            scale = 1.0f;
        }
    }
    if (scale == _renderScale)
        return;

    _renderScale = scale;
    if (_renderer != nullptr)
        _renderer->setRenderScale(scale); // syncs + recreates frame resources

    // Full pipeline image recreation at the new render dims (display size
    // unchanged) — resize() also resizes the upscaler context and requests
    // the history reset.
    resize(_displayWidth, _displayHeight);
}

void Pipeline::setRenderResolutionOverride(std::optional<luisa::uint2> size) {
    if (_renderer == nullptr)
        return;
    const auto current = _renderer->renderSizeOverride();
    const bool same = (size.has_value() == current.has_value()) &&
        (!size.has_value() ||
         (size->x == current->x && size->y == current->y));
    if (same)
        return;

    // Renderer recreates its frame resources at the override size; resize()
    // below rebuilds every pipeline image at the renderer's new render dims.
    // The upscaler path is meaningless with a decoupled aspect — FSR's fov /
    // NDC-depth contract only holds for the plain perspective camera.
    _renderer->setRenderSizeOverride(size);
    if (size.has_value() && _upscalerMode != upscal::UpscalerMode::None)
        setUpscalerMode(upscal::UpscalerMode::None);
    resize(_displayWidth, _displayHeight);
}

void Pipeline::setProgressiveAccumulation(bool enabled) noexcept {
    if (enabled == _progressiveAccum)
        return;
    _progressiveAccum = enabled;
    _progressiveResetPending = true;
    if (enabled) {
        // Progressive export wants raw samples: the denoiser's own temporal
        // accumulation would double-smooth and its reconstruction assumes a
        // pinhole frustum for the non-perspective projections anyway.
        _denoiser.setEnabled(false);
        CI_LOG_I("Progressive accumulation enabled (denoiser forced off)");
    }
}

void Pipeline::setFsrJitterEnabled(bool enabled) noexcept {
    if (enabled == _fsrJitterEnabled) return;
    _fsrJitterEnabled = enabled;
    // The jitter pattern change invalidates both temporal chains.
    requestUpscalerReset();
    requestAccumReset();
}

bool Pipeline::_consumeUpscalerReset(const FrameContext& ctx) {
    bool reset = _upscalerReset || ctx.accumReset;
    // Teleport heuristic: a single-frame camera jump the reprojection cannot
    // bridge (> 1 world unit — pan/orbit speeds stay far below this).
    if (_upscalerHavePrevCamPos) {
        auto d = ctx.camera.position - _upscalerPrevCamPos;
        if (dot(d, d) > 1.0f)
            reset = true;
    }
    _upscalerPrevCamPos = ctx.camera.position;
    _upscalerHavePrevCamPos = true;
    _upscalerReset = false;
    return reset;
}

//==============================================================================
// UI
//==============================================================================

void Pipeline::drawUi() {
    // Snapshot GI enabled state to detect toggle. PassGI::drawUi binds the
    // checkbox directly to its _enabled field, so we observe the change here.
    // Resetting accum on toggle prevents the first post-toggle frame from
    // temporal-merging against stale prev-slot reservoirs.
    const bool giWasEnabled = _passGI.enabled();
    _passDI.drawUi();
    _passGI.drawUi();
    if (_passGI.enabled() != giWasEnabled)
        requestAccumReset();
    _passSSS.drawUi();
#if NT_ENABLE_SHARC
    _passSharc.drawUi();
    // Occupancy readback is submitted at the END of render() (event-signaled,
    // no stream sync); read the results here at a low cadence. The query
    // hit-rate read rides the same cadence (counters advance whenever the
    // cache is enabled — the query consumers are always live then).
    if (_passSharc.enabled() && (_frameCount % 30u) == 0u) {
        _passSharc.readPolls();
    }
#endif

    // this should be constexpr
    //ImGui::Checkbox("Checkerboard", &_checkerboardEnabled);

    // Zeroes CameraData::jitter/prev_jitter at the Pipeline::render boundary —
    // deterministic primaries, no shader recompiles (docs/primary_jitter_plan.md).
    ImGui::Checkbox("Primary jitter (TAA)", &_primaryJitterEnabled);

    // Offline export path: running average of raw frames
    // (docs/non_perspective_camera_report.md Phase 1).
    {
        bool prog = _progressiveAccum;
        if (ImGui::Checkbox("Progressive accumulation (offline)", &prog))
            setProgressiveAccumulation(prog);
        if (ImGui::IsItemHovered())
            ImGui::SetItemTooltip("Running average of raw (undenoised) frames for noise-free stills.\n"
                                  "Forces the denoiser off; resets on camera motion / resize.");
        if (_progressiveAccum) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%u frames)", _progressiveFrame);
            if (ImGui::Button("Reset accumulation"))
                requestProgressiveReset();
        }
    }

    // Upscaler: HDR render-res -> display-res, pre-tonemap (Phase 0/1 —
    // docs/upscaling_feasibility_report.md). Mode/backend switches and scale
    // changes recreate GPU resources, so scale applies on slider release.
    if (ImGui::CollapsingHeader("Upscaler")) {
        static constexpr const char* kUpscalerNames[] = { "None", "FSR 3.1", "DLSS 4.5" };
        int mode = static_cast<int>(_upscalerMode);
        if (ImGui::Combo("Backend", &mode, kUpscalerNames, 3))
            setUpscalerMode(static_cast<upscal::UpscalerMode>(mode));

        const bool hasUpscaler = _upscalerMode != upscal::UpscalerMode::None;
        float scale = hasUpscaler ? _renderScale : 1.0f;
        if (ImGui::SliderFloat("Render scale", &scale, 1.0f / 3.0f, 1.0f, "%.3f",
                               ImGuiSliderFlags_AlwaysClamp)) {
            _pendingRenderScale = scale;
        }
        if (ImGui::IsItemDeactivatedAfterEdit() && _pendingRenderScale.has_value()) {
            setRenderScale(*_pendingRenderScale);
            _pendingRenderScale.reset();
        }
        ImGui::SliderFloat("Sharpness (RCAS)", &_upscalerSharpness, 0.0f, 1.0f, "%.2f");

        // perf R2 item 14: upscaler input jitter + FXAA upscaler guard.
        bool fsrJit = _fsrJitterEnabled;
        if (ImGui::Checkbox("Upscaler input jitter", &fsrJit))
            setFsrJitterEnabled(fsrJit);
        if (ImGui::IsItemHovered())
            ImGui::SetItemTooltip("Enable the Halton primary jitter while rendering at renderScale < 1.\n"
                                  "Temporal upscalers resolve the pattern at display res (DLSS expects\n"
                                  "jittered input; it is forced on when DLSS is selected).\n"
                                  "Independent of the full-res 'Primary jitter (TAA)' toggle.");
        bool fxaaSkip = _fxaaSkipWhenUpscaled;
        if (ImGui::Checkbox("Skip FXAA when upscaled", &fxaaSkip))
            setFxaaSkipWhenUpscaled(fxaaSkip);
        if (ImGui::IsItemHovered())
            ImGui::SetItemTooltip("Skip the post-tonemap FXAA pass on upscaled frames (renderScale < 1):\n"
                                  "FSR already reconstructs and RCAS-sharpens — FXAA on top is a\n"
                                  "display-res double-AA pass.");

        // Quality presets (per-dimension ratios: FSR from ffx_upscale.h,
        // DLSS from the DLSS Programming Guide optimal settings).
        if (hasUpscaler) {
            ImGui::TextDisabled("presets:");
            ImGui::SameLine();
            const bool dlss = _upscalerMode == upscal::UpscalerMode::Dlss;
            static constexpr std::pair<const char*, float> kFsrPresets[] = {
                {"NativeAA 1.0", 1.0f}, {"Quality .75", 3.f / 4.f},
                {"Balanced .677", 1.f / 1.5f}, {"Perf .588", 1.f/1.7f},
            };
            static constexpr std::pair<const char*, float> kDlssPresets[] = {
                {"DLAA 1.0", 1.0f}, {"Quality .67", 2.f / 3.f},
                {"Balanced .58", 0.58f}, {"Perf .5", 0.5f},
            };
            for (auto& [label, s] : dlss ? kDlssPresets : kFsrPresets) {
                if (ImGui::Button(label)) setRenderScale(s);
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }
        ImGui::TextDisabled("%s", upscalerActive()
            ? _upscalerBackend->name()
            : "inactive (bilinear fallback)");
    }

    // Denoiser backend: ReLAX (default) or DLSS Ray Reconstruction (RTX).
    // Replaces the Pass-9 denoiser 1:1 at render res; ReLAX's prefilter still
    // runs either way (it prepares the shared albedo/normal signals).
    {
        static constexpr const char* kDenoiserNames[] = { "ReLAX", "DLSS 4.5 Ray Reconstruction" };
        int dm = static_cast<int>(_denoiserMode);
        if (ImGui::Combo("Denoiser backend", &dm, kDenoiserNames, 2))
            setDenoiserMode(static_cast<DenoiserMode>(dm));
        if (ImGui::IsItemHovered())
            ImGui::SetItemTooltip("DLSS-RR replaces the denoiser: it consumes the noisy shade output\n"
                                  "(plus albedo/F0/normal guides) at render resolution and denoises with\n"
                                  "the DLSS 4.5 transformer model. RTX GPUs only (driver 580.00+);\n"
                                  "the GI decorrelation preset is applied/restored automatically.");
        if (_denoiserMode == DenoiserMode::DlssRR && !rrDenoiserActive() &&
            !_denoiserModeUnavailableLogged) {
            ImGui::TextDisabled("%s", "DLSS-RR inactive (NGX unavailable?) - ReLAX fallback");
            _denoiserModeUnavailableLogged = true;
        }
    }

    _denoiser.drawUi();

    // Sky / Environment light controls
    if (ImGui::CollapsingHeader("Sky")) {
        float yaw_deg = _lightSampler->env_yaw() * 180.0f / 3.14159265f;
        float elev_deg = _lightSampler->env_elevation() * 180.0f / 3.14159265f;
        float env_exp = _lightSampler->env_exposure();

        bool changed = false;
        if (ImGui::SliderFloat("Yaw", &yaw_deg, -180.0f, 180.0f, "%.1f deg")) {
            _lightSampler->set_env_yaw(yaw_deg * 3.14159265f / 180.0f);
            changed = true;
        }
        if (ImGui::SliderFloat("Elevation", &elev_deg, 0.0f, 90.0f, "%.1f deg")) {
            _lightSampler->set_env_elevation(elev_deg * 3.14159265f / 180.0f);
            changed = true;
        }
        // Min 0.0 = true env-off (HDR envmaps with values in the hundreds
        // remain visible at small nonzero exposure).
        if (ImGui::SliderFloat("Env Exposure", &env_exp, 0.0f, 1.0f, "%.3f")) {
            _lightSampler->set_env_exposure(env_exp);
        }

        if (_lightSampler->has_procedural_sky()) {
            float turbidity = _lightSampler->env_turbidity();
            if (ImGui::SliderFloat("Turbidity", &turbidity, 1.0f, 10.0f, "%.1f")) {
                _lightSampler->set_env_turbidity(turbidity);
                _lightSampler->regenerate_procedural_sky(_computeStream);
                // Envmap pixels + CDF rewritten: env presampled candidates
                // are stale until regenerated.
                _passDI.markPresampleEnvDirty();
                // compressTexture self-syncs the compute stream, but the CDF
                // buffer uploads it queues after run unsynced — order this
                // frame's render-stream env consumers (presample regen reads
                // the CDF and bakes results into tiles that persist) behind
                // the regeneration, same as the rotation-buffer path in
                // beginFrame.
                _signalGeomUpdate();
            }
        }

        if (changed)
            requestAccumReset();

        ImGui::Checkbox("Solid Background", &_solidBackgroundEnabled);
        if (_solidBackgroundEnabled) {
            float color[3] = { _solidBackgroundColor.x, _solidBackgroundColor.y, _solidBackgroundColor.z };
            if (ImGui::ColorEdit3("Background Color", color)) {
                _solidBackgroundColor = luisa::make_float3(color[0], color[1], color[2]);
            }
        }
    }

    // Tone mapping controls
    if (ImGui::CollapsingHeader("Tone Mapping")) {
        const char* toneMapNames[] = { "ACES", "Hejl", "Reinhard", "Lottes", "Uchimura", "LUT" };
        int toneMapIdx = static_cast<int>(_toneMapMode);
        if (ImGui::Combo("Curve", &toneMapIdx, toneMapNames, IM_ARRAYSIZE(toneMapNames)))
            _toneMapMode = static_cast<ToneMapMode>(toneMapIdx);
        ImGui::SliderFloat("Exposure", &_toneMapExposure, 0.1f, 10.0f);
        ImGui::SliderFloat("Gamma", &_toneMapGamma, 1.0f, 3.0f);

        if (_toneMapMode == ToneMapMode::LUT) {
            if (ImGui::Button("Load LUT...")) {
                auto result = ci::app::getOpenFilePath({}, { "cube" });
                if (!result.empty())
                    _loadToneMapLut(result);
            }
            if (!_toneMapLutPath.empty()) {
                auto fname = std::filesystem::path(_toneMapLutPath).filename().string();
                ImGui::SameLine();
                ImGui::Text("Loaded: %s", fname.c_str());
            }
        }
    }

    util::Profiler::instance().draw_ui();

#if NT_DEBUG_VIZ
    if (ImGui::CollapsingHeader("Debug")) {
        ImGui::SliderInt("Shade Debug Viz (0=off, 9=GI reservoir, 11=GI vis, 13=GI disocclusion dx)", &_shadeDebugVizMode, 0, 13);

        // Debug visualization mode ("Dispersion Rings" = per-channel
        // first-interface TIR topology; needs SHARC + dispersion builds)
        const char* debugTagNames[] = { "None", "Depth", "Visibility", "BaryCentric", "Motion", "PDF", "Jacobian", "Normal", "Glass", "GIReservoir", "Dispersion Rings" };
        int debugTagIdx = static_cast<int>(_debugTag);
        if (ImGui::Combo("Debug Tag", &debugTagIdx, debugTagNames, IM_ARRAYSIZE(debugTagNames))) {
            _debugTag = static_cast<DebugTag>(debugTagIdx);
            requestAccumReset();
        }

        // DI Debug: per-pass reservoir visualization
        const char* diDebugNames[] = { "Off", "AfterCandidate", "AfterTemporal", "AfterBoiling", "AfterSpatial", "AfterShade" };
        int diDebugIdx = static_cast<int>(_diDebugPass);
        if (ImGui::Combo("DI Debug Pass", &diDebugIdx, diDebugNames, IM_ARRAYSIZE(diDebugNames))) {
            _diDebugPass = static_cast<DIDebugPass>(diDebugIdx);
            requestAccumReset();
        }
    }
#endif

    // Config save/load
    if (ImGui::CollapsingHeader("Config")) {
        if (ImGui::Button("Save Config")) {
            auto path = ci::app::getAssetPath("") / "config.json";
            saveConfig(path);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Config")) {
            auto path = ci::app::getAssetPath("") / "config.json";
            loadConfig(path);
        }
    }
}

//==============================================================================
// Config
//==============================================================================

ci::Json Pipeline::captureConfig() const {
    ci::Json config;

    // Pipeline-level
    {
        ci::Json p;
        p["checkerboardEnabled"]     = _checkerboardEnabled;
        p["primaryJitterEnabled"]    = _primaryJitterEnabled;
        p["boilingFilterStrength"]   = _boilingFilterStrength;
        p["toneMapMode"]             = static_cast<uint>(_toneMapMode);
        p["toneMapExposure"]         = _toneMapExposure;
        p["toneMapGamma"]            = _toneMapGamma;
        p["toneMapLutPath"]          = _toneMapLutPath;
        p["denoiserEnabled"]         = _denoiser.enabled();
        p["denoiserMode"]            = static_cast<uint>(_denoiserMode);
        p["giEnabled"]               = _passGI.enabled();
        p["upscalerMode"]            = static_cast<uint>(_upscalerMode);
        p["renderScale"]             = _renderScale;
        p["upscalerSharpness"]       = _upscalerSharpness;
        p["fsrJitterEnabled"]        = _fsrJitterEnabled;
        p["fxaaSkipWhenUpscaled"]    = _fxaaSkipWhenUpscaled;
        config["pipeline"] = p;
    }

    // Subsystems
    _passDI.toJson(config["di"]);
    _passGI.toJson(config["gi"]);
    _denoiser.toJson(config["denoiser"]);

    // Sky / Environment light
    {
        ci::Json sky;
        sky["yaw"]       = _lightSampler->env_yaw();
        sky["elevation"] = _lightSampler->env_elevation();
        sky["exposure"]  = _lightSampler->env_exposure();
        sky["turbidity"] = _lightSampler->env_turbidity();
        sky["solidBgEnabled"] = _solidBackgroundEnabled;
        sky["solidBgColor"] = ci::Json::array({_solidBackgroundColor.x, _solidBackgroundColor.y, _solidBackgroundColor.z});
        config["sky"] = sky;
    }

    return config;
}

void Pipeline::applyConfig(const ci::Json& config) {
    if (!config.is_object()) return;

    // Pipeline-level
    if (config.contains("pipeline")) {
        const auto& p = config["pipeline"];
        _checkerboardEnabled     = p.value("checkerboardEnabled",   _checkerboardEnabled);
        _primaryJitterEnabled    = p.value("primaryJitterEnabled",  _primaryJitterEnabled);
        _boilingFilterStrength   = p.value("boilingFilterStrength", _boilingFilterStrength);
        _toneMapMode             = static_cast<ToneMapMode>(p.value("toneMapMode", static_cast<uint>(_toneMapMode)));
        _toneMapExposure         = p.value("toneMapExposure",       _toneMapExposure);
        _toneMapGamma            = p.value("toneMapGamma",          _toneMapGamma);
        _toneMapLutPath          = p.value("toneMapLutPath",        _toneMapLutPath);
        if (p.contains("denoiserEnabled"))
            _denoiser.setEnabled(p["denoiserEnabled"].get<bool>());
        if (p.contains("denoiserMode"))
            setDenoiserMode(static_cast<DenoiserMode>(
                p.value("denoiserMode", 0u)));
        if (p.contains("giEnabled"))
            _passGI.setEnabled(p["giEnabled"].get<bool>());
        // Upscaler — mode first (scale < 1 is only valid with an active
        // backend; both setters recreate GPU resources on change).
        if (p.contains("upscalerMode"))
            setUpscalerMode(static_cast<upscal::UpscalerMode>(
                p.value("upscalerMode", 0u)));
        if (p.contains("renderScale"))
            setRenderScale(p.value("renderScale", 1.0f));
        if (p.contains("upscalerSharpness"))
            setUpscalerSharpness(p.value("upscalerSharpness", 0.0f));
        // perf R2 item 14 knobs (plain fields — no resource recreation; the
        // jitter toggle routes through the setter for history resets).
        if (p.contains("fsrJitterEnabled"))
            setFsrJitterEnabled(p.value("fsrJitterEnabled", true));
        if (p.contains("fxaaSkipWhenUpscaled"))
            setFxaaSkipWhenUpscaled(p.value("fxaaSkipWhenUpscaled", true));
    }

    // Subsystems
    if (config.contains("di"))       _passDI.fromJson(config["di"]);
    if (config.contains("gi"))       _passGI.fromJson(config["gi"]);
    if (config.contains("denoiser")) _denoiser.fromJson(config["denoiser"]);

    // Sky / Environment light
    if (config.contains("sky")) {
        const auto& s = config["sky"];
        _lightSampler->set_env_yaw(s.value("yaw", 0.0f));
        _lightSampler->set_env_elevation(s.value("elevation", 0.698f));
        _lightSampler->set_env_exposure(s.value("exposure", 1.0f));
        _lightSampler->set_env_turbidity(s.value("turbidity", 2.5f));
        _solidBackgroundEnabled = s.value("solidBgEnabled", false);
        if (s.contains("solidBgColor") && s["solidBgColor"].is_array()) {
            auto& arr = s["solidBgColor"];
            if (arr.size() >= 3)
                _solidBackgroundColor = luisa::make_float3(arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>());
        }
        if (_lightSampler->has_procedural_sky()) {
            _lightSampler->regenerate_procedural_sky(_computeStream);
            _passDI.markPresampleEnvDirty();
            // Order the regenerated CDF uploads (compute stream) against this
            // frame's render-stream env consumers — see the turbidity-slider
            // note above. No-op ordering-wise pre-build (render() not yet
            // running).
            _signalGeomUpdate();
        }
    }

    // Materials live in scene.json now; a materials section here is a
    // pre-split config — honor it once so upgrades keep their materials, and
    // point at the migration path.
    if (config.contains("materials")) {
        CI_LOG_W("config.json still contains \"materials\" (deprecated — they "
                 "now belong to scene.json); applying for compatibility. "
                 "Re-save the scene (Ctrl+S) to migrate them.");
        applyMaterials(config["materials"]);
    }

    CI_LOG_I("Applied config: "
        << "checkerboard=" << _checkerboardEnabled
        << " primaryJitter=" << _primaryJitterEnabled
        << " denoiser=" << _denoiser.enabled()
        << " gi=" << _passGI.enabled()
        << " upscaler=" << static_cast<uint>(_upscalerMode)
        << " renderScale=" << _renderScale);

    // Config load runs after buildScene's compile, so specialized flags
    // (giOneBounce / giHalfRes / diBiasCorrectionEnabled) may have flipped
    // from their baked defaults. Do NOT recompile here: this runs on the UI
    // thread (Reload Config button) where the previous frame's render
    // commands can still be in flight — shader destruction requires the
    // render-thread safe point. Pipeline::render() re-checks specialization
    // every frame before the first dispatch and recompiles there; for the
    // startup path (config loaded before the render loop) the first render()
    // call lands before any dispatch either way.
}

ci::Json Pipeline::captureMaterials() const {
    return _materialPool->materialsToJson();
}

void Pipeline::applyMaterials(const ci::Json& json) {
    if (!json.is_array()) return;
    _materialPool->materialsFromJson(json);
}

void Pipeline::loadConfig(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        CI_LOG_I("No config file at " << path << ", using defaults");
        return;
    }
    try {
        auto config = ci::loadJson(path);
        //auto config = nlohmann::json::parse(ci::loadString(ci::app::loadAsset("settings.json")));
        applyConfig(config);
        CI_LOG_I("Loaded config from " << path);
    } catch (const std::exception& e) {
        CI_LOG_E("Failed to load config: " << e.what());
    }
}

void Pipeline::saveConfig(const std::filesystem::path& path) {
    try {
        auto config = captureConfig();
        util::writeJsonWithBackup(path, config);

        CI_LOG_I("Saved config to " << path);
    } catch (const std::exception& e) {
        CI_LOG_E("Failed to save config: " << e.what());
    }
}

} // namespace newtype::core
