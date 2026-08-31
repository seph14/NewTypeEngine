#include "newtype/core/Pipeline.h"
#include "newtype/util/Camera.h"
#include "cinder/Log.h"
#include "cinder/Json.h"
#include "newtype/util/UiHelper.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
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

    // 2. Advance to next frame buffer
    _currentFrame = &renderer.beginFrame();

    // 3. Upload material changes from UI edits
    _materialPool->update(_bufferStream);

    // 4. Upload envmap rotation if changed (yaw/elevation from UI)
    _lightSampler->update_env_rotation(_computeStream);

    // 5. Dispatch PreUpdate features on compute stream
    //    (user GPU sim: VATMesh interpolation, particles, fluids)
    {
        auto& preFeatures = _features[static_cast<size_t>(FeaturePoint::PreUpdate)];
        if (!preFeatures.empty()) {
            util::CameraData dummyCam{};
            FrameContext preCtx = {
                _gbufDepth, _gbufVis, _gbufBaryMotion, _glassThroughput,
                dummyCam, *_geom, *_materialPool, *_lightSampler,
                _frameCount, _width, _height, 0u, _accumReset,
                0.0f,
                _seedImage,
                _accumBuffer, _specularBuffer,
                _denoiseAlbedo, _denoiseSpecFactor, _denoiseNormal,
                _gbufDepthPrev, _gbufVisPrev, _denoiseNormalPrev,
#if NT_ENABLE_BSSRDF
                _passSSS.sssRadiance(),
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

void Pipeline::update(float time, float dt) noexcept {
    // GPU-GPU sync: wait for previous frame's render tail to complete before
    // touching geometry resources (BLAS, vertex buffers, TLAS).
    if (_frameSubmitted)
        _computeStream << _frameTailEvent.wait(_frameTailFence);

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
        _computeStream << synchronize();
    } else if (matLightsDirty || geomLightsVisibilityDirty) {
        // Emission value OR light visibility toggled: cheap weight refresh,
        // no geometry rescan. _triangle_lights / _alias_table are read by
        // shade shaders on Renderer::stream() (a separate DX12 compute queue);
        // wait for the previous frame's last read to complete before rewriting
        // them on _computeStream. The trailing synchronize() ensures this
        // frame's writes are visible to this frame's render — Renderer::stream()
        // and _computeStream are separate queues, so cross-queue visibility
        // requires an explicit barrier in this direction.
        if (_lightSamplerReadyFence != 0u) {
            _computeStream << _lightSamplerReadyEvent.wait(_lightSamplerReadyFence);
        }
        _lightSampler->update_weights(_computeStream, *_materialPool, *_geom);
        _materialPool->clearLightsNeedRebuild();
        _geom->clear_lights_visibility_dirty();
        _computeStream << synchronize();
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
    }

    if (requireSync && !geomLightsDirty && !matLightsDirty && !geomLightsVisibilityDirty) _computeStream << synchronize();
}

//==============================================================================
// Runtime Shape Manipulation
//==============================================================================

void Pipeline::setShapeTransform(scene::ShapeId id, const float4x4 &matrix,
                                  scene::Change changeHint) noexcept {
    _geom->set_transform(id, matrix, changeHint);
}

void Pipeline::setShapeVisibility(scene::ShapeId id, bool visible) noexcept {
    _geom->set_visibility(id, visible);
}

void Pipeline::setShapeMaterial(scene::ShapeId id, uint32_t material_layers) noexcept {
    _geom->set_material_layers(id, material_layers);
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

//==============================================================================
// Resize
//==============================================================================

void Pipeline::resize(uint width, uint height) {
    // Mid-frame resize: beginFrame() was called but render() hasn't completed
    // yet. _currentFrame holds a live frame resource; recreating images now
    // invalidates the in-flight command list.
    if (_currentFrame != nullptr && !_frameSubmitted) {
        CI_LOG_W("Pipeline::resize called between beginFrame() and render() — "
            "recreating images mid-frame invalidates the live command list. "
            "Call resize() outside the frame loop.");
    }
    // Wait for any in-flight GPU work before recreating images
    if (_frameSubmitted) {
        Renderer::stream() << synchronize();
        _frameSubmitted = false;
    }

    _width  = width;
    _height = height;
    _createImages    (width, height);
    _initSeedImage   ();
    _denoiser.resetFrameIdx();

    // Resize all registered features
    auto& device = Renderer::device();
    for (auto& featureList : _features)
        for (auto& feature : featureList)
            feature->onResize(device, width, height);

    // Recreate temp image pool at new dimensions
    for (auto& entry : _tempImages) {
        entry.width = width;
        entry.height = height;
        entry.image = device.create_image<float>(entry.format, width, height);
    }

    requestAccumReset();
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

    // this should be constexpr
    //ImGui::Checkbox("Checkerboard", &_checkerboardEnabled);
    
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

        // Debug visualization mode
        const char* debugTagNames[] = { "None", "Depth", "Visibility", "BaryCentric", "Motion", "PDF", "Jacobian", "Normal", "Glass", "GIReservoir" };
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
        p["boilingFilterStrength"]   = _boilingFilterStrength;
        p["toneMapMode"]             = static_cast<uint>(_toneMapMode);
        p["toneMapExposure"]         = _toneMapExposure;
        p["toneMapGamma"]            = _toneMapGamma;
        p["toneMapLutPath"]          = _toneMapLutPath;
        p["denoiserEnabled"]         = _denoiser.enabled();
        p["giEnabled"]               = _passGI.enabled();
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

    // Materials
    config["materials"] = _materialPool->materialsToJson();

    return config;
}

void Pipeline::applyConfig(const ci::Json& config) {
    if (!config.is_object()) return;

    // Pipeline-level
    if (config.contains("pipeline")) {
        const auto& p = config["pipeline"];
        _checkerboardEnabled     = p.value("checkerboardEnabled",   _checkerboardEnabled);
        _boilingFilterStrength   = p.value("boilingFilterStrength", _boilingFilterStrength);
        _toneMapMode             = static_cast<ToneMapMode>(p.value("toneMapMode", static_cast<uint>(_toneMapMode)));
        _toneMapExposure         = p.value("toneMapExposure",       _toneMapExposure);
        _toneMapGamma            = p.value("toneMapGamma",          _toneMapGamma);
        _toneMapLutPath          = p.value("toneMapLutPath",        _toneMapLutPath);
        if (p.contains("denoiserEnabled"))
            _denoiser.setEnabled(p["denoiserEnabled"].get<bool>());
        if (p.contains("giEnabled"))
            _passGI.setEnabled(p["giEnabled"].get<bool>());
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
        if (_lightSampler->has_procedural_sky())
            _lightSampler->regenerate_procedural_sky(_computeStream);
    }

    // Materials
    if (config.contains("materials"))
        _materialPool->materialsFromJson(config["materials"]);

    CI_LOG_I("Applied config: "
        << "checkerboard=" << _checkerboardEnabled
        << " denoiser=" << _denoiser.enabled()
        << " gi=" << _passGI.enabled());
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
        ci::writeJson(path, config);
        
        CI_LOG_I("Saved config to " << path);
    } catch (const std::exception& e) {
        CI_LOG_E("Failed to save config: " << e.what());
    }
}

} // namespace newtype::core
