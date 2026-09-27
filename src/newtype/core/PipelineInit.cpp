#include "newtype/core/Pipeline.h"
#include "newtype/core/FeatureContext.h"
#if NT_ALLOW_RASTER_FEATURES
#include "newtype/core/IVoxelGridUser.h"
#include "newtype/feature/RasterBase.h"
#include "newtype/render/DDAMarch.h"
#endif
#include "newtype/util/LutLoader.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"
#include "newtype/render/ProcBindlessSlots.h"
#include "newtype/render/ProceduralTrace.h"
#include "cinder/Log.h"
#include "cinder/app/App.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <future>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

//==============================================================================
// Constructor
//==============================================================================

inline Pipeline::Pipeline(Renderer& renderer) noexcept {
    auto& device = Renderer::device();

    _computeStream = device.create_stream();
    _bufferStream  = device.create_stream(compute::StreamTag::COPY);
    _featureGbufEvent = device.create_timeline_event();
    _renderReadyEvent = device.create_timeline_event();
    _frameTailEvent = device.create_timeline_event();
    _lightSamplerReadyEvent = device.create_timeline_event();
    _geomUpdateEvent = device.create_timeline_event();

    _materialPool = std::make_unique<render::MaterialPool>(device);
    // Resolver params UI commit -> accum reset (MaterialPool can't reach the
    // Pipeline; bridge mirrors Geometry's material-pool back-pointer pattern).
    _materialPool->setAccumResetCallback([this] { requestAccumReset(); });
    _lightSampler = std::make_unique<render::LightSampler>(device);
    _lightSampler->set_uniform_sampling(kUniformLightSampling);
    _geom         = Geometry::create(device);
    _geom->set_material_pool(_materialPool.get());

#if NT_ALLOW_RASTER_FEATURES
    // Voxel grid for point cloud self-shadowing (shared by multiple PointCloud features)
    _voxelGrid = luisa::make_unique<scene::VoxelGrid>(device);
    _rasterContext = luisa::make_unique<feature::RasterContext>(device);
#endif

    _requireExplicitBlit = false;// renderer.backend() == Backend::DirectX;

    // Render/display split: _width/_height are RENDER dims (display * scale).
    // Every internal pass and G-buffer image follows the render dims; only the
    // upscaler output + tonemap + AfterToneMap features use display dims.
    _renderer      = &renderer;
    _displayWidth  = renderer.width();
    _displayHeight = renderer.height();
    _renderScale   = renderer.renderScale();
    _width         = renderer.renderWidth();
    _height        = renderer.renderHeight();

    _createImages(_width, _height);
    _initSeedImage();
    requestAccumReset();

    // Fiber scheduler for CPU/GPU overlap (MARL-based)
    _fiberScheduler = std::make_unique<luisa::fiber::scheduler>();

    // Register built-in material types (0-13) as IdentitySurfaceResolver.
    // Tags 0..13 match MaterialType enum values.
    // Custom resolvers (14+) can be registered between create() and buildScene().
    for (uint i = 0; i <= 13u; ++i) {
        _surfaceResolver.create<IdentitySurfaceResolver>();
    }

    // NOTE: Shader compilation is deferred to buildScene() so that custom
    // material resolvers can be registered between Pipeline::create() and
    // buildScene(). The $switch dispatch in resolve_surface() reads the
    // Polymorphic entries at compile time, so all resolvers must be present first.
}

Pipeline::~Pipeline() {
    // Explicitly reset all smart-pointer members before automatic destruction.
    // EASTL's default_delete has a static_assert for complete types that can fire
    // during implicit member destruction. Explicit .reset() avoids this by
    // controlling destruction order and template instantiation point.

    // Sync any in-flight GPU work before teardown
    if (_frameSubmitted) {
        Renderer::stream() << synchronize();
        _frameSubmitted = false;
    }
    // _geomUpdateEvent is signaled from _computeStream — drain it before
    // releasing the event handle.
    _computeStream << synchronize();

    // Destroy fiber scheduler first (waits for in-flight fibers)
    _fiberScheduler.reset();

    // Sync and release timeline events
    _featureGbufEvent = {};
    _renderReadyEvent = {};
    _frameTailEvent = {};
    _lightSamplerReadyEvent = {};
    _geomUpdateEvent = {};

    _computeStream.release();
    _bufferStream.release();

    _passDI.release();
    _passGI.release();
    _denoiser.release();
    _rrDenoiser.shutdown();
    _passSSS.release();
#if NT_ENABLE_SHARC
    _passSharc.release();
#endif

    _materialPool.reset();
    _lightSampler.reset();
#if NT_ALLOW_RASTER_FEATURES
    _rasterContext.reset();
    _voxelGrid.reset();
#endif
    _geom.reset();
}

PipelinePtr Pipeline::create(Renderer& renderer) noexcept {
    return PipelinePtr(new Pipeline(renderer));
}

//==============================================================================
// Scene Construction API
//==============================================================================

uint Pipeline::addMaterial(const std::string& name, MaterialData data) {
    return _materialPool->createMaterial(name, data);
}

scene::ShapeId Pipeline::addShape(luisa::unique_ptr<MeshShape> shape, Transform* xform) {
    return _geom->add_shape(std::move(shape), xform);
}

scene::ShapeId Pipeline::addShape(luisa::unique_ptr<MeshShape> shape, luisa::unique_ptr<Transform> xform) {
    return _geom->add_shape(std::move(shape), std::move(xform));
}

scene::ShapeId Pipeline::addLightShape(luisa::unique_ptr<LightShape> light, Transform* xform) {
    return _geom->add_shape(std::move(light), xform);
}

scene::ShapeId Pipeline::addLightShape(luisa::unique_ptr<LightShape> light, luisa::unique_ptr<Transform> xform) {
    return _geom->add_shape(std::move(light), std::move(xform));
}

void Pipeline::addEnvMap(ci::Surface32fRef envmap) {
    if (isBuilt()) {
        CI_LOG_W("Pipeline::addEnvMap called after buildScene() - env CDFs "
            "are already wired into the light sampler; the new envmap will "
            "not take effect without rebuilding the sampler.");
    }
    _lightSampler->build_envmap(*envmap, _computeStream);
}

//==============================================================================
// Prototype Instancing API
//==============================================================================

Pipeline::PrototypeHandle Pipeline::addPrototype(luisa::unique_ptr<scene::MeshShape> prototype) {
    auto id = _geom->add_prototype(std::move(prototype));
    return PrototypeHandle{id};
}

scene::ShapeId Pipeline::addPrototypeInstance(PrototypeHandle proto, const float4x4 &transform,
                                              uint32_t material_layers) {
    return _geom->add_instance(proto.id, transform, material_layers);
}

scene::ShapeId Pipeline::addPrototypeInstance(PrototypeHandle proto, Transform* xform,
                                              uint32_t material_layers) {
    return _geom->add_instance(proto.id, xform, material_layers);
}

scene::ShapeId Pipeline::addPrototypeInstance(PrototypeHandle proto,
                                              luisa::unique_ptr<Transform> xform,
                                              uint32_t material_layers) {
    return _geom->add_instance(proto.id, std::move(xform), material_layers);
}

void Pipeline::addPrototypeInstances(PrototypeHandle proto,
                                      luisa::span<const float4x4> transforms,
                                      luisa::vector<scene::ShapeId> &out_ids) {
    _geom->add_instances(proto.id, transforms, out_ids);
}

void Pipeline::addPrototypeInstances(PrototypeHandle proto,
                                      luisa::span<const float4x4> transforms,
                                      luisa::span<const uint32_t> material_layers,
                                      luisa::vector<scene::ShapeId> &out_ids) {
    _geom->add_instances(proto.id, transforms, material_layers, out_ids);
}

void Pipeline::addPrototypeInstances(PrototypeHandle proto,
                                      luisa::span<Transform *const> transforms,
                                      luisa::vector<scene::ShapeId> &out_ids) {
    _geom->add_instances(proto.id, transforms, out_ids);
}

scene::MeshShape* Pipeline::getPrototype(PrototypeHandle proto) noexcept {
    return _geom->get_prototype(proto.id);
}

luisa::unique_ptr<scene::InstancedMesh> Pipeline::createInstancedMesh(
    PrototypeHandle proto, uint instance_count) {
    auto *proto_shape = _geom->get_prototype(proto.id);
    if (!proto_shape) {
        CI_LOG_W("Pipeline::createInstancedMesh: prototype handle " << proto.id
            << " is not registered (or was freed) - returning nullptr. Call "
            "addPrototype() first and retain the returned PrototypeHandle.");
        return nullptr;
    }
    auto mesh = scene::InstancedMesh::create(Renderer::device(), proto_shape, instance_count);
    _instancedMeshes.emplace_back(mesh.get());
    return mesh;
}

void Pipeline::releaseInstancedMesh(scene::InstancedMesh *mesh) noexcept {
    _instancedMeshes.erase(
        std::remove(_instancedMeshes.begin(), _instancedMeshes.end(), mesh),
        _instancedMeshes.end());
}

#if NT_ENABLE_PROCEDURAL
void Pipeline::setProceduralGeometry(luisa::unique_ptr<scene::ProceduralGeometry> geom) noexcept {
    if (isBuilt()) {
        CI_LOG_W("Pipeline::setProceduralGeometry called after buildScene() - "
            "the procedural TLAS entry is registered during build(); late "
            "registration is silently ignored by the renderer.");
    }
    _procGeom = std::move(geom);
}
#endif

void Pipeline::buildScene() {
    // Load custom material callable DLL (Debug builds; warns and continues
    // when absent). Must happen BEFORE shader compilation — resolvers are
    // baked into $switch.
#ifdef _DEBUG
    loadCallableDLL();
#endif

    // Create identity LUT so _toneMapLut.view() is always safe to pass to shader
    _createIdentityLut();

    // Compile all shader passes in parallel (4 async threads).
    // The Polymorphic _surfaceResolver is read at compile time to generate
    // $switch/$case dispatch, so all resolvers must be registered by now.
    {
        auto& device = Renderer::device();
        auto f_di       = std::async(std::launch::async, [&] { _passDI.compile(device, *_geom, _surfaceResolver, _checkerboardEnabled); });
        auto f_shade    = std::async(std::launch::async, [&] { _compileShadeShader(); });
        auto f_gi       = std::async(std::launch::async, [&] { _passGI.compile(device, _surfaceResolver, _checkerboardEnabled,
                                                                                 _geom->has_transparent_shadow_casters()); });
        auto f_denoiser = std::async(std::launch::async, [&] { _denoiser.compile(device, _surfaceResolver); });
        auto f_sss      = std::async(std::launch::async, [&] { _passSSS.compile(device, *_geom, _surfaceResolver); });
#if NT_ENABLE_SHARC
        auto f_sharc    = std::async(std::launch::async, [&] { _passSharc.compile(device, *_geom, _surfaceResolver); });
#endif
        _compileUtilityShaders();
#if NT_DEBUG_VIZ
        _compileDebugShaders();
#endif
        f_di.get(); f_shade.get(); f_gi.get(); f_denoiser.get();
        f_sss.get();
#if NT_ENABLE_SHARC
        f_sharc.get();
#endif
    }

    // Snapshot the checkerboard state the kernels above were compiled with —
    // a later runtime change (DLSS-RR engage, config load) diverges
    // _checkerboardEnabled from this and render() applies the flip through
    // _applyCheckerboardReconfigure.
    _checkerboardBaked = _checkerboardEnabled;

    // Phase 2 fixed pools: PassDI allocates its presample pools lazily inside
    // compile(), but the constructor's _createImages wired (empty, pre-compile)
    // buffer handles into PassGI/PassSharc — refresh the wiring now that the
    // pools exist. Both consumers read the pool uniformly at a hashed global
    // index, so only the handle + entry count matter. Resize re-wiring in
    // _createImages stays valid: the pool buffers survive resizes unchanged.
    _passGI.set_presample_env_tiles(_passDI.presample_env_tiles());
#if NT_ENABLE_SHARC
    _passSharc.set_presample_env_tiles(_passDI.presample_env_tiles());
#endif

    // Zero-initialize ReLAX persistent images to prevent uninitialized GPU memory
    // (NaN/Inf) from polluting temporal accumulation on the first frame.
    // Must happen after denoiser.compile() since _clearImageShader is needed.
    _denoiser.zeroInitImages(_width, _height);

    // Same for the ReSTIR reservoir buffers created by the Pipeline
    // constructor's _createImages call (which runs before compile() — see the
    // zero-fill note there). Startup allocations are usually zero pages, but
    // do not rely on that: the resize path proved recycled-heap garbage wedges
    // the GPU (docs/resize_tdr_root_cause.md).
    _passDI.zeroInitBuffers();
    _passGI.zeroInitBuffers();

    // Ensure all BLAS builds (from Renderer::stream()) are complete before
    // building TLAS on _computeStream. Without this cross-queue sync, the
    // TLAS may reference incomplete BLAS data on DX12, causing wrong hits.
    Renderer::stream() << synchronize();

    _materialPool->update(_bufferStream);

#if NT_ENABLE_PROCEDURAL
    // Wire procedural geometry BEFORE _geom->build() so the TLAS includes it.
    // has_instances() guard: a built-empty ProceduralGeometry (build()
    // early-returns) has no GPU buffers — emplacing those into the bindless
    // array crashes; the placeholder path below keeps the shader bindings
    // valid instead.
    if (_procGeom && _procGeom->has_instances()) {
        _geom->set_procedural_geometry(_procGeom.get());
        // Create bindless array and emplace procedural buffers
        _procBindless = Renderer::device().create_bindless_array(render::kProcSlotCount);
        _procBindless.emplace_on_update(render::kSlot_ProcInstances, _procGeom->instance_buffer());
        _procBindless.emplace_on_update(render::kSlot_ProcPositions, _procGeom->vat_positions());
        _procBindless.emplace_on_update(render::kSlot_ProcIndices, _procGeom->vat_indices());
        _procBindless.emplace_on_update(render::kSlot_ProcAABBs, _procGeom->aabb_buffer());
        _procBindless.emplace_on_update(render::kSlot_ProcNormals, _procGeom->vat_normals());
        _procBindless.emplace_on_update(render::kSlot_ProcDeformState, _procGeom->deform_state_buffer());
        _computeStream << _procBindless.update();

        _passDI.set_procedural_buffers(_procBindless);
        _passGI.set_procedural_buffers(_procBindless);
    } else {
        auto& device = Renderer::device();
        // Create 1-element placeholder buffers so shaders always have valid bindings
        _procInstanceBuf = device.create_buffer<scene::ProcInstanceData>(1u);
        _vatPosBuf = device.create_buffer<luisa::float4>(1u);
        _vatIdxBuf = device.create_buffer<compute::Triangle>(1u);
        _procAabbBuf = device.create_buffer<compute::AABB>(1u);
        _vatNormalsBuf = device.create_buffer<luisa::float4>(1u);
        _deformStateBuf = device.create_buffer<scene::ProcDeformState>(1u);

        // Create bindless array for placeholders
        _procBindless = device.create_bindless_array(render::kProcSlotCount);
        _procBindless.emplace_on_update(render::kSlot_ProcInstances, _procInstanceBuf);
        _procBindless.emplace_on_update(render::kSlot_ProcPositions, _vatPosBuf);
        _procBindless.emplace_on_update(render::kSlot_ProcIndices, _vatIdxBuf);
        _procBindless.emplace_on_update(render::kSlot_ProcAABBs, _procAabbBuf);
        _procBindless.emplace_on_update(render::kSlot_ProcNormals, _vatNormalsBuf);
        _procBindless.emplace_on_update(render::kSlot_ProcDeformState, _deformStateBuf);
        _computeStream << _procBindless.update();

        _passDI.set_procedural_buffers(_procBindless);
        _passGI.set_procedural_buffers(_procBindless);
    }
#endif

    _geom->build         (_computeStream);

    _bufferStream   << synchronize();
    _computeStream  << synchronize();

    // Load tone map LUT if path was set (from config). Without a configured
    // LUT the pipeline stays linear: the identity LUT from _createIdentityLut()
    // remains bound as a no-op passthrough. Apps that want a curve ship the
    // .cube as an asset and point toneMapLutPath at it.
    if (!_toneMapLutPath.empty())
        _loadToneMapLut(_toneMapLutPath);
    else
        CI_LOG_D("Tone map LUT not configured; using linear identity LUT");

    // Initialize voxel grid — query IVoxelGridUser features for bounds, fall back to defaults
#if NT_ALLOW_RASTER_FEATURES
    {
        luisa::float3 vmin{ 50.f, 50.f, 50.f};
        luisa::float3 vmax{-50.f,-50.f,-50.f};
        luisa::uint res = 256u;

        for (auto& featureList : _features) {
            for (auto& feature : featureList) {
                if (auto* vgUser = dynamic_cast<core::IVoxelGridUser*>(feature.get())) {
                    auto cfg = vgUser->voxelGridConfig();
                    vmin = min(vmin, cfg.bounds_min);
                    vmax = max(vmax, cfg.bounds_max);
                    res = luisa::min(res, cfg.resolution);
                }
            }
        }

        // check if we do have feature that requires voxel grid
        if (vmin.x < vmax.x)
            _voxelGrid->create(res, vmin, vmax);
        else _voxelGrid->create(1, luisa::float3(-.1f), luisa::float3(.1f));
    }
#endif

    CI_LOG_I("Pipeline: Geometry built with "
        << static_cast<uint>(_geom->instances().size()) << " instances");

    const auto& light_idx = _geom->light_indices();
    const auto& instances = _geom->instances();
    for (size_t i = 0; i < light_idx.size(); ++i) {
        const auto *shape = instances[light_idx[i]].get_shape();
        CI_LOG_D("Light " << i << " - triangles: " << shape->triangle_count()
            << ", tlas_index: " << light_idx[i]
            << ", material_id: " << shape->material_id());
    }

    CI_LOG_I("Pipeline: Scene created with "
        << static_cast<uint>(light_idx.size()) << " light sources");

    _lightSampler->build(_computeStream, *_geom, *_materialPool);

    CI_LOG_I("Pipeline: LightSampler built with " << _lightSampler->emissive_triangle_count() << " emissive triangles");

    // Create procedural sky (Rayleigh/Mie) if no user envmap loaded
    if (!_lightSampler->has_environment()) {
        _lightSampler->generate_procedural_sky(_computeStream);
        CI_LOG_D("Pipeline: Procedural sky generated");
    }

    // Initialize rotation buffer with default rotation
    _lightSampler->update_env_rotation(_computeStream);

    CI_LOG_I("Pipeline ready");
}

//==============================================================================
// Tone Map LUT Loading
//==============================================================================

void Pipeline::_loadToneMapLut(const std::filesystem::path& path) {
    if (path.empty()) return;

    try {
        auto tpath = path;
        if (path.is_relative()) tpath = ci::app::getAssetPath(path);
        auto lut = util::loadCubeLut(tpath);

        _uploadToneMapLut(lut, path.string());
        _toneMapLutPath = path.string();
    } catch (const std::exception& e) {
        CI_LOG_E("Failed to load LUT: " << e.what());
        _toneMapLutEnabled = false;
    }
}

void Pipeline::_uploadToneMapLut(const util::LutData& lut, const std::string& label) {
    auto& device = Renderer::device();
    uint32_t n   = lut.size;

    // Create 3D volume and upload
    _toneMapLut = device.create_volume<float>(
        PixelStorage::FLOAT4, n, n, n);
    Renderer::stream() << _toneMapLut.copy_from(lut.data.data()) << synchronize();

    _toneMapLutEnabled = true;

    CI_LOG_D("Loaded tone map LUT: " << label
        << " (size=" << n << ", " << n*n*n << " entries)");
}

void Pipeline::_createIdentityLut() {
    // Create a small 2x2x2 identity LUT so _toneMapLut.view() is always safe.
    // This LUT is a no-op: output RGB = input RGB.
    auto& device = Renderer::device();
    constexpr uint32_t n = 2u;
    luisa::vector<float> rgba(n * n * n * 4);
    for (uint32_t z = 0; z < n; ++z)
        for (uint32_t y = 0; y < n; ++y)
            for (uint32_t x = 0; x < n; ++x) {
                uint32_t idx = (z * n * n + y * n + x) * 4;
                rgba[idx + 0] = static_cast<float>(x) / static_cast<float>(n - 1u);
                rgba[idx + 1] = static_cast<float>(y) / static_cast<float>(n - 1u);
                rgba[idx + 2] = static_cast<float>(z) / static_cast<float>(n - 1u);
                rgba[idx + 3] = 1.0f;
            }
    _toneMapLut = device.create_volume<float>(PixelStorage::FLOAT4, n, n, n);
    Renderer::stream() << _toneMapLut.copy_from(rgba.data()) << synchronize();
}

//==============================================================================
// Image Creation
//==============================================================================

void Pipeline::_createImages(uint width, uint height) {
    auto& device = Renderer::device();

    // G-Buffer images (16 bytes/pixel)
    // simultaneous_access=true: both main stream and GI stream read these concurrently
    _gbufDepth      = device.create_image<float>(PixelStorage::FLOAT1, width, height, 1u, true);
    _gbufVis        = device.create_image<uint> (PixelStorage::INT2,   width, height, 1u, true);
    _gbufBaryMotion = device.create_image<float>(PixelStorage::HALF4,  width, height, 1u, true);
    _glassThroughput = device.create_image<float>(PixelStorage::HALF4,  width, height, 1u, true);

    _gbufDepth.set_name("gbuf_depth");
    _gbufVis.set_name("gbuf_vis");
    _gbufBaryMotion.set_name("gbuf_barymotion");
    _glassThroughput.set_name("glass_through");

    // Upscaler inputs — written by the same G-buffer store as the packed
    // bary/motion image above (full coverage every frame incl. checkerboard:
    // the half-width dispatch covers interleaved columns). Velocity is
    // unjittered camera+geometry motion in render-res pixels; depth is NDC
    // z/w of the virtual hit. Only consumed by the upscaler stage.
    _gbufVelocity = device.create_image<float>(PixelStorage::HALF2, width, height, 1u, true);
    _gbufDepthUpscale = device.create_image<float>(PixelStorage::FLOAT1, width, height, 1u, true);
    _gbufVelocity.set_name("gbuf_velocity");
    _gbufDepthUpscale.set_name("gbuf_depth_upscale");

    // Upscaler output at DISPLAY resolution — the tonemap block reads it
    // instead of the render target when the upscaler ran. Plain (non-
    // simultaneous-access) image: single stream, strictly ordered.
    if (_upscalerMode != upscal::UpscalerMode::None) {
        _upscaledHdr = device.create_image<float>(PixelStorage::FLOAT4,
                                                  _displayWidth, _displayHeight);
        _upscaledHdr.set_name("upscaled_hdr");
    }

    // DLSS-RR working images (render resolution): _rrColor holds the noisy
    // composited HDR (RR's Color input, must differ from the output target);
    // the other three are the NGX guide buffers written by the format pass.
    // Plain images like _upscaledHdr: single stream, strictly ordered.
    if (_denoiserMode == DenoiserMode::DlssRR) {
        _rrColor           = device.create_image<float>(PixelStorage::HALF4, width, height);
        _rrAlbedo          = device.create_image<float>(PixelStorage::HALF4, width, height);
        _rrF0              = device.create_image<float>(PixelStorage::HALF4, width, height);
        _rrNormalRoughness = device.create_image<float>(PixelStorage::HALF4, width, height);
        _rrColor.set_name("rr_color");
        _rrAlbedo.set_name("rr_albedo");
        _rrF0.set_name("rr_f0");
        _rrNormalRoughness.set_name("rr_normal_roughness");
    }

    // Accumulation buffers. accumBuffer is HALF4: the clamp-blit that used to
    // convert FLOAT4→HALF4 for the ReLAX inputs is folded into the shade
    // writes (radiance clamped to [0,256] at write time), and nothing
    // downstream needs FLOAT4 precision (denoiser inputs, denoiser-off
    // fallback composite, debug views all read Float4 from HALF4 storage).
    _accumBuffer      = device.create_image<float>(PixelStorage::HALF4,  width, height, 1u, true);
    _specularBuffer   = device.create_image<float>(PixelStorage::HALF4,  width, height, 1u, true);
    _seedImage        = device.create_image<uint> (PixelStorage::INT1,   width, height, 1u, true);

    _accumBuffer.set_name("accum_buffer");
    _specularBuffer.set_name("specular_buffer");
    _seedImage.set_name("seed_image");

    // ReLAX denoiser (creates its own textures internally)
    _denoiser.createImages(device, width, height);
    // LC places textures in a pooled heap with NO zero-fill: recreated ReLAX
    // histories / tiles contain recycled garbage (NaN/Inf) — every recreation
    // must repeat zeroInitImages() or the first TA / atrous passes read
    // garbage (resize-TDR debugging 2026-09-16). The constructor call
    // precedes denoiser.compile() (no _clearImageShader yet); buildScene()
    // zero-fills that generation right after compiling.
    static bool s_ctorCall = true; // constructor call precedes denoiser.compile()
    if (!s_ctorCall) _denoiser.zeroInitImages(width, height);
    else s_ctorCall = false;

    // Previous frame G-Buffer (for SVGF disocclusion detection)
    _gbufDepthPrev   = device.create_image<float>(PixelStorage::FLOAT1, width, height, 1u, true);
    _gbufVisPrev     = device.create_image<uint> (PixelStorage::INT2,   width, height, 1u, true);
    _denoiseNormalPrev = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);

    _gbufDepthPrev.set_name("gbuf_depth_prev");
    _gbufVisPrev.set_name("gbuf_vis_prev");
    _denoiseNormalPrev.set_name("denoise_normal_prev");

#if NT_ENABLE_SHARC
    // Phase 3 rough-glass gather resources. Info is written by the PSR
    // G-buffer pass every frame (zeros on non-rough pixels). The history
    // ping-pong accumulates the gather across frames; both start "garbage"
    // and are zero-filled by the clear shader on the first gather frame
    // (gap detection in render()) — .w holds the accumulation length
    // (0 = invalid), which makes reads safe.
    _roughGlassInfo    = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _roughGlassHist[0] = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _roughGlassHist[1] = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _roughGlassInfo.set_name("rough_glass_info");
    _roughGlassHist[0].set_name("rough_glass_hist0");
    _roughGlassHist[1].set_name("rough_glass_hist1");
    _roughGlassHistIdx = 0u;
    _roughGlassLastFrame = ~0u;  // resized/recreated: force a history reset
#endif

#if NT_ALLOW_RASTER_FEATURES
    _shadowCache     = device.create_image<float>(PixelStorage::HALF2, width, height);
    _shadowCachePrev = device.create_image<float>(PixelStorage::HALF2, width, height);
    _shadowCache.set_name("shadow_cache");
    _shadowCachePrev.set_name("shadow_cache_prev");
#endif

    // PassDI (creates reservoirs + presample tiles internally)
    _passDI.createImages(device, width, height);

    // PassGI (creates its own reservoirs internally)
    _passGI.createImages(device, width, height);

    // PassSSS (creates sss_radiance + params buffer internally)
    _passSSS.createImages(device, width, height);

    // Zero the recreated ReSTIR reservoirs. LC's DX backend places buffers in
    // a pooled heap with no zero-fill, and the first post-resize temporal /
    // shade dispatches read prev-slot entries never written at the new size
    // (checkerboard: the other field; plain: taps outside the freshly-written
    // region). A garbage entry passes is_valid()==M()>0 and carries a wild
    // light index / sample position into shadow-ray construction — a
    // data-dependent GPU wedge with no page fault (resize TDR 2026-09-16;
    // full notes in docs/resize_tdr_root_cause.md). Zeroed == invalid == the
    // same state as a fresh frame, which every consumer already handles.
    // zeroInitBuffers no-ops on the constructor call (shaders not yet
    // compiled); buildScene zero-fills that generation after compiling.
    _passDI.zeroInitBuffers();
    _passGI.zeroInitBuffers();

    // Wire PassDI's presampled env tiles into PassGI for the multi-candidate x2 NEE.
    _passGI.set_presample_env_tiles(_passDI.presample_env_tiles());

#if NT_ENABLE_SHARC
    // PassSharc (cache buffers + params; capacity fixed per scene, plan §8-1)
    _passSharc.createResources(device);
    // Same presampled env tiles feed the SHARC update-path env NEE candidate.
    _passSharc.set_presample_env_tiles(_passDI.presample_env_tiles());
    // Phase 2: GI initial binds the cache read-only for its x2 query (the
    // params buffer carries the grid parameters Update/Resolve hash with),
    // plus the hit/miss counter buffer for the panel's hit-rate readout.
    _passGI.set_sharc_cache(_passSharc.entriesBuffer(), _passSharc.resolvedBuffer(),
                            _passSharc.paramsBuffer(), _passSharc.queryStatsBuffer());
#endif

    // Denoiser auxiliary images (HALF4)
    _denoiseAlbedo = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _denoiseSpecFactor = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _denoiseNormal = device.create_image<float>(PixelStorage::HALF4, width, height, 1u, true);
    _denoiseAlbedo.set_name("denoise_albedo");
    _denoiseNormal.set_name("denoise_normal");

    // PreUpdate dummy target (1x1 FLOAT4 for features that don't write to render target)
    _dummyTarget = device.create_image<float>(PixelStorage::FLOAT4, 1u, 1u);
    _dummyTarget.set_name("dummy_target");

    // Post-wiring specialization refresh: the parallel compile section above
    // baked the flag snapshot before SHARC cache pointers were wired into
    // PassGI (set_sharc_cache above) — re-bake if the derived sharcQueryOn
    // or the geometry's transparent-caster state differs, before the render
    // loop starts.
    _refreshSpecialization();
}

void Pipeline::_initSeedImage() {
    // Per-pixel seed must be decorrelated across neighbors, otherwise the
    // raw-seed reads in the DI candidate pass (u_brdf, RIS u_accept) inherit
    // a linear ramp across the screen. LCG iteration preserves linearity
    // (a*s+c is affine, affine∘affine is affine), so a linear init like
    // `i*k+c` stays linear forever and shows up as a coherent "flowing water"
    // pattern in DI shadows. Hashing the pixel index destroys that structure;
    // LCG is a bijection on uint32 so the decorrelation is preserved as the
    // seed evolves frame-to-frame. Mirrors RTXDI's RandomSamplerState init
    // (JenkinsHash on linearPixelIndex, RandomSamplerState.hlsli:37).
    auto& stream = Renderer::stream();
    luisa::vector<uint> seeds(_width * _height);
    auto jenkins_hash = [](uint32_t a) noexcept -> uint32_t {
        a = (a + 0x7ed55d16u) + (a << 12);
        a = (a ^ 0xc761c23cu) ^ (a >> 19);
        a = (a + 0x165667b1u) + (a << 5);
        a = (a + 0xd3a2646cu) ^ (a << 9);
        a = (a + 0xfd7046c5u) + (a << 3);
        a = (a ^ 0xb55a4f09u) ^ (a >> 16);
        return a;
    };
    for (uint i = 0; i < seeds.size(); ++i)
        seeds[i] = jenkins_hash(i);
    stream << _seedImage.copy_from(seeds.data()) << synchronize();
}

//==============================================================================
// Shade Shader Compilation (stays in Pipeline — reads both DI and GI reservoirs)
//==============================================================================

void Pipeline::_compileShadeShader() {
    auto& device = Renderer::device();

    // Specialization bake (see the DI_Shade body): snapshot for the
    // render-time change check in _refreshSpecialization.
    _shadeBakedTransparentShadowCasters =
        (_geom != nullptr && _geom->has_transparent_shadow_casters()) ? 1u : 0u;

    _shadeShader = device.compile<2>([&](
        ImageFloat output,
        BufferVar<Reservoir> reservoir_buffer,
        ImageFloat gbuf_depth,
        ImageUInt  gbuf_vis,
        ImageFloat gbuf_bary_motion,
        UInt       frame_count,
        AccelVar   accel,
        Var<util::CameraData> camera,
        Var<SceneGeometryResources> scene,
        BindlessVar vertex_bindless,
        BindlessVar tex_bindless,
        Var<LightSamplingResources> lights,
        BufferVar<GIReservoir> gi_reservoir_buffer,
        Var<EnvLightResources> env,
        Float env_exp,
        UInt cbField,
        UInt debugVizMode,
        // Specular output
        ImageFloat spec_output,
        // Glass throughput
        ImageFloat glass_throughput,
        // Denoise albedo output (xyz=albedo, w=metallic*0.49 or 1.0 for emissive)
        ImageFloat albedo_output,
        ImageFloat spec_factor_output,
        // Visibility reuse
        UInt visMaxAge,
        Float visMaxDistance,
        Float envVisMaxDistance
        // Formerly an UInt arg: hasTransparentShadowCasters (shadow-ray mode,
        // 0 = opaque-only any-hit fast path) is compile-time specialized below.
#if NT_ALLOW_RASTER_FEATURES
        ,
        // Voxel grid binding group for point cloud self-shadowing
        Var<scene::VoxelGrid::Resources> voxel,
        // Temporal shadow cache
        ImageFloat shadow_cache_prev,
        ImageFloat shadow_cache
#endif
        ,
        // Frozen initial GI reservoir snapshot (for RTXDI FinalShading MIS)
        BufferVar<GIReservoir> gi_initial_buffer,
        // giMISRoughness: 0 = MIS disabled, otherwise floor for roughened BRDF (RTXDI default 0.3)
        Float gi_mis_roughness,
        // Delta-branch NEE: gates NEE at mirror-hit x2 for r<kMinRoughness metals (default OFF = bit-identical)
        UInt delta_branch_nee_enabled
        ,
        // Dispersive shadow walk dials (PassDI; docs/Dispersive-Glass-Shadows.md
        // Phase 2): glass crossings per channel sub-walk after the fan-out
        // (2 = plan dial (a); runtime bound also stops DXC unrolling the
        // sub-walk body), and the RGB-split saturation (1 = exact per-channel
        // estimator, lower desaturates toward the d-line gray).
        UInt disp_shadow_interfaces,
        Float disp_shadow_split
        ,
            // SSS radiance (HALF4, per-channel demodulated — written by PassSSS probe)
            ImageFloat sss_radiance
            ,
            // Stagnancy decorrelation (RTXDI 3.1): smoothed stagnancy + firefly
            // flag, both R16F at shaded-space (checkerboard-compacted) coords
            // — read with dispatch_id, NOT the full-res coord.
            ImageFloat gi_stagnancy,
            ImageFloat gi_firefly_flag,
            UInt  gi_decor_mode,      // 0=None 1=Uniform 2=Stagnancy
            Float gi_decor_factor,
            Float gi_decor_exponent,
            UInt  gi_decor_firefly,   // 1 = honor the firefly flag (forces swap)
            Float gi_decor_bound      // firefly bias-reduction weight cap
#if NT_ENABLE_PROCEDURAL
            ,
            BindlessVar proc_bindless
#endif
            ) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("DI_Shade");

        // Compile-time-specialized flag (former UInt arg): shadow-ray mode,
        // 0 = opaque-only. Baked at _compileShadeShader time — same value the
        // removed runtime arg carried, so bit-identical by construction;
        // flips recompile at the render-thread safe point (see
        // _refreshSpecialization). C++ const so DXC folds the dead branch.
        const uint hasTransparentShadowCasters = _shadeBakedTransparentShadowCasters;

#if !NT_ENABLE_DISPERSION
        // Dispersive shadow dials are read only inside NT_ENABLE_DISPERSION
        // regions of trace_shadow below.
        (void)disp_shadow_interfaces;
        (void)disp_shadow_split;
#endif

        // Clamp-blit folded into the producer: the ReLAX inputs (and the
        // denoiser-off fallback) expect radiance clamped to [0, 256] with the
        // .w channel (1.0 diffuse / hitDist specular) preserved — the former
        // _clampBlitShader full-screen passes did this after shade.
        auto clamp_radiance = [](Float3 c) noexcept {
            return make_float3(
                luisa::compute::clamp(c.x, 0.0f, 256.0f),
                luisa::compute::clamp(c.y, 0.0f, 256.0f),
                luisa::compute::clamp(c.z, 0.0f, 256.0f));
        };

        // Transparent shadow ray: trace through dielectric/emissive surfaces,
        // accumulating Fresnel + absorption attenuation.
        // Returns compose(visible: Bool, attenuation: Float3).
        // When hasTransparentShadowCasters == 0, short-circuits to a single
        // any-hit trace (intersect_any) — no transparent materials exist in
        // the scene so the closest-hit loop is wasted work.
        // tmin skips the self-intersection band near the origin (RTXDI
        // RAB_TraceRay uses TMin 0.001..0.01). Callers offset the origin one
        // base unit along the facing normal plus a quarter base along the
        // light direction (NRD GetXoffset scheme) — the along-dir slide
        // protects tangent rays without amplifying the offset at grazing
        // cosines, which leaked past near-tangent occluders.
        Callable trace_shadow = [&](Float3 origin, Float3 dir, Float tmin, Float tmax) noexcept {
            Bool visible = def(true);
            Float3 att = def(make_float3(1.0f));
            Float original_tmax = tmax;  // capture before loop reduces it
            Float hit_dist = def(0.0f);  // distance traveled (to occluder or light)

            // Fast path: opaque-only scene. Single any-hit BVH traversal.
            // No material reads, no Fresnel/absorption math. When the scene
            // has no transparent/cutout casters, this is the only shadow
            // work needed.
            if (hasTransparentShadowCasters == 0u) {
                auto s_ray = make_ray(origin, dir, tmin, tmax);
                Bool occluded = render::trace_occluded(accel, s_ray
#if NT_ENABLE_PROCEDURAL
                    , proc_bindless
#endif
                );
                visible = !occluded;
                att = ite(occluded, make_float3(0.0f), make_float3(1.0f));
                hit_dist = ite(occluded, 0.0f, original_tmax);
            }
            else {
                static constexpr uint kMaxShadowBounces = 4u;
#if NT_ENABLE_DISPERSION
                // ---- Dispersive glass shadows (docs/Dispersive-Glass-Shadows.md).
                // The shared walk continues straight (d-line factors) until the
                // first dispersive dielectric interface. There the deviation
                // gate refracts the three channels in pure ALU: sub-pixel
                // spread → keep the straight walk with per-channel Fresnel
                // only (zero extra rays); otherwise arm the fan-out below and
                // stop — the post-loop block runs three deterministic
                // per-channel refracted sub-walks. All state below is captured
                // at the fan-out interface for those sub-walks.
                static constexpr float kDispersionGateCos = 0.9999619f; // cos(0.5°)
                Bool disp_fan = def(false);      // fan-out armed (gate failed)
                Bool disp_straight = def(false); // gate-passed: straight walk, per-channel Fresnel
                Float3 fo_att = def(make_float3(1.0f));      // shared pre-fan-out factor (att at stop)
                Float3 fo_pos = def(make_float3(0.0f));
                Float3 fo_wo = def(make_float3(0.0f, 0.0f, 1.0f));
                Float3 fo_refr_n = def(make_float3(0.0f, 0.0f, 1.0f));
                Float fo_cos_i = def(0.0f);
                Bool fo_entering = def(true);
                Float fo_ior = def(1.5f);        // d-line index of the fan-out medium
                Float fo_abbe = def(0.0f);       // its Abbe V
                Float3 fo_att_v = def(make_float3(1.0f));
                Float fo_att_dist = def(0.0f);
                Float fo_blend = def(1.0f);   // glass_blend at the fan-out interface
                Float fo_t = def(0.0f);          // leg length to the interface
                Float fo_hit_dist = def(0.0f);   // walk distance through it
                Float fo_tmax = def(0.0f);       // remaining budget past it
#endif
                $for(b, kMaxShadowBounces) {
                    auto s_ray = make_ray(origin, dir, tmin, tmax);
                    auto s_hit = render::trace_closest(accel, s_ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );

                    $if(s_hit->miss()) {
                        if constexpr (config::kAllowRasterFeatures) {
#if NT_ALLOW_RASTER_FEATURES
                            Float voxel_att = dda_march(voxel,
                                origin, dir, tmax);
                            att = att * voxel_att;
                            $if(voxel_att < 0.001f) {
                                visible = false;
                                att = make_float3(0.0f);
                                $break;
                            };
#endif
                        }
                        visible = true;
                        hit_dist = original_tmax;  // reached light at full distance
                        $break;
                    };

                    UInt s_inst = s_hit.inst;
                    UInt s_prim = s_hit.prim;
                    // Blocker material layers. Procedural blockers: s_inst is the
                    // shared proc TLAS slot (out of range for instance_buffer) —
                    // classify from the AABB's own material layers. Only emissive
                    // (5) procedural passes (free — the walk advances straight, no
                    // vertex surface needed); dielectric procedural blocks without
                    // attenuation because the Fresnel/absorption walk below is
                    // mesh-only (vertex-interpolated surface) today.
                    UInt4 s_inst_data = def(make_uint4(0u));
                    UInt s_mat_layers = def(0u);
#if NT_ENABLE_PROCEDURAL
                    Bool s_hit_proc = s_hit.is_procedural;
                    $if(s_hit_proc) {
                        s_mat_layers = proc_bindless
                            .buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                            .read(s_prim).material_layers;
                    } $else {
#endif
                        s_inst_data = scene.instance_buffer.read(s_inst);
                        s_mat_layers = s_inst_data.y;
#if NT_ENABLE_PROCEDURAL
                    };
#endif
                    Var<MaterialData> s_mat = scene.material_buffer.read(Expr{ s_mat_layers & 0xFFu });

                    // Use effective BSDF type so custom callables (e.g. GlassResolver
                    // with bsdf_type_override=3) are still classified as glass.
                    UInt s_bsdf = get_effective_bsdf_type(s_mat);

                    Float s_hit_t_early = s_hit.committed_ray_t;

#if NT_ENABLE_PROCEDURAL
                    $if(s_hit_proc & s_bsdf != 5u) {
                        visible = false;
                        att = make_float3(0.0f);
                        hit_dist = hit_dist + s_hit_t_early;  // occluder distance
                        $break;
                    };
#endif

                    // Opaque non-alpha-cutout hit blocks the shadow ray
                    $if(s_bsdf != 3u & s_bsdf != 5u & s_bsdf != 11u) {
                        $if(!is_alpha_cutout(s_mat, vertex_bindless, tex_bindless, s_inst_data, s_prim, s_hit.bary)) {
                            visible = false;
                            att = make_float3(0.0f);
                            hit_dist = hit_dist + s_hit_t_early;  // occluder distance
                            $break;
                        };
                    };

                    Float s_hit_t = s_hit.committed_ray_t;
                    Float3 s_hit_pos = origin + dir * s_hit_t;

                    // Fresnel + absorption for dielectric (skip emissive — uninitialized ior/attenuation)
                    Float3 s_att_factor = make_float3(1.0f);
                    $if(s_bsdf != 5u) {
                        // Slim shadow resolve (perf review R2 item 6): the walk
                        // consumes ior/dispersion/attenuation/attenuation_distance
                        // + the interpolated normal only — built-in materials
                        // (identity resolver) copy those scalars straight from
                        // MaterialData; custom resolvers fall back to the full
                        // resolve inside. One fused triangle fetch serves both
                        // the normal and the fallback.
                        Float3 s_wo = -dir;
                        auto s_xform = scene.instance_transform_buffer.read(s_inst);
                        ShadowSurfaceData s_surf = resolve_shadow_surface(
                            _surfaceResolver, vertex_bindless, tex_bindless,
                            s_inst_data.z, s_inst_data.w, s_prim, s_hit.bary,
                                                        s_mat, s_wo, s_xform, s_hit.inst);
                        Float3 s_geo_ns = s_surf.geo_ns;
                        Float s_cos_wo = dot(s_wo, s_geo_ns);
                        Bool s_entering = s_cos_wo > 0.0f;
                        Float3 s_refract_normal = ite(s_entering, s_geo_ns, -s_geo_ns);
                        Float s_cos_i = abs(dot(s_wo, s_refract_normal));
#if NT_ENABLE_DISPERSION
                        // Dispersive interface? Thin (11) is skipped — dispersion
                        // is inert for thin walls; emissive (5) never gets here.
                        Bool s_disp_hit = def(false);
                        Float s_dior_r = def(s_surf.ior);
                        Float s_dior_g = def(s_surf.ior);
                        Float s_dior_b = def(s_surf.ior);
                        $if((s_bsdf == 3u) & (s_surf.dispersion > 0.0f)) {
                            s_disp_hit = true;
                            s_dior_r = dispersed_ior(s_surf.ior, s_surf.dispersion, 0u);
                            s_dior_g = dispersed_ior(s_surf.ior, s_surf.dispersion, 1u);
                            s_dior_b = dispersed_ior(s_surf.ior, s_surf.dispersion, 2u);
                            $if(!disp_straight) {
                                // Deviation gate: refract the three channels once
                                // in ALU. If the max pairwise spread is sub-pixel
                                // (< ~0.5°) the channel separation can't show at
                                // this geometry — continue straight with
                                // per-channel Fresnel only. TIR in any channel is
                                // a topological difference — always fan out.
                                // (refract_dir reflects at TIR; the sin² checks
                                // keep those dirs out of the dot products.)
                                auto chan_refr = [&](Float ior_c) noexcept {
                                    Float eta_c = ite(s_entering, 1.0f / max(ior_c, 1.0001f), max(ior_c, 1.0001f));
                                    Float sin2_c = eta_c * eta_c * (1.0f - s_cos_i * s_cos_i);
                                    return make_float4(refract_dir(s_wo, s_refract_normal, eta_c), sin2_c);
                                };
                                Float4 refr_r = chan_refr(s_dior_r);
                                Float4 refr_g = chan_refr(s_dior_g);
                                Float4 refr_b = chan_refr(s_dior_b);
                                Bool gate_ok = (refr_r.w < 1.0f) & (refr_g.w < 1.0f) & (refr_b.w < 1.0f)
                                    & (dot(refr_r.xyz(), refr_g.xyz()) > kDispersionGateCos)
                                    & (dot(refr_r.xyz(), refr_b.xyz()) > kDispersionGateCos)
                                    & (dot(refr_g.xyz(), refr_b.xyz()) > kDispersionGateCos);
                                $if(gate_ok) {
                                    disp_straight = true;
                                } $else {
                                    // Arm the fan-out; the shared walk stops HERE
                                    // (this interface is consumed by the sub-walks).
                                    disp_fan = true;
                                    fo_att = att;
                                    fo_pos = s_hit_pos;
                                    fo_wo = s_wo;
                                    fo_refr_n = s_refract_normal;
                                    fo_cos_i = s_cos_i;
                                    fo_entering = s_entering;
                                    fo_ior = s_surf.ior;
                                    fo_abbe = s_surf.dispersion;
                                    fo_att_v = s_surf.attenuation;
                                    fo_att_dist = s_surf.attenuation_distance;
                                    fo_blend = s_surf.glass_blend;
                                    fo_t = s_hit_t;
                                    fo_hit_dist = hit_dist + s_hit_t;
                                    fo_tmax = tmax - s_hit_t;
                                    $break;
                                };
                            };
                        };
#endif
                        Float s_F = fresnel_dielectric(s_cos_i,
                            ite(s_entering, 1.0f, s_surf.ior),
                            ite(s_entering, s_surf.ior, 1.0f));

                        Float3 s_absorption = make_float3(1.0f);
                        $if(s_surf.attenuation_distance > 0.0f) {
                            Float3 att_v = s_surf.attenuation;
                            Float ratio = s_hit_t / s_surf.attenuation_distance;
                            s_absorption = exp(ratio * log(max(att_v, make_float3(1e-10f))));
                        };
#if NT_ENABLE_DISPERSION
                        // Straight (gate-passed) dispersive interfaces: only the
                        // Fresnel term splits per channel — the bend is sub-pixel.
                        // fresnel_dielectric returns 1 at TIR, so a TIR channel's
                        // factor reaches 0 on its own. Non-dispersive interfaces
                        // keep the broadcast scalar term. The split dial
                        // desaturates toward the d-line term (ite keeps 1.0
                        // bit-identical to the exact estimator).
                        Float3 s_trans_f = def(make_float3(1.0f - s_F));
                        $if(s_disp_hit & disp_straight) {
                            Float3 s_f_rgb = 1.0f - make_float3(
                                fresnel_dielectric(s_cos_i,
                                    ite(s_entering, 1.0f, s_dior_r),
                                    ite(s_entering, s_dior_r, 1.0f)),
                                fresnel_dielectric(s_cos_i,
                                    ite(s_entering, 1.0f, s_dior_g),
                                    ite(s_entering, s_dior_g, 1.0f)),
                                fresnel_dielectric(s_cos_i,
                                    ite(s_entering, 1.0f, s_dior_b),
                                    ite(s_entering, s_dior_b, 1.0f)));
                            s_trans_f = ite(disp_shadow_split < 1.0f,
                                make_float3(1.0f - s_F)
                                    + (s_f_rgb - make_float3(1.0f - s_F)) * disp_shadow_split,
                                s_f_rgb);
                        };
                        s_att_factor = s_trans_f * s_absorption;
#else
                        s_att_factor = (1.0f - s_F) * s_absorption;
#endif
                        // Callable-driven glass blend (docs/glass_blend_plan.md):
                        // deterministic expectation on the shadow path — the
                        // diffuse fraction blocks, the glass fraction transmits.
                        // Built-in glass carries glass_blend = 1 (no change).
                        s_att_factor = (1.0f - s_surf.glass_blend)
                                     + s_surf.glass_blend * s_att_factor;
                    };

                    att = att * s_att_factor;
                    $if(dot(att, att) < 1e-6f) {
                        att = make_float3(0.0f);
                        hit_dist = hit_dist + s_hit_t;  // exhausted through transparent
                        $break;
                    };

                    Float s_offset = max(0.001f * s_hit_t, 1e-4f);
                    origin = s_hit_pos + dir * s_offset;
                    hit_dist = hit_dist + s_hit_t;  // advance through transparent
                    tmax = tmax - s_hit_t;
                };

#if NT_ENABLE_DISPERSION
                // ---- Three deterministic per-channel refracted sub-walks.
                // Channel c crosses the fan-out interface at its Cauchy IOR and
                // then every later dielectric interface the same way —
                // dispersed_ior inside dispersive media, the medium's own
                // d-line index otherwise (wavelength constant along the path;
                // same convention as gather_transmission_ray's leg replay) —
                // accumulating tau_c = prod (1−F_c)·exp(σ_c·d). Thin (11)
                // crosses straight (dispersion inert), emissive (5) is free,
                // and opaque/cutout classification mirrors the shared walk.
                // The walks are deterministic, so the x3 channel-basis weight
                // and the 1/3 average cancel: att = shared ⊙ (τ_R, τ_G, τ_B)
                // component-wise. hit_dist follows the green walk (channel 1,
                // nearest the d-line the denoiser specular guide expects).
                // TIR (here or at any later interface) terminates the channel
                // with tau_c = 0 — the same energy state 1−F → 0 reaches at
                // the critical angle today.
                $if(disp_fan) {
                    Float3 tau = def(make_float3(0.0f));
                    Float3 tau_hd = def(make_float3(0.0f)); // per-channel hit distances
                    $for(c, 3u) {
                        Float ior_c = dispersed_ior(fo_ior, fo_abbe, c);
                        Float eta_c = ite(fo_entering, 1.0f / max(ior_c, 1.0001f), max(ior_c, 1.0001f));
                        Float sin2_c = eta_c * eta_c * (1.0f - fo_cos_i * fo_cos_i);
                        Float3 wi_c = refract_dir(fo_wo, fo_refr_n, eta_c);
                        Float tau_c = def(0.0f);
                        Float hd_c = def(fo_hit_dist); // fan-out leg already counted
                        $if(sin2_c < 1.0f) {
                            // Fan-out interface factors at this channel's IOR
                            Float F_c = fresnel_dielectric(fo_cos_i,
                                ite(fo_entering, 1.0f, ior_c),
                                ite(fo_entering, ior_c, 1.0f));
                            Float abs_c = def(1.0f);
                            $if(fo_att_dist > 0.0f) {
                                abs_c = exp((fo_t / fo_att_dist)
                                    * log(max(fo_att_v[c], 1e-10f)));
                            };
                            tau_c = (1.0f - F_c) * abs_c;
                            // Fan-out interface blend — same deterministic
                            // expectation as the shared walk's mix above.
                            tau_c = (1.0f - fo_blend) + fo_blend * tau_c;

                            Float fo_off = max(0.001f * fo_t, 1e-4f);
                            Float3 o_c = def(fo_pos + wi_c * fo_off);
                            Float3 d_c = def(wi_c);
                            Float tx_c = def(fo_tmax);
                            Bool done_c = def(false);
                            // Runtime bound (PassDI dial): also prevents DXC
                            // from unrolling this body (same technique as the
                            // spatial kernel's neighbor loop). Budget
                            // exhaustion keeps the accumulated tau_c and
                            // assumes the rest clear — the shared walk's
                            // give-up semantics under kMaxShadowBounces.
                            $for(b2, disp_shadow_interfaces) {
                                $if(!done_c) {
                                    auto c_ray = make_ray(o_c, d_c, tmin, tx_c);
                                    auto c_hit = render::trace_closest(accel, c_ray
#if NT_ENABLE_PROCEDURAL
                                        , proc_bindless
#endif
                                    );
                                    $if(c_hit->miss()) {
                                        Bool c_vox_kill = def(false);
                                        if constexpr (config::kAllowRasterFeatures) {
#if NT_ALLOW_RASTER_FEATURES
                                            Float c_voxel = dda_march(voxel, o_c, d_c, tx_c);
                                            tau_c = tau_c * c_voxel;
                                            $if(c_voxel < 0.001f) {
                                                tau_c = 0.0f;
                                                c_vox_kill = true;
                                            };
#endif
                                        }
                                        hd_c = ite(c_vox_kill, hd_c, original_tmax);
                                        done_c = true;
                                    } $else {
                                        UInt c_inst = c_hit.inst;
                                        UInt c_prim = c_hit.prim;
                                        // Procedural blocker: classify via the AABB's own
                                        // material layers (c_inst is the shared proc TLAS
                                        // slot — out of range for instance_buffer). Only
                                        // emissive passes; the refractive walk below is
                                        // mesh-only (vertex-interpolated surface) today.
                                        UInt4 c_inst_data = def(make_uint4(0u));
                                        UInt c_mat_layers = def(0u);
#if NT_ENABLE_PROCEDURAL
                                        Bool c_hit_proc = c_hit.is_procedural;
                                        $if(c_hit_proc) {
                                            c_mat_layers = proc_bindless
                                                .buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances)
                                                .read(c_prim).material_layers;
                                        } $else {
#endif
                                            c_inst_data = scene.instance_buffer.read(c_inst);
                                            c_mat_layers = c_inst_data.y;
#if NT_ENABLE_PROCEDURAL
                                        };
#endif
                                        Var<MaterialData> c_mat = scene.material_buffer.read(Expr{ c_mat_layers & 0xFFu });
                                        UInt c_bsdf = get_effective_bsdf_type(c_mat);
                                        Float c_t = c_hit.committed_ray_t;

#if NT_ENABLE_PROCEDURAL
                                        $if(c_hit_proc & c_bsdf != 5u) {
                                            tau_c = 0.0f;
                                            hd_c = hd_c + c_t;
                                            done_c = true;
                                        };
#endif

                                        // Opaque non-alpha-cutout hit kills the channel
                                        $if(c_bsdf != 3u & c_bsdf != 5u & c_bsdf != 11u) {
                                            $if(!is_alpha_cutout(c_mat, vertex_bindless, tex_bindless, c_inst_data, c_prim, c_hit.bary)) {
                                                tau_c = 0.0f;
                                                hd_c = hd_c + c_t;
                                                done_c = true;
                                            };
                                        };

                                        $if(!done_c) {
                                            Float3 c_hit_pos = o_c + d_c * c_t;
                                            Float3 c_wo = -d_c;
                                            Float c_f = def(1.0f);
                                            Float3 c_next_dir = def(d_c);
                                            $if(c_bsdf != 5u) {
                                                auto c_xform = scene.instance_transform_buffer.read(c_inst);
                                                // Slim shadow resolve — see the shared
                                                // walk above (perf review R2 item 6).
                                                ShadowSurfaceData c_surf = resolve_shadow_surface(
                                                    _surfaceResolver, vertex_bindless, tex_bindless,
                                                    c_inst_data.z, c_inst_data.w, c_prim, c_hit.bary,
                                                                                                        c_mat, c_wo, c_xform, c_hit.inst);
                                                Float3 c_geo_ns = c_surf.geo_ns;
                                                Bool c_entering = dot(c_wo, c_geo_ns) > 0.0f;
                                                Float3 c_refr_n = ite(c_entering, c_geo_ns, -c_geo_ns);
                                                Float c_cos_i = abs(dot(c_wo, c_refr_n));

                                                // Channel-effective IOR: Cauchy index at
                                                // this channel in dispersive glass, the
                                                // medium's d-line index otherwise.
                                                Float c_ior = def(c_surf.ior);
                                                $if((c_bsdf == 3u) & (c_surf.dispersion > 0.0f)) {
                                                    c_ior = dispersed_ior(c_surf.ior, c_surf.dispersion, c);
                                                };
                                                Float F_cc = fresnel_dielectric(c_cos_i,
                                                    ite(c_entering, 1.0f, c_ior),
                                                    ite(c_entering, c_ior, 1.0f));
                                                Float abs_cc = def(1.0f);
                                                $if(c_surf.attenuation_distance > 0.0f) {
                                                    abs_cc = exp((c_t / c_surf.attenuation_distance)
                                                        * log(max(c_surf.attenuation[c], 1e-10f)));
                                                };
                                                c_f = (1.0f - F_cc) * abs_cc;
                                                // Later blendable interfaces in
                                                // the channel walk — same mix.
                                                c_f = (1.0f - c_surf.glass_blend)
                                                    + c_surf.glass_blend * c_f;

                                                // Refract at real dielectric interfaces
                                                // (type 3 only — thin and cutout-passed
                                                // surfaces keep the straight direction,
                                                // matching the shared walk's behavior
                                                // for everything it doesn't refract).
                                                Float cross_eta = ite(c_entering, 1.0f / max(c_ior, 1.0001f), max(c_ior, 1.0001f));
                                                Float sin2_t = cross_eta * cross_eta * (1.0f - c_cos_i * c_cos_i);
                                                $if((c_bsdf == 3u) & (sin2_t >= 1.0f)) {
                                                    // TIR: channel terminates ((1−F)=0
                                                    // already zeroed c_f).
                                                    tau_c = 0.0f;
                                                    hd_c = hd_c + c_t;
                                                    done_c = true;
                                                } $else {
                                                    c_next_dir = ite(c_bsdf == 3u,
                                                        refract_dir(c_wo, c_refr_n, cross_eta), d_c);
                                                };
                                            };

                                            $if(!done_c) {
                                                Float c_off = max(0.001f * c_t, 1e-4f);
                                                o_c = c_hit_pos + c_next_dir * c_off;
                                                d_c = c_next_dir;
                                                tx_c = tx_c - c_t;
                                                hd_c = hd_c + c_t;
                                                tau_c = tau_c * c_f;
                                                $if(tau_c < 1e-6f) {
                                                    tau_c = 0.0f;
                                                    done_c = true; // exhausted through transparent
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                        // Scatter into RGB (no dynamic-index vector writes in DSL)
                        Float3 c_mask = ite(c == 0u, make_float3(1.0f, 0.0f, 0.0f),
                                     ite(c == 1u, make_float3(0.0f, 1.0f, 0.0f),
                                                   make_float3(0.0f, 0.0f, 1.0f)));
                        tau = tau + c_mask * tau_c;
                        tau_hd = tau_hd + c_mask * hd_c;
                    };
                    // Split dial: desaturate (τR,τG,τB) toward gray(τG) — G is
                    // nearest the d-line. ite keeps 1.0 bit-identical to the
                    // exact per-channel estimator.
                    Float3 tau_att = ite(disp_shadow_split < 1.0f,
                        make_float3(tau.y) + (tau - make_float3(tau.y)) * disp_shadow_split,
                        tau);
                    att = att * tau_att;
                    visible = (tau_att.x + tau_att.y + tau_att.z) > 1e-6f;
                    hit_dist = tau_hd.y; // green sub-walk
                };
#endif
            };

            return compose(visible, att, hit_dist);
        };

#if NT_ENABLE_DELTA_BRANCH_NEE
        // NEE at mirror-hit x2 for delta-branch metals (r < kMinRoughness).
        // Body mirrors `gi_bounce` (PassGI.cpp:152-296) but returns only Float3 radiance
        // (avoids C2908 LUISA_STRUCT registration pitfall per feedback-luisa-callable-compile-time).
        // Env NEE is skipped — `presample_env_tiles` is not captured by _shadeShader; the existing
        // `mirror_hit->miss()` branch (L1367-1370) already handles sky reflections via envmap eval.
        // Triangle RIS loop bound is `kShadeMirrorNEECandidateCount` (independent of GI's
        // `kGiInitialCandidateCount`) so _shadeShader DXC compile time stays tunable.
        // Gated by NT_ENABLE_DELTA_BRANCH_NEE because the body perturbs DXC IR/register allocation
        // (feedback-bsdf-pdf-perturbs-reuse-loops pattern) — eliding it restores bit-identical shade.
        Callable mirror_nee_at_x2 = [&](
            Float3 pos, Float3 ns, Float3 wo,
            Float3 albedo, Float roughness, Float metallic, Float ior,
            UInt bsdf_type, Float meta,
            Float hit_t,
            UInt seed,
            Float3 conductor_eta, Float3 conductor_k
        ) noexcept -> Float3 {
            Float3 radiance = def(make_float3(0.0f));

            // Unlit materials at x2 - return albedo, no NEE (matches gi_bounce L160-162).
            $if(Expr{ bsdf_type == 12u } &meta < 0.5f) {
                radiance = albedo;
            } $else {
                // RIS state
                Float ris_w_sum   = def(0.0f);
                UInt  ris_M       = def(0u);
                Float3 sel_MC     = def(make_float3(0.0f));
                Float  sel_p_hat  = def(0.0f);
                Float3 sel_dir    = def(make_float3(0.0f, 1.0f, 0.0f));
                Float  sel_dist   = def(0.0f);

                // ======== Triangle candidates (alias-table power-weighted) ========
                $if(lights.emissive_count > 0u) {
                    $for(c, PassGI::kShadeMirrorNEECandidateCount) {
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
                        $if(cos_x2 > 0.0f & Expr{ light_dist > 0.05f }) {
                            auto tri_light = lights.triangle_lights.read(light_idx);
                            Float source_pdf;
                            if constexpr (Pipeline::kUniformLightSampling) {
                                source_pdf = lights.emissive_count_inv / tri_light.area;
                            } else {
                                source_pdf = tri_light.pdf / tri_light.area;
                            }
                            source_pdf = luisa::compute::max(source_pdf, 1e-10f);
                            Float3 light_normal = tri_light->normal();
                            Float cos_light = max(0.0f, dot(light_normal, -light_dir));
                            Float dist_sq = light_dist * light_dist;
                            Float3 direct = evaluate_direct_illuminance_with_geometry(
                                tri_light->emission(), light_dir, dist_sq, cos_x2, cos_light,
                                ns, wo,
                                albedo, roughness, metallic, ior, bsdf_type,
                                0.f, 0.f, 0.f, 0.5f, 0.f, 1.3f, 0.f, 0.f, 0.f,
                                luisa::compute::make_float3(1.f, 0.f, 0.f), 1.f,
                                conductor_eta, conductor_k);
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

                // ======== Resolve RIS winner: trace one shadow ray ========
                // Visibility uses render::trace_closest + alpha-cutout branch (matches gi_bounce
                // L275-286). Using trace_shadow instead would change semantics and bias relative to GI.
                $if(ris_M > 0u & Expr{ sel_p_hat > 1e-8f }) {
                    Float s_offset = max(0.001f * hit_t, 1e-4f);
                    Float shadow_tmax = sel_dist - s_offset;
                    auto shadow_ray = make_ray(pos + ns * s_offset, sel_dir, 0.0f, shadow_tmax);
                    auto shadow_hit = render::trace_closest(accel, shadow_ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                    Bool visible = shadow_hit->miss();
                    $if(!visible) {
                        UInt4 s_inst_data = scene.instance_buffer.read(shadow_hit.inst);
                        Var<MaterialData> s_mat = scene.material_buffer.read(Expr{ s_inst_data.y & 0xFFu });
                        visible = (s_mat.type == 5u) |
                            is_alpha_cutout(s_mat, vertex_bindless, tex_bindless, s_inst_data, shadow_hit.prim, shadow_hit.bary);
                    };
                    $if(visible) {
                        Float3 rad_x2 = (ris_w_sum / cast<Float>(ris_M)) * (sel_MC / max(sel_p_hat, 1e-8f));
                        radiance = rad_x2;
                    };
                };
            };

            return radiance;
        };
#endif // NT_ENABLE_DELTA_BRANCH_NEE

        UInt2 rsv = dispatch_id().xy();
        UInt2 rsv_res = dispatch_size().xy();
        UInt2 coord;
        UInt2 resolution;

        if (_checkerboardEnabled) {
            // Half-res dispatch: convert reservoir position to full-res pixel position
            coord = make_uint2(rsv.x << 1u, rsv.y);
            coord.x = coord.x + ((coord.y + cbField) & 1u);
            resolution = make_uint2(rsv_res.x * 2u, rsv_res.y);
            $if(rsv.x >= rsv_res.x | rsv.y >= rsv_res.y) { $return(); };
        } else {
            coord = rsv;
            resolution = rsv_res;
            $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };
        }

        Float  depth = gbuf_depth.read(coord).x;
        UInt4  vis = gbuf_vis.read(coord);
        UInt   inst_id = vis.x;
        UInt   prim_id = vis.y & 0x3FFFFFFFu;

        // Read envmap rotation matrix (envmap-local → world space)
        Float3x3 env_rot = env.env_rotation;

        Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
        auto ray = camera->generate_ray(ndc);
        // Hoist ray-equation world position once — used by the point-light,
        // procedural, and final mesh branches below. Glass overrides this
        // via instance_xform of the background surface hit.
        Float3 ray_world_pos = ray->origin() + ray->direction() * depth;
        Float3 color = def(make_float3(0.0f));
        Float3 spec_color = def(make_float3(0.0f));
        Float spec_hit_dist = def(0.0f);
        // GI bounce distance kept separate from the DI/mirror distance — the
        // final .w must be the DOMINANT specular event's distance (selected
        // after both contributions exist, before the delta-mirror branch).
        Float gi_spec_hit_dist = def(0.0f);

        // Shade diagnostics — hoisted outside $if for debug visualization
#if NT_DEBUG_VIZ
        Float dbg_mat_type = def(0.0f);      // background material type / 12.0
        Float dbg_reservoir_valid = def(0.0f); // 1.0 if DI reservoir valid
        Float dbg_direct_lum = def(0.0f);     // direct_light luminance
        Float dbg_indirect_lum = def(0.0f);   // indirect_light luminance
        Float dbg_is_env = def(0.0f);         // 1.0 if env light selected
        Float dbg_local_direct_lum = def(0.0f); // local light contribution lum
        Float dbg_env_direct_lum = def(0.0f);   // env light contribution lum
        Float dbg_shadow_result = def(-1.0f);  // -1=no shadow ray, 0=occluded, 1=visible
        Float dbg_shadow_att_lum = def(0.0f);   // shadow attenuation lum (for local)
        Float dbg_cos_shading = def(0.0f);       // cos_shading for local light
        // GI reservoir diagnostics (modes 9-11). Populated in the GI section below.
        Float dbg_gi_W            = def(0.0f);  // gi_r->weight() (capped at 20)
        Float dbg_gi_target_pdf   = def(0.0f);  // gi_r.target_pdf
        Float dbg_gi_M            = def(0.0f);  // cast<Float>(gi_r.M())
        Float dbg_gi_age          = def(0.0f);  // cast<Float>(gi_r.age())
        Float dbg_gi_weight_sum   = def(0.0f);  // gi_r.weight_sum
        Float dbg_gi_rad_lum      = def(0.0f);  // luminance(gi_r.rad())
        Float dbg_gi_cos_theta    = def(0.0f);  // cos_theta at x1 toward x2
        Float dbg_gi_dist         = def(0.0f);  // distance x1 -> x2
        Float dbg_gi_vis          = def(-1.0f); // -1=not traced, 0=occluded, 1=visible
        Float dbg_gi_shadow_att   = def(0.0f);  // luminance(gi_shadow_att)
        Float dbg_gi_valid        = def(0.0f);  // 1.0 if gi_r.is_valid()
        Float dbg_gi_selfhit      = def(0.0f);  // 1.0 if x2 on same instance as x1 (debug mode 12)
#endif

        $if(inst_id != ~0u) {
            Bool   is_glass = (vis.y >> 31u) > 0u;
            Bool   is_point = ((vis.y >> 30u) & 1u) > 0u;
            Bool   is_procedural = ((vis.y >> 29u) & 1u) > 0u;
            Float4 barymotion = gbuf_bary_motion.read(coord);
            Float2 bary = barymotion.xy();
            Float3 wo = -ray->direction();

            // Point primitive: envmap-based diffuse lighting (NOT for procedural geometry)
            $if(is_point & !is_procedural) {
                // Decode octahedral normal from bary_motion.zw
                Float2 oct_n = barymotion.zw();
                Float3 point_ns = render::oct_decode(oct_n);
                Float3 point_geo_ns = point_ns;
                Float3 point_facing_ns = ite(dot(wo, point_geo_ns) < 0.f, -point_geo_ns, point_geo_ns);
                point_ns = point_facing_ns;

                // UV from bary_motion.xy (fragment writes both UV components directly)
                Float2 point_uv = barymotion.xy();

                // Read material and world position
                Var<MaterialData> pmat;
#if NT_ENABLE_PROCEDURAL
                // Procedural hits now route through deferred shading, not here
                pmat = scene.material_buffer.read(Expr{inst_id});
#else
                pmat = scene.material_buffer.read(Expr{inst_id});
#endif
                Float3 world_pos = ray_world_pos;

                // Evaluate envmap radiance in normal direction (Lambertian)
                Float3 env_radiance = render::eval_envmap_radiance(
                    point_ns, env.envmap, env.env_width, env.env_height, env_rot, env_exp);

                // Self-shadow via voxel grid DDA with temporal caching
                Float env_offset = max(0.001f * depth, 1e-4f);
                Float3 shadow_origin = world_pos + point_facing_ns * env_offset;
                auto shadow_ray = make_ray(shadow_origin, point_ns, env_offset, 1e10f);
                Float point_shadow_att = 1.0f;
                // Check temporal cache BEFORE tracing — skip DDA on cache hit
                Bool shadow_needs_dda = true;
#if NT_ALLOW_RASTER_FEATURES
                if constexpr (config::kAllowRasterFeatures) {
                    Float4 prev_cached = shadow_cache_prev.read(coord);
                    Float prev_att = prev_cached.x;
                    Float prev_depth = prev_cached.y;
                    Bool depth_similar = abs(depth - prev_depth) < max(0.01f * depth, 0.001f);
                    $if(depth_similar & prev_att > 0.0f) {
                        // Cache hit — skip DDA, use cached attenuation directly
                        point_shadow_att = prev_att;
                        shadow_needs_dda = false;
                    };
                }
#endif
                auto shadow_hit = render::trace_closest(accel, shadow_ray
#if NT_ENABLE_PROCEDURAL
                    , proc_bindless
#endif
                );
                $if(shadow_hit->miss()) {
                    if constexpr (config::kAllowRasterFeatures) {
#if NT_ALLOW_RASTER_FEATURES
                        $if(shadow_needs_dda) {
                            point_shadow_att = dda_march(voxel,
                                shadow_origin, point_ns, 1e10f);
                        };
#endif
                    }
                } $else {
                    // Opaque geometry occludes (unless alpha cutout)
                    UInt s_inst = shadow_hit.inst;
                    UInt4 s_inst_data = scene.instance_buffer.read(s_inst);
                    Var<MaterialData> s_mat = scene.material_buffer.read(Expr{ s_inst_data.y & 0xFFu });
                    $if(s_mat.type != 3u & s_mat.type != 5u & s_mat.type != 11u) {
                        point_shadow_att = ite(
                            is_alpha_cutout(s_mat, vertex_bindless, tex_bindless, s_inst_data, shadow_hit.prim, shadow_hit.bary),
                            point_shadow_att, 0.0f);
                    };
                };
#if NT_ALLOW_RASTER_FEATURES
                // Always write shadow cache for current frame
                shadow_cache.write(coord, make_float4(point_shadow_att, depth, 0.0f, 0.0f));
#endif

                // BSDF evaluation
                render::SurfaceData surface;
                surface.albedo = pmat.albedo.xyz();
                surface.emission = pmat.emission;
                surface.roughness = pmat.roughness;
                surface.metallic = pmat.metallic;
                surface.ior = pmat.ior;
                surface.albedo_alpha = pmat.albedo.w;
                surface.alphacut = pmat.alphacut;
                surface.sheen = pmat.sheen;
                surface.sheen_tint = pmat.sheen_tint;
                surface.clearcoat = pmat.clearcoat;
                surface.clearcoat_gloss = pmat.clearcoat_gloss;
                surface.iridescence = pmat.iridescence;
                surface.iridescence_ior = pmat.iridescence_ior;
                surface.iridescence_thickness = pmat.iridescence_thickness;
                surface.anisotropic = pmat.anisotropic;
                surface.anisotropic_rot = pmat.anisotropic_rot;
                surface.attenuation = pmat.attenuation;
                surface.attenuation_distance = pmat.attenuation_distance;
                surface.specular_tint = pmat.specular_tint;
                surface.specular_trans = pmat.specular_trans;
                surface.flatness = pmat.flatness;
                surface.fabric = pmat.fabric;
                surface.ns = point_ns;
                surface.geo_ns = point_geo_ns;
                surface.position = world_pos;
                surface.bsdf_type = ite(cast<UInt>(cast<uint>(pmat.bsdf_type_override)) > 0u,
                    cast<UInt>(cast<uint>(pmat.bsdf_type_override)), pmat.type);
                surface.material_type = pmat.type;

                MaterialBSDF bsdf = surface.make_bsdf();
                Float3 diff_brdf, spec_brdf;
                Float3 wi = point_ns;
                Float cos_theta = luisa::compute::max(0.0f, dot(point_ns, wi));
                bsdf.evaluate_split(wo, wi, point_ns, diff_brdf, spec_brdf);
                Float3 geom = env_radiance * cos_theta;

                Float3 direct_light_point = point_shadow_att * diff_brdf * geom;
                Float3 direct_spec_point = point_shadow_att * spec_brdf * geom;

                // --- Local light sampling (alias table IS, N samples) ---
                $if(lights.emissive_count > 0u) {
                    UInt seed = util::xxhash32(make_uint2(
                        coord.x * 1973u + coord.y * 9277u, frame_count));
                    UInt raster_light_samples{_rasterLightSamples};
                    Float inv_N = 1.0f / cast<Float>(raster_light_samples);

                    $for(s, raster_light_samples) {
                        // Alias table selection (power-biased IS)
                        seed = util::lcg_ui(seed);
                        Float u_light = util::uniform_uint_to_float(seed);
                        Float u_scaled = u_light * cast<Float>(lights.emissive_count);
                        UInt aidx = cast<uint>(u_scaled);
                        aidx = min(aidx, lights.emissive_count - 1u);
                        auto aentry = lights.alias_table.read(aidx);
                        UInt light_idx = ite(u_scaled - cast<float>(aidx) < aentry.pdf,
                            aentry.triangle_index, aentry.alias_index);

                        // Random barycentric coordinates (area-weighted triangle sampling)
                        seed = util::lcg_ui(seed);
                        Float bu = util::uniform_uint_to_float(seed);
                        seed = util::lcg_ui(seed);
                        Float bv = util::uniform_uint_to_float(seed);
                        bu = sqrt(bu);
                        Float b0 = 1.0f - bu;
                        Float b1 = bu * (1.0f - bv);
                        Float b2 = bu * bv;

                        auto tri_light = lights.triangle_lights.read(light_idx);
                        auto verts = lights.triangle_vertices.read(light_idx);
                        Float3 light_point = b0 * verts.v0 + b1 * verts.v1 + b2 * verts.v2;

                        // IS weight: 1/(pdf_select * N_samples), pdf_select stored in tri_light.pdf
                        Float pdf_select = max(tri_light.pdf, 1e-10f);
                        Float3 is_weight = make_float3(inv_N / pdf_select);

                        Float3 to_light = light_point - world_pos;
                        Float dist_sq = luisa::compute::max(
                            luisa::compute::dot(to_light, to_light), 1e-6f);
                        Float3 light_dir = to_light * rsqrt(dist_sq);
                        Float cos_shading = luisa::compute::max(0.0f,
                            luisa::compute::dot(point_ns, light_dir));

                        $if(cos_shading > 0.0f) {
                            Float s_offset = max(0.001f * depth, 1e-4f);
                            auto pt_shadow_result = trace_shadow(
                                world_pos + point_facing_ns * s_offset + light_dir * (0.25f * s_offset),
                                light_dir, s_offset, sqrt(dist_sq) - s_offset);

                            $if(pt_shadow_result.get<0>()) {
                                Float3 s_shadow_att = pt_shadow_result.get<1>();
                                spec_hit_dist = pt_shadow_result.get<2>();  // DI specular hit distance
                                Float3 local_spec_contrib = def(make_float3(0.0f));
                                Float3 local_contribution = evaluate_direct_illuminance_split(
                                    tri_light->emission(), tri_light->normal(),
                                    light_point, world_pos, point_ns, wo,
                                    pmat.albedo.xyz(), pmat.roughness, pmat.metallic, pmat.ior,
                                    surface.bsdf_type,
                                    local_spec_contrib,
                                    pmat.sheen, pmat.sheen_tint,
                                    pmat.clearcoat, pmat.clearcoat_gloss,
                                    pmat.iridescence, pmat.iridescence_ior, pmat.iridescence_thickness,
                                    pmat.anisotropic, pmat.anisotropic_rot,
                                    surface.tangent, surface.tangent_w,
                                    pmat.attenuation, pmat.conductor_k);
                                direct_light_point = direct_light_point
                                    + is_weight * s_shadow_att * local_contribution;
                                direct_spec_point = direct_spec_point
                                    + is_weight * s_shadow_att * local_spec_contrib;
                            };
                        };
                    };
                };

                // Demodulate with NRD MaterialFactors (bounded [0.02,1],
                // NoV/roughness aware) — raw albedo division amplified the
                // non-albedo-proportional white lobes into green/magenta
                // YCoCg clamp collapse in the denoiser.
                Float3 rf0_pt = lerp(make_float3(0.04f), pmat.albedo.xyz(), pmat.metallic);
                Float3 diff_factor_pt = def(make_float3(1.0f));
                Float3 spec_factor_pt = def(make_float3(1.0f));
                nrd_material_factors(point_ns, wo, pmat.albedo.xyz(), rf0_pt,
                    pmat.roughness, diff_factor_pt, spec_factor_pt);
                color = (pmat.albedo.xyz() * 0.01f + direct_light_point) / diff_factor_pt;
                spec_color = direct_spec_point / spec_factor_pt;
                spec_hit_dist = 0.0f;

                output.write(coord, make_float4(clamp_radiance(color), 1.0f));
                spec_output.write(coord, make_float4(clamp_radiance(spec_color), 0.0f));
                albedo_output.write(coord, make_float4(diff_factor_pt, pmat.metallic * 0.49f));
                spec_factor_output.write(coord, make_float4(spec_factor_pt, 0.0f));
                $return();
            };

            // Read instance data — procedural hits use ProcInstanceData, not mesh instance_buffer.
            // For procedural hits, synthesize a compatible uint4: .y = material_layers from proc_instances.
            // The .z/.w bindless slots are zero (procedural geometry has no bindless vertex buffers).
            UInt4 inst_data;
#if NT_ENABLE_PROCEDURAL
            $if(is_procedural) {
                Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(inst_id);
                inst_data = make_uint4(0u, proc_inst.material_layers, 0u, 0u);
            } $else {
                inst_data = scene.instance_buffer.read(inst_id);
            };
#else
            inst_data = scene.instance_buffer.read(inst_id);
#endif

            // Resolve surface + reconstruct normals
            Float3 ns = def(compute::make_float3(0.0f, 1.0f, 0.0f));
            Float3 tangent = def(compute::make_float3(1.0f, 0.0f, 0.0f));
            Float  tangent_w = def(1.0f);
            Float3 geo_ns;

            SurfaceData surface;
            auto instance_xform = scene.instance_transform_buffer.read(inst_id);
#if NT_ENABLE_PROCEDURAL
            $if(is_procedural) {
                Float2 screen_uv = (make_float2(coord) + 0.5f) / make_float2(resolution);
                Float3 proc_world_pos = ray_world_pos;
                // Recompute bary at unjittered pixel center for stable texture
                // sampling — same rationale as the mesh branch below.
                auto proc_ray_unjit = camera->generate_ray(Expr{ ndc - camera->jitter });
                $if(!is_glass) {
                    bary = reconstruct_unjittered_bary_procedural(
                        proc_bindless, inst_id, prim_id, bary,
                        proc_ray_unjit->origin(), proc_ray_unjit->direction());
                };
                surface = resolve_procedural_surface_textured(
                    _surfaceResolver, proc_bindless, tex_bindless,
                    inst_id, prim_id, proc_world_pos, wo,
                    scene.material_buffer, bary,
                    cast<Float>(frame_count), screen_uv, resolution.x, resolution.y);
                ns     = surface.ns;
                geo_ns = surface.geo_ns;
                tangent = surface.tangent;
            } $else {
#endif
                // Mesh: full surface resolution with bindless vertex + texture sampling
                Float2 screen_uv = (make_float2(coord) + 0.5f) / make_float2(resolution);
                // Recompute bary at unjittered pixel center for stable texture
                // sampling. The G-Buffer stored jittered bary (Halton sub-pixel
                // jitter via generate_ray), which made albedo/normal/roughness
                // sampling walk across texels frame-to-frame and shimmer in the
                // denoiser's demod/remod path.
                auto ray_unjit = camera->generate_ray(Expr{ ndc - camera->jitter });
                // Single fused triangle fetch shared by the unjittered-bary
                // recompute and the surface resolve below.
                MeshTriVerts tri_verts = read_mesh_triangle(
                    vertex_bindless, inst_data.z, inst_data.w, prim_id);
                $if(!is_glass) {
                    bary = reconstruct_unjittered_bary(
                        tri_verts, bary,
                        ray_unjit->origin(), ray_unjit->direction());
                };
                surface = resolve_surface_from_instance_verts(
                    _surfaceResolver, vertex_bindless, tex_bindless,
                    tri_verts, inst_data, prim_id, bary,
                    scene.material_buffer, wo,
                    instance_xform,
                                        cast<Float>(frame_count), screen_uv, resolution.x, resolution.y, inst_id);

                ns       = surface.ns;
                geo_ns   = surface.geo_ns;
                tangent  = surface.tangent;
#if NT_ENABLE_PROCEDURAL
            };
#endif

            // Callable-driven glass blending (docs/glass_blend_plan.md):
            // rolled-opaque pixels stored the blendable surface with the glass
            // bit clear — reclass the resolver's 3/11 to a shadable opaque 1
            // before any make_bsdf / is_glass_bsdf gate below.
            Bool blend_rolled_opaque = ((surface.bsdf_type == 3u) |
                                        (surface.bsdf_type == 11u)) & !is_glass;
            reclass_blend_rolled_opaque(surface, is_glass);

            // Facing normal for shadow ray offsets: flipped to face the incoming ray.
            // geo_ns must stay raw for glass entering/exiting checks.
            Float3 facing_ns = ite(dot(wo, geo_ns) < 0.f, -geo_ns, geo_ns);

            // For glass PSR: use actual background surface position.
            // The ray-based position lands on the camera ray at virtual depth, which is inside
            // the glass sphere — shadow rays and GI bounces would hit the glass.
            // For opaque: ray-based position is correct and world-space.
            // surface.position is already world-space after resolve_surface_from_instance.
            // Procedural hits skip this — proc positions are already world space
            // (proc BLAS registered with identity TLAS transform in Geometry.cpp
            // and instance_transform_buffer does not contain procedural instances).
            // Blend-rolled-opaque pixels use the resolved position too: their
            // stored depth came from the unjittered classification ray, so the
            // jittered ray-based reconstruction would be slightly off-surface.
            Float3 world_pos = ite((is_glass | blend_rolled_opaque) & !is_procedural,
                surface.position,
                ray_world_pos);

            Var<MaterialData> material = scene.material_buffer.read(Expr{ inst_data.y & 0xFFu });

            // --- DI/GI reservoir-based lighting (applies to both mesh and procedural) ---

#if NT_DEBUG_VIZ
            dbg_mat_type = cast<float>(surface.bsdf_type) * (1.0f / 12.0f);
#endif
            Bool is_unlit = surface.bsdf_type == 12u;
            Bool unlit_gi = is_unlit & (material.meta >= 0.5f);
            Bool is_glass_bsdf      = surface.bsdf_type == 3u;
            Bool is_thin_dielectric = surface.bsdf_type == 11u;

            // Shared BSDF for env/local/GI evaluations (avoid 1-3 redundant constructions)
            MaterialBSDF bsdf = surface.make_bsdf();

            // Direct-lighting BSDF: when the Burley BSSRDF probe covers this
            // pixel (Subsurface material, flatness > 0, not thin-wall), zero
            // the HK lobe so direct light doesn't double-count surface term +
            // probe — the probe add below is the single direct SSS model.
            // Thin-wall subsurface (diffuse_trans > 0) keeps the full BSDF:
            // its probe is suppressed, the BSDF lobes are the only model.
            // Point primitives never reach here (early-out above); GI keeps
            // the full BSDF.
            Bool sss_probe_direct = Expr{ surface.bsdf_type == 6u }
                                  & Expr{ surface.flatness > 0.0f }
                                  & Expr{ surface.diffuse_trans <= 0.0f };
            MaterialBSDF bsdf_direct = surface.make_bsdf(sss_probe_direct);

            Float3 direct_light = def(make_float3(0.0f));
            Float3 direct_spec = def(make_float3(0.0f));

            UInt pixel_index;
            if (_checkerboardEnabled)
                pixel_index = rsv.y * rsv_res.x + rsv.x;
            else
                pixel_index = coord.y * resolution.x + coord.x;
            Var<Reservoir> r = reservoir_buffer.read(pixel_index);
#if NT_DEBUG_VIZ
            dbg_reservoir_valid = ite(r->is_valid(), 1.0f, 0.0f);
#endif

            // --- Direct Illumination (skip for unlit materials) ---
            $if(!is_unlit) {
            // P1-8: gate on SAMPLE validity, not just M>0 — after the
            // empty-but-alive semantics change (PassDI P1-1) a reservoir can
            // carry M>0 with light_idx == ~0u; the local-light path below
            // would clamp that to a legal (wrong) triangle index and shade a
            // phantom light. The visibility write-back below stays inside
            // this gate, so empty reservoirs skip it (nothing to confirm).
            $if(r->is_valid() & r.light_idx != ~0u) {
                $if(r.light_idx == Pipeline::kEnvLightSentinel) {
                    // ======== Environment light path ========
                    Float3 wi = uv_to_direction(r.light_bary_u, r.light_bary_v, env.env_width, env.env_height, env_rot);
                    Float3 env_radiance = eval_envmap_from_uv(r.light_bary_u, r.light_bary_v, env.envmap, env.env_width, env.env_height, env_exp);
                    Float cos_theta = luisa::compute::max(0.0f, luisa::compute::dot(surface.ns, wi));

                    // Shadow ray for env light — trace along wi to check occlusion.
                    // Env light is at infinity so tmax is very large.
                    Float env_dist_sq = cast<Float>(r->spatial_dist_x() * r->spatial_dist_x()
                        + r->spatial_dist_y() * r->spatial_dist_y());
                    Bool env_vis_reused = r->visibility() == 1u
                        & r->vis_age() < visMaxAge
                        & env_dist_sq < envVisMaxDistance * envVisMaxDistance;
                    Bool env_is_visible = ite(env_vis_reused, cos_theta > 0.0f, false);
                    Float3 env_shadow_att = def(make_float3(1.0f));
                    Float env_hit_dist = def(0.0f);  // hoisted: declared outside $if(!env_vis_reused)

                    $if(cos_theta > 0.0f) {
                    $if(!env_vis_reused) {
                        Float env_offset = max(0.001f * depth, 1e-4f);
                        auto env_result = trace_shadow(
                            world_pos + facing_ns * env_offset + wi * (0.25f * env_offset),
                            wi, env_offset, 1e10f);
                        env_hit_dist = env_result.get<2>();  // assign to outer-scope var
                        env_is_visible = env_result.get<0>();
                        env_shadow_att = env_result.get<1>();

                        r->set_visibility(ite(env_is_visible, 1u, 0u));
                        // Tinted shadow transmittance (glass) can't live in the
                        // 1-bit visibility state — reuse would restore white
                        // next frame (tint↔white flicker). Park the age at 15
                        // so the reuse gate `vis_age < visMaxAge` (visMaxAge
                        // <= 15, ReSTIR.h) always fails and the next frame
                        // re-traces; att ~= white keeps the amortized reuse
                        // (docs/Dispersive-Glass-Shadows.md Phase 0).
                        Bool env_att_tinted = env_is_visible
                            & any(env_shadow_att < make_float3(0.999f));
                        r->set_vis_age(ite(env_att_tinted, 15u, 0u));
                        r->set_spatial_dist_x(0);
                        r->set_spatial_dist_y(0);
                        reservoir_buffer.write(pixel_index, r);
                    };
                    } $else {
                        env_is_visible = false;
                    };

                    $if(env_is_visible) {
                        Float W = r->weight();
                        $if(!is_glass_bsdf & !is_thin_dielectric) {
                            Float3 diff_brdf, spec_brdf;
                            bsdf_direct.evaluate_split(wo, wi, surface.ns, diff_brdf, spec_brdf);
                            Float3 geom = env_radiance * cos_theta;
                            direct_light = env_shadow_att * W * diff_brdf * geom;
                            direct_spec = env_shadow_att * W * spec_brdf * geom;
                            spec_hit_dist = env_hit_dist;  // env shadow ray distance (1e10 if visible)
                        };
                    };

                    // Debug: track env light selection
#if NT_DEBUG_VIZ
                    dbg_is_env = 1.0f;
                    dbg_shadow_result = ite(env_is_visible, 1.0f, 0.0f);
                    dbg_shadow_att_lum = luminance(env_shadow_att);
                    dbg_env_direct_lum = luminance(direct_light);
#endif
                }
                $else {
                    // ======== Local light path ========
#if NT_DEBUG_VIZ
                    dbg_is_env = 0.0f;
#endif
                    $if(lights.emissive_count > 0u) {
                        UInt light_idx = min(r.light_idx, lights.emissive_count - 1u);
                        auto tri_light = lights.triangle_lights.read(light_idx);
                        // Hidden lights have pdf == 0 (see LightSampler::_upload_to_gpu).
                        // Reservoirs may still reference them after a visibility toggle;
                        // skip shadow trace + BRDF eval so contribution is exactly zero
                        // without invalidating temporal accumulation.
                        $if(tri_light.pdf > 0.0f) {
                        auto verts = lights.triangle_vertices.read(light_idx);

                        Float b0 = 1.0f - r->light_bary_u - r->light_bary_v;
                        Float3 light_point = b0 * verts.v0
                            + r->light_bary_u * verts.v1
                            + r->light_bary_v * verts.v2;

                        Float3 to_light = light_point - world_pos;
                        Float  dist_sq = luisa::compute::max(luisa::compute::dot(to_light, to_light), 1e-6f);
                        Float  inv_dist = rsqrt(dist_sq);
                        Float3 light_dir = to_light * inv_dist;
                        Float  light_dist = dist_sq * inv_dist;

                        Float cos_shading = luisa::compute::max(0.0f, luisa::compute::dot(surface.ns, light_dir));

                        // Debug: track cos_shading for local light
#if NT_DEBUG_VIZ
                        dbg_cos_shading = cos_shading;
#endif

                        // RTXDI-style visibility reuse: skip shadow ray if
                        // visibility was confirmed recently (age < maxAge)
                        // AND spatial displacement is within threshold
                        Float spatial_dist_sq = cast<Float>(r->spatial_dist_x() * r->spatial_dist_x()
                            + r->spatial_dist_y() * r->spatial_dist_y());
                        Bool vis_reused = r->visibility() == 1u
                            & r->vis_age() < visMaxAge
                            & spatial_dist_sq < visMaxDistance * visMaxDistance;
                        Bool is_visible = ite(vis_reused, cos_shading > 0.0f, false);
                        Float3 shadow_att = def(make_float3(1.0f));
                        Float dir_hit_dist = def(0.0f);  // hoisted: declared outside $if(!vis_reused)


                        $if(cos_shading > 0.0f) {
                        $if(!vis_reused) {
                            Float offset = max(0.001f * depth, 1e-4f);
                            auto shadow_result = trace_shadow(
                                world_pos + facing_ns * offset + light_dir * (0.25f * offset),
                                light_dir, offset, light_dist - offset);
                            is_visible = shadow_result.get<0>();
                            shadow_att = shadow_result.get<1>();
                            dir_hit_dist = shadow_result.get<2>();  // assign to outer-scope var

                            // Debug: track shadow ray result
#if NT_DEBUG_VIZ
                            dbg_shadow_result = ite(is_visible, 1.0f, 0.0f);
#endif
#if NT_DEBUG_VIZ
                            dbg_shadow_att_lum = luminance(shadow_att);
#endif

                            // Cache visibility result and reset age. Tinted
                            // transmittance parks the age at 15 instead (see
                            // the env-light site above).
                            r->set_visibility(ite(is_visible, 1u, 0u));
                            Bool att_tinted = is_visible
                                & any(shadow_att < make_float3(0.999f));
                            r->set_vis_age(ite(att_tinted, 15u, 0u));
                            r->set_spatial_dist_x(0);
                            r->set_spatial_dist_y(0);
                            reservoir_buffer.write(pixel_index, r);
                        };
                        } $else {
                            is_visible = false;
                        };

                        $if(is_visible) {
                            Float W = r->weight();
                            Float3 light_normal = tri_light->normal();
                            Float cos_light = max(0.0f, dot(light_normal, -light_dir));
                            Float3 geom = tri_light->emission() * cos_shading * cos_light / dist_sq;

                            Float3 diff_brdf, spec_brdf;
                            bsdf_direct.evaluate_split(wo, light_dir, surface.ns, diff_brdf, spec_brdf);
                            direct_light = shadow_att * W * geom * diff_brdf;
                            direct_spec = shadow_att * W * geom * spec_brdf;
                            spec_hit_dist = dir_hit_dist;  // DI specular hit distance (direct light)
                            // Debug: track local light contribution
#if NT_DEBUG_VIZ
                            dbg_local_direct_lum = luminance(direct_light);
#endif
                        };
                        };
                    };
                };
            };
            }; // end $if(!is_unlit)

            // --- GI Indirect Illumination (also for unlit+GI) ---
            Float3 indirect_light = def(make_float3(0.0f));
            Float3 indirect_spec = def(make_float3(0.0f));
            Var<GIReservoir> gi_r = gi_reservoir_buffer.read(pixel_index);
#if NT_DEBUG_VIZ
            dbg_gi_valid = ite(gi_r->is_valid(), 1.0f, 0.0f);
            dbg_gi_W = gi_r->weight();
            dbg_gi_target_pdf = gi_r.target_pdf;
            dbg_gi_M = cast<Float>(gi_r->M());
            dbg_gi_age = cast<Float>(gi_r->age());
            dbg_gi_weight_sum = gi_r.weight_sum;
            dbg_gi_rad_lum = luminance(gi_r->rad());
            dbg_gi_selfhit = 0.0f;
#endif

            $if(gi_r->is_valid()) {
                Float3 sample_pos = gi_r->pos();
                Float3 to_sample = sample_pos - world_pos;
                Float  dist_sq = luisa::compute::max(luisa::compute::length(to_sample), 1e-4f);
                Float3 wi = to_sample / dist_sq;
                Float  cos_theta = luisa::compute::max(0.0f, luisa::compute::dot(surface.ns, wi));
#if NT_DEBUG_VIZ
                dbg_gi_cos_theta = cos_theta;
                dbg_gi_dist = dist_sq;
#endif

                // RTXDI GI FinalShading: always trace a fresh visibility ray.
                // GI vis_age reuse was removed -- on low-roughness metals the cached
                // visibility survives camera motion while the BRDF lobe shifts by
                // orders of magnitude, producing fireflies. RTXDI has no vis_age
                // field on GIReservoir at all (only DI has visibility reuse).
                Bool gi_visible = cos_theta > 0.0f;
                Float3 gi_shadow_att = def(make_float3(1.0f));
                $if(gi_visible) {
                    Float offset = max(0.001f * depth, 1e-4f);
                    auto gi_shadow_result = trace_shadow(
                        world_pos + facing_ns * offset + wi * (0.25f * offset),
                        wi, offset, dist_sq - offset);
                    gi_visible = gi_shadow_result.get<0>();
                    gi_shadow_att = gi_shadow_result.get<1>();
                };
#if NT_DEBUG_VIZ
                dbg_gi_vis = ite(gi_visible, 1.0f, 0.0f);
                dbg_gi_shadow_att = luminance(gi_shadow_att);
#endif

                $if(gi_visible) {
                    $if(!is_glass_bsdf & !is_thin_dielectric & (!is_unlit | unlit_gi)) {
                        // --- Final (post-reuse) reservoir evaluation ---
                        // RTXDI FinalShading parity (FinalShading.hlsl:95 ->
                        // ShadingHelpers.hlsli EvaluateBrdf): evaluate with the same
                        // kMinRoughness-floored BRDF that p_hat used (GIShading.h).
                        // Reused W is normalized by the floored target function;
                        // multiplying it by the raw lobe leaves an unbounded
                        // f_true/f_floored ratio on smooth conductors (grazing-angle
                        // glow). Floored eval keeps the estimator consistent.
                        MaterialBSDF bsdf_gi = make_bsdf_roughened(bsdf, kMinRoughness);
                        Float3 final_diff_true, final_spec_true;
                        bsdf_gi.evaluate_split(wo, wi, surface.ns, final_diff_true, final_spec_true);
                        Float W_final = gi_r->weight();
                        Float3 final_rad = gi_shadow_att * W_final * gi_r->rad() * cos_theta;

                        // Cache products and MIS-relevant combinations early to free the raw
                        // BRDF outputs and final_rad before the init-path BRDF evals + shadow
                        // ray. FP order preserved: (final_diff_true*final_rad)*finalWeight +
                        // (init_diff_true*init_rad)*initWeight == original A*B*C + D*E*F.
                        Float3 albedo_for_mis    = surface.albedo;
                        Float3 final_contrib_diff = final_diff_true * final_rad;
                        Float3 final_contrib_spec = final_spec_true * final_rad;
                        Float3 f_comb_true        = final_diff_true * albedo_for_mis + final_spec_true;
                        // final_diff_true, final_spec_true, final_rad are now dead.

                        // Default: single-reservoir result (MIS block overrides below).
                        indirect_light = final_contrib_diff;
                        indirect_spec  = final_contrib_spec;
                        // GI bounce distance (x1->x2). Kept separate from the
                        // DI/mirror distance — the dominant event's distance is
                        // selected after both contributions exist.
                        gi_spec_hit_dist = dist_sq;

                        // --- RTXDI 3.1 stagnancy decorrelation (ref
                        // Decorrelation.hlsli:76-136) ---
                        // With per-pixel probability p — Uniform: factor;
                        // Stagnancy: saturate(4*factor*smoothedStagnancy^exp);
                        // boiling-flagged fireflies force p=1 — swap the reused
                        // reservoir for the frozen initial one (1-sample MIS).
                        // Decorrelates temporally-correlated input noise for
                        // ReLAX (and, later, DLSS-RR).
                        Bool decor_swap = def(false);
                        $if(gi_decor_mode != 0u & gi_decor_factor > 0.0f) {
                            Float decor_p = def(gi_decor_factor);
                            $if(gi_decor_mode == 2u) {
                                Float stag = saturate(gi_stagnancy.read(rsv).x);
                                decor_p = saturate(4.0f * gi_decor_factor *
                                    luisa::compute::pow(stag, max(gi_decor_exponent, 0.0f)));
                            };
                            $if(gi_decor_firefly != 0u) {
                                $if(gi_firefly_flag.read(rsv).x > 0.5f) {
                                    decor_p = 1.0f;
                                };
                            };
                            Float decor_u = util::uniform_uint_to_float(
                                util::xxhash32(make_uint2(pixel_index, frame_count)));
                            $if(decor_u < decor_p) {
                                decor_swap = true;
                            };
                        };

                        // --- RTXDI FinalShading MIS: blend final with frozen initial reservoir ---
                        // Roughened-BRDF MIS weights (kMISRoughness=0.3 default). On narrow
                        // specular lobes where the true BRDF spikes, the roughened BRDF acts
                        // as a stable reference and the math prefers the initial sample
                        // (single-sample, unbiased). On wide lobes, prefers the reused final.
                        // Ref: RTXDI FinalShading.hlsl:97-125, ShadingHelpers.hlsli:32-42.
                        // Entered for the blend OR a decorrelation swap (the swap reuses
                        // this block's initial-sample evaluation; a swap with MIS off
                        // skips the roughened-BRDF reference weights entirely).
                        $if((gi_mis_roughness > 0.0f) | decor_swap) {
                            Var<GIReservoir> gi_init = gi_initial_buffer.read(pixel_index);
                            $if(gi_init->is_valid()) {
                                // Initial-reservoir geometry
                                Float3 init_pos = gi_init->pos();
                                Float3 init_to_sample = init_pos - world_pos;
                                Float  init_dist_sq = luisa::compute::max(luisa::compute::length(init_to_sample), 1e-4f);
                                Float3 init_wi = init_to_sample / init_dist_sq;
                                Float  init_cos_theta = luisa::compute::max(0.0f, luisa::compute::dot(surface.ns, init_wi));

                                // Shadow ray for initial sample
                                Bool init_visible = init_cos_theta > 0.0f;
                                Float3 init_shadow_att = def(make_float3(1.0f));
                                $if(init_visible) {
                                    Float init_offset = max(0.001f * depth, 1e-4f);
                                    auto init_shadow_result = trace_shadow(
                                        world_pos + facing_ns * init_offset + init_wi * (0.25f * init_offset),
                                        init_wi, init_offset, init_dist_sq - init_offset);
                                    init_visible = init_shadow_result.get<0>();
                                    init_shadow_att = init_shadow_result.get<1>();
                                };

                                $if(init_visible) {
                                    // Firefly bias reduction (ref Decorrelation.hlsli:124-131):
                                    // cap the swapped-in initial weight when the swap lands on a
                                    // fresh reservoir — RTXDI's raw-stagnancy==0 test maps to
                                    // gi_r->age()==0 here.
                                    $if(decor_swap & (gi_r->age() == 0u)) {
                                        gi_init.weight_sum = luisa::compute::min(
                                            gi_decor_bound * gi_r.weight_sum, gi_init.weight_sum);
                                    };

                                    // initial-sample evaluation — floored, same as the
                                    // final-reservoir eval above (RTXDI EvaluateBrdf).
                                    Float3 init_diff_true, init_spec_true;
                                    make_bsdf_roughened(bsdf, kMinRoughness)
                                        .evaluate_split(wo, init_wi, surface.ns, init_diff_true, init_spec_true);

                                    Float W_init = gi_init->weight();
                                    Float3 init_rad = init_shadow_att * W_init * gi_init->rad() * init_cos_theta;
                                    Float3 init_contrib_diff = init_diff_true * init_rad;
                                    Float3 init_contrib_spec = init_spec_true * init_rad;

                                    $if(decor_swap) {
                                        // 1-sample MIS: initial-only contribution.
                                        indirect_light = init_contrib_diff;
                                        indirect_spec  = init_contrib_spec;
                                    } $else {
                                    // Roughened BRDF: same view with roughness floored to
                                    // gi_mis_roughness. Shares the exact lobe weights (standard
                                    // weights have no roughness dependence; composed lists never
                                    // did) and MS-GGX invariants with the true BSDF — matching
                                    // the former copy-then-modify behavior bit-for-bit.
                                    MaterialBSDF rough_bsdf = surface.make_bsdf_roughened(gi_mis_roughness);

                                    // 3 additional BRDF evaluations: rough-at-final, true-at-init, rough-at-init
                                    Float3 final_diff_rough, final_spec_rough;
                                    rough_bsdf.evaluate_split(wo, wi, surface.ns, final_diff_rough, final_spec_rough);
                                    Float3 init_diff_rough, init_spec_rough;
                                    rough_bsdf.evaluate_split(wo, init_wi, surface.ns, init_diff_rough, init_spec_rough);

                                    // RTXDI GetMISWeight (cubic): w = (lum_true / (lum_true + lum_rough))^3
                                    // combined = diffuse * albedo + specular (matches RTXDI)
                                    Float3 f_comb_rough = final_diff_rough * albedo_for_mis + final_spec_rough;
                                    Float  f_lt = clamp(luminance(f_comb_true),  0.0f, 1e4f);
                                    Float  f_lr = clamp(luminance(f_comb_rough), 1e-4f, 1e4f);
                                    Float  f_w  = saturate(f_lt / (f_lt + f_lr));
                                    Float  finalWeight = 1.0f - f_w * f_w * f_w;

                                    Float3 i_comb_true  = init_diff_true  * albedo_for_mis + init_spec_true;
                                    Float3 i_comb_rough = init_diff_rough * albedo_for_mis + init_spec_rough;
                                    Float  i_lt = clamp(luminance(i_comb_true),  0.0f, 1e4f);
                                    Float  i_lr = clamp(luminance(i_comb_rough), 1e-4f, 1e4f);
                                    Float  i_w  = saturate(i_lt / (i_lt + i_lr));
                                    Float  initWeight = i_w * i_w * i_w;

                                    indirect_light = final_contrib_diff * finalWeight
                                                   + init_contrib_diff  * initWeight;
                                    indirect_spec  = final_contrib_spec * finalWeight
                                                   + init_contrib_spec  * initWeight;
                                    };
                                };
                            };
                        };
                    };
                };
            };

            // NRD hit-distance contract: .w must carry the DOMINANT specular
            // event's distance. The GI branch used to overwrite the DI distance
            // unconditionally, so pixels whose visible specular is the direct
            // highlight (a light reflected on metal) carried the GI bounce
            // distance instead — a churning, radiance-decoupled field that
            // displaces the denoiser's virtual-motion ray every frame
            // (report §12 R2).
            Float di_spec_l = luminance(direct_spec);
            Float gi_spec_l = luminance(indirect_spec);
            spec_hit_dist = ite(gi_spec_l > di_spec_l, gi_spec_hit_dist, spec_hit_dist);

            // --- Delta surface branch (RTXDI-style isDeltaSurface) ---
            // For roughness < kMinRoughness on standard (non-glass) materials,
            // treat as a perfect mirror: replace the GGX specular (a noisy
            // near-delta at alpha^2 = 8.1e-7) with an explicit mirror ray that
            // captures emissive geometry and environment reflections cleanly.
            // Zero indirect_specular because GI has no specular-aware motion on
            // metals and produces fireflies/stuck pixels. Non-emissive reflected
            // geometry stays black until virtual motion for GI lands. Glass
            // (bsdf_type 3/11) is skipped — it has its own Fresnel/refraction.
            $if(surface.roughness < kMinRoughness
                & !is_glass_bsdf
                & !is_thin_dielectric) {
                Float3 mirror_dir = reflect(-wo, surface.ns);
                // Only trace when the mirror dir is on the viewer-facing side of
                // the geometry normal; at extreme grazing angles numerical error
                // can otherwise push the ray into the surface.
                $if(dot(mirror_dir, facing_ns) > 0.0f) {
                    Float mirror_offset = max(0.001f * depth, 1e-4f);
                    auto mirror_ray = make_ray(world_pos + facing_ns * mirror_offset,
                                                mirror_dir, 0.0f, 1e10f);
                    auto mirror_hit = render::trace_closest(accel, mirror_ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                    // Pass through camera-invisible geometry (an invisible
                    // light must not appear in mirror reflections); retraces
                    // only while the current hit is invisible, bounded at 4.
                    $for(mirror_skip, 4u) {
                        Bool mirror_hit_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                        mirror_hit_proc = mirror_hit.is_procedural;
#endif
                        Bool mirror_hit_invisible = def(false);
                        $if(!mirror_hit->miss() & !mirror_hit_proc) {
                            UInt4 mskip = scene.instance_buffer.read(mirror_hit.inst);
                            mirror_hit_invisible = (mskip.x & kCameraInvisibleFlag) != 0u;
                        };
                        $if(mirror_hit_invisible) {
                            Float mskip_off = max(0.001f * mirror_hit.committed_ray_t, 1e-4f);
                            Float3 mskip_pos = mirror_ray->origin()
                                + mirror_ray->direction() * mirror_hit.committed_ray_t;
                            mirror_ray = make_ray(mskip_pos + mirror_ray->direction() * mskip_off,
                                mirror_ray->direction(), 0.0f, 1e10f);
                            mirror_hit = render::trace_closest(accel, mirror_ray
#if NT_ENABLE_PROCEDURAL
                                , proc_bindless
#endif
                            );
                        } $else { $break; };
                    };

                    Float3 mirror_radiance = def(make_float3(0.0f));
                    Float  mirror_dist     = def(1e10f);
                    $if(mirror_hit->miss()) {
                        mirror_radiance = render::eval_envmap_radiance(
                            mirror_dir, env.envmap, env.env_width, env.env_height,
                            env_rot, env_exp);
                    } $else {
                        mirror_dist = mirror_hit.committed_ray_t;
                        UInt m_inst = mirror_hit.inst;
                        Float3 m_wo = -mirror_dir;
                        Float3 x2_pos = mirror_ray->origin() + mirror_ray->direction() * mirror_dist;

                        Var<MaterialData> m_mat;  // only read inside the runtime-gated NEE block (for .meta)
                        SurfaceData       m_surface;
                        UInt              m_material_layers;
#if NT_ENABLE_PROCEDURAL
                        // Procedural convention (matches glass PSR - PipelineInit.cpp:1861-1867):
                        // mirror_hit.inst is the proc TLAS slot (out of range for instance_buffer);
                        // mirror_hit.prim is the AABB index = inst_id for the proc resolver.
                        $if(mirror_hit.is_procedural) {
                            UInt   m_aabb_idx   = mirror_hit.prim;
                            UInt   m_local_tri  = mirror_hit.local_tri;
                            Float2 m_local_bary = mirror_hit.local_bary;
                            Var<scene::ProcInstanceData> proc =
                                proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(m_aabb_idx);
                            m_material_layers = proc.material_layers;
                            UInt packed_prim = (1u << 29u) | m_local_tri;
                            m_surface = resolve_procedural_surface_textured(
                                _surfaceResolver, proc_bindless, tex_bindless,
                                m_aabb_idx, packed_prim,
                                x2_pos, m_wo,
                                scene.material_buffer, m_local_bary,
                                0.0f, make_float2(0.0f), 0u, 0u);
                        } $else {
#endif
                            UInt4 m_inst_data = scene.instance_buffer.read(m_inst);
                            m_material_layers = m_inst_data.y;
                            m_surface = resolve_surface_from_instance(
                                _surfaceResolver, vertex_bindless, tex_bindless,
                                m_inst_data, mirror_hit.prim, mirror_hit.bary,
                                scene.material_buffer, m_wo,
                                scene.instance_transform_buffer.read(m_inst),
                                                                0.0f, make_float2(0.0f), 0u, 0u, m_inst);
#if NT_ENABLE_PROCEDURAL
                        };
#endif
                        Float3 x2_ns = m_surface.ns;

#if NT_ENABLE_DELTA_BRANCH_NEE
                        // Emissive contribution (always). For a delta reflection the BRDF pdf is a
                        // delta function, so MIS weight for emissive is implicitly 1.0 — matches
                        // RTXDI GetMISWeightForEmissiveSurface at delta.
                        Float3 emission_rad = m_surface.emission;

                        // NEE at mirror-hit x2 — direct lighting at x2 reflected to camera via mirror.
                        // Two gates:
                        //   delta_branch_nee_enabled — runtime toggle (default OFF = bit-identical)
                        //   m_surface.roughness >= kMinRoughness — RTXDI ShouldSampleNeeLights:294 safety:
                        //     skip if x2 is itself a delta mirror (polish-on-polish would evaluate a
                        //     near-delta GGX at x2, unstable + physically wrong).
                        Float3 nee_rad = def(make_float3(0.0f));
                        $if(delta_branch_nee_enabled != 0u & Expr{ m_surface.roughness >= kMinRoughness }) {
                            // x2_pos, x2_ns, m_surface resolved before the #if gate above.
                            // Fetch material here (only when NEE will actually fire) — needed for .meta
                            // since SurfaceData doesn't carry it. Saves a buffer read on pixels where the
                            // runtime gate is off or roughness is below the threshold.
                            m_mat = scene.material_buffer.read(Expr{ m_material_layers & 0xFFu });

                            // Per-pixel/per-frame/per-hit decorrelated seed
                            UInt nee_seed = util::xxhash32(make_uint4(
                                coord.x * 1973u + coord.y * 9277u,
                                frame_count,
                                m_inst,
                                mirror_hit.prim));

                            // wo at x2 = m_wo (already computed as -mirror_dir).
                            nee_rad = mirror_nee_at_x2(x2_pos, x2_ns, m_wo,
                                                       m_surface.albedo, m_surface.roughness, m_surface.metallic, m_surface.ior,
                                                       m_surface.bsdf_type, m_mat.meta,
                                                       mirror_dist, nee_seed,
                                                       m_surface.attenuation, m_surface.conductor_k);
                        };

                        mirror_radiance = emission_rad + nee_rad;
#else
                        mirror_radiance = m_surface.emission;
#endif // NT_ENABLE_DELTA_BRANCH_NEE
                    };

                    // Fresnel at the macro-surface angle (NoV), matching
                    // RTXDI RAB_SurfaceEvaluateDeltaReflectionThroughput.
                    // Conductor materials (k != 0, always): fresnel_conductor (PBRT FrComplex).
                    // Non-Conductor metals / dielectrics: Schlick with F0 = albedo or ((ior-1)/(ior+1))^2.
                    Float mirror_NoV = max(dot(surface.ns, wo), 0.0f);
                    Float3 F = [&]() noexcept {
                        Float ior_ratio = (surface.ior - 1.0f) / (surface.ior + 1.0f);
                        Float3 F0 = ite(surface.metallic > 0.5f,
                                        surface.albedo,
                                        make_float3(sqr(ior_ratio)));
                        auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };
                        Float one_minus = max(1.0f - mirror_NoV, 0.0f);
                        return F0 + (1.0f - F0) * pow5(one_minus);
                    }();
                    // Per-material sentinel: only Conductor materials (k != 0) take
                    // the FrComplex branch. Branch (not ite) so non-complex pixels skip
                    // the ~40-Float3-op Callable.
                    $if(any(surface.conductor_k != 0.0f)) {
                        F = fresnel_conductor(mirror_NoV, surface.attenuation, surface.conductor_k);
                    };

                    direct_spec    = F * mirror_radiance;
                    indirect_spec  = make_float3(0.0f);
                    spec_hit_dist  = mirror_dist;
                };
            };

            // GI now runs for glass pixels using background wall surface data.
            // glass_att applied later (line ~602) provides energy conservation.

            // --- SSS probe contribution (Burley 2015 BSSRDF) ---
            // The probe is the ONLY direct SSS response: the HK lobe is zeroed
            // in the DI BSDFs (make_bsdf(probe_covered), PassDI + the
            // evaluate_split calls above) so surface term + probe no longer
            // stack — that double count is what made SSS pale and flat. The HK
            // lobe stays active for GI throughput.
            // PassSSS writes per-channel DEMODULATED radiance (albedo excluded).
            // Modulate by the tinted albedo (albedo * attenuation) — this
            // matches HK's effective BRDF albedo and preserves the absorption
            // tint; demodulation below then cancels albedo correctly.
            // sss_blend: partial flatness blends Lambert <-> probe instead of
            // stacking a full probe on the Lambert lobe. sss_probe_scale applies
            // the coat transmittance (1-F12)(1-F23) + fuzz budget from
            // resolve_surface_layered so coat-over-SSS darkens like the other
            // base lobes (the probe's own F_x0/F_x2 cover the base-medium
            // interface only — no double count).
            $if(!is_unlit & !is_glass_bsdf & !is_thin_dielectric
                & Expr{ surface.bsdf_type == 6u }
                & Expr{ surface.flatness > 0.0f }
                & Expr{ surface.diffuse_trans <= 0.0f }) {
                Float sss_blend = surface.flatness * (1.0f - surface.metallic);
                Float3 sss_tinted_albedo = surface.albedo * surface.attenuation;
                direct_light = direct_light
                    + sss_blend * surface.sss_probe_scale
                    * sss_tinted_albedo * sss_radiance.read(coord).xyz();
            };

            // Track direct/indirect for shade diagnostics
#if NT_DEBUG_VIZ
            dbg_direct_lum = luminance(direct_light);
            dbg_indirect_lum = luminance(indirect_light);
#endif

            Float em_lum = luminance(surface.emission);
            // NRD MaterialFactors for demodulation (see Shading.h). Computed
            // here (jittered attributes) and again in the denoiser prefilter
            // (unjittered, stable) — the same scheme NRD-Sample uses for its
            // demod/remod pair.
            Float3 rf0 = lerp(make_float3(0.04f), surface.albedo, surface.metallic);
            Float3 diff_factor = def(make_float3(1.0f));
            Float3 spec_factor = def(make_float3(1.0f));
            nrd_material_factors(surface.ns, wo, surface.albedo, rf0,
                surface.roughness, diff_factor, spec_factor);
            $if(is_unlit) {
                color = surface.albedo;
                $if(unlit_gi) {
                    // Additive GI on top of flat albedo
                    color = color + indirect_light;
                };
                spec_color = make_float3(0.0f);
            } $elif(em_lum > 0.01f) {
                // Write zero to denoiser input for emissive pixels.
                // Emission is stored in albedo image and composited after denoising.
                // Writing surface.emission here causes ghost images because the
                // denoiser temporally accumulates bright emission values.
                color = make_float3(0.0f);
            } $else{
                color = (surface.albedo * 0.01f + direct_light + indirect_light) / diff_factor;
                spec_color = (direct_spec + indirect_spec) / spec_factor;
            };
            // Write factors for compositing: rgb carries the diffuse factor
            // (or emission / unlit albedo), w keeps the flag encoding.
            Float albedo_w = ite(em_lum > 0.01f, 1.0f, ite(is_unlit, 1.0f, surface.metallic * 0.49f));
            Float3 albedo_rgb = ite(em_lum > 0.01f, surface.emission,
                ite(is_unlit, surface.albedo, diff_factor));
            albedo_output.write(coord, make_float4(albedo_rgb, albedo_w));
            spec_factor_output.write(coord, make_float4(spec_factor, 0.0f));
        };

        // Do NOT apply glass attenuation here — let the denoiser work on un-attenuated
        // wall irradiance. Glass attenuation is applied in the post-denoiser glass tint
        // pass. Applying it pre-denoiser causes the bilateral filter to blur across
        // spatially-varying glass thickness, producing a milky appearance.

        // Read glass data for debug visualization (non-debug path doesn't need it)
        Float4 glass_tp = glass_throughput.read(coord);
        Float3 glass_att = glass_tp.xyz();
        Bool   is_glass  = (vis.y >> 31u) > 0u;

        // Debug visualization: override output with reservoir diagnostics
#if NT_DEBUG_VIZ
        $if(debugVizMode != 0u) {
            // NOTE: must match the indexing used by the main shade path above
            // (line ~973). The previous code used `coord.y * dispatch_size().x
            // + coord.x`, which mixed full-res coord.x with half-res
            // dispatch_size().x in checkerboard mode and silently read the
            // wrong reservoir slot.
            UInt dbg_pixel_index;
            if (_checkerboardEnabled)
                dbg_pixel_index = rsv.y * rsv_res.x + rsv.x;
            else
                dbg_pixel_index = coord.y * resolution.x + coord.x;
            Var<Reservoir> dbg_r = reservoir_buffer.read(dbg_pixel_index);
            Float dbg_W = dbg_r->weight();
            Float dbg_target_pdf = dbg_r.target_pdf;
            UInt dbg_M = dbg_r->M();
            Float dbg_w_sum = dbg_r.w_sum;
            Float dbg_r_is_env = ite(dbg_r.light_idx == Pipeline::kEnvLightSentinel, 1.0f, 0.0f);
            Float dbg_spec_lum = luminance(spec_color);

            Float4 dbg_out = def(make_float4(0.0f));
            $if(debugVizMode == 1u) {
                dbg_out = make_float4(dbg_W, dbg_target_pdf, dbg_r_is_env, dbg_spec_lum);
            } $elif(debugVizMode == 2u) {
                dbg_out = make_float4(cast<float>(dbg_M) * 0.01f, dbg_w_sum * 0.01f, dbg_r_is_env, dbg_spec_lum);
            } $elif(debugVizMode == 4u) {
                Bool psr_sky = is_glass & (inst_id == ~0u);
                Float tp_lum = luminance(glass_att);
                Float bounces_norm = cast<float>(vis.z) / 8.0f;
                dbg_out = make_float4(
                    ite(psr_sky, 1.0f, 0.0f), 
                    ite(is_glass, glass_tp.w, 0.0f),
                    ite(is_glass, tp_lum, 0.0f),
                    ite(is_glass, bounces_norm, 0.0f));
            } $elif(debugVizMode == 5u) {
                dbg_out = make_float4(
                    ite(is_glass, dbg_mat_type, 0.0f),
                    ite(is_glass, dbg_reservoir_valid, 0.0f),
                    ite(is_glass, dbg_direct_lum, 0.0f),
                    ite(is_glass, dbg_indirect_lum, 0.0f));
            } $elif(debugVizMode == 6u) {
                dbg_out = make_float4(
                    dbg_is_env,
                    dbg_shadow_result * 0.5f + 0.5f,
                    min(dbg_local_direct_lum * 0.1f, 1.0f),
                    min(dbg_env_direct_lum * 0.1f, 1.0f));
            } $elif(debugVizMode == 7u) {
                dbg_out = make_float4(
                    ite(dbg_is_env < 0.5f, dbg_shadow_result, -1.0f),
                    ite(dbg_is_env < 0.5f, dbg_shadow_att_lum, -1.0f),
                    ite(dbg_is_env < 0.5f, dbg_cos_shading, -1.0f),
                    min(dbg_local_direct_lum * 0.1f, 1.0f));
            } $elif(debugVizMode == 8u) {
                Float local_scale = min(dbg_local_direct_lum * 0.05f, 1.0f);
                Float env_scale = min(dbg_env_direct_lum * 0.05f, 1.0f);
                Float gi_scale = min(dbg_indirect_lum * 0.1f, 1.0f);
                dbg_out = make_float4(local_scale, env_scale, gi_scale,
                    min((dbg_local_direct_lum + dbg_env_direct_lum) * 0.05f, 1.0f));
            } $elif(debugVizMode == 9u) {
                // GI reservoir overview: R=W/20, G=target_pdf (raw), B=M/64, A=age/30
                dbg_out = make_float4(dbg_gi_valid * (dbg_gi_W * (1.0f / 20.0f)),
                                      dbg_gi_valid * min(dbg_gi_target_pdf * 10.0f, 1.0f),
                                      dbg_gi_valid * (dbg_gi_M * (1.0f / 64.0f)),
                                      dbg_gi_valid * (dbg_gi_age * (1.0f / 30.0f)));
            } $elif(debugVizMode == 10u) {
                // GI weight components: R=weight_sum/100, G=rad_lum, B=indirect_lum, A=cos_theta
                dbg_out = make_float4(dbg_gi_valid * min(dbg_gi_weight_sum * 0.01f, 1.0f),
                                      dbg_gi_valid * min(dbg_gi_rad_lum * 0.1f, 1.0f),
                                      min(dbg_indirect_lum * 0.1f, 1.0f),
                                      dbg_gi_cos_theta);
            } $elif(debugVizMode == 11u) {
                // GI geometry/visibility: R=dist (1m=1.0), G=vis (-1/0/1 -> 0/0.5/1),
                // B=shadow_att lum, A=unused
                dbg_out = make_float4(min(dbg_gi_dist * 0.1f, 1.0f),
                                      dbg_gi_vis * 0.5f + 0.5f,
                                      dbg_gi_shadow_att,
                                      dbg_gi_valid);
            } $elif(debugVizMode == 12u) {
                // GI self-hit debug: R=selfhit (red where BRDF ray hit same instance as receiver),
                // G=valid (gi reservoir valid), B=rad_lum (bright = stored radiance is bright),
                // A=dist (closer = brighter). If red pattern matches the bright "leak" pattern,
                // self-hit is confirmed as the source.
                dbg_out = make_float4(dbg_gi_valid * dbg_gi_selfhit,
                                      dbg_gi_valid,
                                      dbg_gi_valid * min(dbg_gi_rad_lum * 0.1f, 1.0f),
                                      dbg_gi_valid * (1.0f - min(dbg_gi_dist * 0.1f, 1.0f)));
            } $elif(debugVizMode == 13u) {
                // GI disocclusion diagnosis (cornell-box shadow-pattern investigation):
                //   R = gi_r invalid at shade (Step 3: initial failed + spatial gate skipped recovery)
                //   G = gi_r valid + shade shadow ray visible (correct GI)
                //   B = gi_r valid + shade shadow ray occluded (Step 4: x2 occluded from receiver)
                //   A = surface valid (1 if geometry hit, 0 if sky)
                // Healthy scene = mostly green with sparse blue near occluders.
                // Persistent red/blue patches near occluders confirm the bug.
                Bool valid_and_visible = (dbg_gi_valid > 0.5f) & (dbg_gi_vis > 0.5f);
                Bool valid_and_occluded = (dbg_gi_valid > 0.5f) & (dbg_gi_vis < 0.5f) & (dbg_gi_vis > -0.5f);
                Bool gi_invalid = (dbg_gi_valid < 0.5f);
                dbg_out = make_float4(
                    ite(gi_invalid, 1.0f, 0.0f),
                    ite(valid_and_visible, 1.0f, 0.0f),
                    ite(valid_and_occluded, 1.0f, 0.0f),
                    1.0f);
            } $else {
                dbg_out = make_float4(dbg_W, dbg_target_pdf * 0.01f, dbg_is_env, dbg_spec_lum);
            };
            output.write(coord, make_float4(
                luisa::compute::clamp(dbg_out.x, 0.0f, 256.0f),
                luisa::compute::clamp(dbg_out.y, 0.0f, 256.0f),
                luisa::compute::clamp(dbg_out.z, 0.0f, 256.0f),
                dbg_out.w));
            spec_output.write(coord, make_float4(clamp_radiance(spec_color), 0.0f));
        } $else {
#endif
            output.write(coord, make_float4(clamp_radiance(color), 1.0f));
            spec_output.write(coord, make_float4(clamp_radiance(spec_color), spec_hit_dist));
#if NT_DEBUG_VIZ
        };
#endif

    });
}

void Pipeline::_compileUtilityShaders() {
    auto& device = Renderer::device();

    // Compositing blit: denoised diffuse for geometry, envmap background for sky
    _compositeBlitShader = device.compile<2>([&](
        ImageFloat  out_frame,
        ImageFloat  denoised_diff,
        ImageFloat  gbuf_depth,
        Var<EnvLightResources> env,
        Var<util::CameraData> camera,
        Float env_exp,
        ImageFloat  albedo_img,
        ImageFloat  spec_factor_img,
        ImageFloat  denoised_spec,
        UInt solid_bg_enabled,
        Float3 solid_bg_color
    ) noexcept {
        set_name("composite_blit");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float3x3 env_rot = env.env_rotation;

        Float depth = gbuf_depth.read(coord).x;
        Float4 diff = denoised_diff.read(coord);
        Float4 spec = denoised_spec.read(coord);

        // Sky detection: depth = FLT_MAX for missed rays
        $if(depth > 1e10f) {
            Float3 sky_color = def(make_float3(0.0f));
            $if(solid_bg_enabled != 0u) {
                sky_color = solid_bg_color;
            } $else {
                auto ray = camera->generate_ray(Expr{
                    (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera.jitter
                });
                sky_color = eval_envmap_radiance(ray->direction(), env.envmap,
                    env.env_width, env.env_height, env_rot, env_exp);
            };
            out_frame.write(coord, make_float4(sky_color, 1.0f));
        } $else {
            // Re-modulate with the stored NRD demod factors. albedo.w encodes:
            // emissive=1.0, non-emissive=metallic*0.49
            Float4 albedo = albedo_img.read(coord);
            Bool isEmissive = albedo.w > 0.5f;
            Float3 remod_diff = ite(isEmissive, make_float3(1.0f), albedo.xyz());
            Float3 remod_spec = spec_factor_img.read(coord).xyz();
            Float3 denoised = diff.xyz() * remod_diff + spec.xyz() * remod_spec;
            Float3 combined = ite(isEmissive, albedo.xyz(), denoised);
            out_frame.write(coord, make_float4(combined, 1.0f));
        };
    });

    // DLSS-RR input format: repackage the denoiser G-buffer signals into the
    // NGX guide buffers (float diffuse albedo / specular F0 / packed
    // normal+roughness). Guide defaults follow the DLSS-RR Integration Guide
    // §3.4: sky = (0.5,0.5,0.5) albedo, black F0, zero normal + roughness;
    // emissive pixels carry emission in the color input, so albedo/F0 go to 0.
    // normal.w encodes floor()=matID (255.0 = sky sentinel) + fract()=roughness.
    //
    // Guide reconstruction: the prefilter stores NRD demod factors, not raw
    // material properties. RR's appendix wants specular albedo =
    // EnvBRDFApprox(F0, alpha, NoV) — the same RT Gems ch.32 fit
    // nrd_material_factors uses (Shading.h) — so both guides are recovered by
    // inverting those factors:
    //   spec_factor = lerp(0.02, 1, Fenv * lerp(0.1, 1, roughness))
    //   diff_factor = lerp(0.02, 1, (1 - Fenv) * albedo)
    // Feeding the previous approximation (diff_factor as albedo and
    // lerp(0.04, diff_factor, metallic) as F0) underestimated metal F0 by
    // roughly an order of magnitude and dimmed grazing-angle albedo, which
    // showed as lost highlights and blur in RR's reconstruction.
    _rrInputFormatShader = device.compile<2>([&](
        ImageFloat rr_albedo,
        ImageFloat rr_f0,
        ImageFloat rr_normal,
        ImageFloat albedo_img,
        ImageFloat spec_factor_img,
        ImageFloat normal_img
    ) noexcept {
        set_name("rr_input_format");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float4 albedo = albedo_img.read(coord);
        Float4 spec_factor = spec_factor_img.read(coord);
        Float4 normal = normal_img.read(coord);
        Bool isEmissive = albedo.w > 0.5f;
        Bool isSky = normal.w >= 254.5f;
        Float roughness = fract(normal.w);

        // Invert the NRD factor encodings (constants mirror Shading.h).
        Float3 fenv = saturate(((spec_factor.xyz() - 0.02f) * (1.0f / 0.98f))
                               / lerp(0.1f, 1.0f, roughness));
        Float3 base_albedo = saturate(((albedo.xyz() - 0.02f) * (1.0f / 0.98f))
                                      / max(1.0f - fenv, 1e-3f));
        // Lambertian guide: metals have no diffuse lobe (metallic rides in
        // albedo.w as metallic*0.49 on non-emissive pixels).
        Float metallic = min(albedo.w / 0.49f, 1.0f);
        Float3 diff_alb = base_albedo * (1.0f - metallic);

        Float3 f0 = fenv;
        Float3 n = ite(isSky, make_float3(0.0f), normalize(normal.xyz()));
        diff_alb = ite(isSky, make_float3(0.5f),
              ite(isEmissive, make_float3(0.0f), diff_alb));
        f0 = ite(isSky | isEmissive, make_float3(0.0f), f0);
        rr_albedo.write(coord, make_float4(diff_alb, 1.0f));
        rr_f0.write(coord, make_float4(f0, 1.0f));
        rr_normal.write(coord, make_float4(n, roughness));
    });

    // Glass tint shader: apply glass attenuation + Fresnel envmap reflection
    // throughput.rgb = product of (1-F_i) * absorption_i across all glass interfaces
    // throughput.a   = accumulated Fresnel reflectivity
    // Background surface normal approximates glass normal for Fresnel reflection
    //
    // Phase 3 (NT_ENABLE_SHARC): pixels the PSR classified as rough glass
    // (rough_glass_info.x > kRoughGlassEpsilon) replace BOTH the smooth tint
    // multiply and the delta envmap mirror with a GGX lobe integral evaluated
    // against the SHARC cache (docs/sharc_rough_glass_plan.md §7 Phase 3):
    //   result = attenuation * E[SHARC(transmission taps)]
    //          + fresnel_accum * E[SHARC(envmap fallback | reflection taps)]
    // with per-pixel temporal accumulation.
    //
    // Speckle-plan Phase 3 (NT_ENABLE_SHARC + NT_ENABLE_DISPERSION): smooth
    // DISPERSIVE glass pixels (rough_glass_info.w > 0) additionally replay 3
    // deterministic per-channel delta-refracted taps post-denoise, replacing
    // the d-line tint multiply with the exact per-channel background
    // (docs/dispersion_speckle_fix_plan.md Phase 3). All other pixels take
    // the $else branch = the previous shader body (bit-identical for
    // non-dispersive content).
    _glassTintShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageFloat denoised_input,
        ImageFloat glass_throughput_img,
        ImageFloat gbuf_depth,
        ImageUInt  gbuf_vis,
        ImageFloat gbuf_bary_motion,
        Var<util::CameraData> camera,
        Var<EnvLightResources> env,
        Float env_exp,
        Var<SceneGeometryResources> scene,
        BindlessVar vertex_bindless,
        AccelVar accel
#if NT_ENABLE_PROCEDURAL
        ,
        BindlessVar proc_bindless
#endif
#if NT_ENABLE_SHARC
        ,
        BindlessVar tex_bindless,
        ImageFloat rough_glass_info,
        BufferVar<SharcParams> sharc_params,
        BufferVar<SharcKeyHost> sharc_entries,
        BufferVar<render::SharcPackedData> sharc_resolved,
        BufferVar<luisa::uint> sharc_query_stats,
        UInt gather_on,
        UInt trans_taps,
        UInt refl_taps,
        ImageFloat gather_hist_prev,
        ImageFloat gather_hist_curr,
        UInt frame_count,
        Float gather_alpha,
        UInt gather_reset
#endif
    ) noexcept {
        set_name("glass_tint");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        // Guard against partial-block threads
        $if(any(coord >= resolution)) { $return(); };

        Float4 denoised = denoised_input.read(coord);
        Float4 glass    = glass_throughput_img.read(coord);

        // Non-glass pixels: glass = (1,1,1,0) → identity passthrough
        Float3 attenuation = glass.xyz();
        Float  fresnel     = glass.w;

        // Apply glass attenuation POST-denoiser: the denoiser worked on un-attenuated
        // wall irradiance (same as non-glass pixels), giving cleaner results. Now
        // apply throughput (1-F)*abs for the view path. Non-glass pixels have
        // attenuation = (1,1,1) → identity, so this is a no-op for them.
        Float3 result = attenuation * denoised.xyz();

#if NT_ENABLE_SHARC
        // ---- SHARC query bindings (same read-only view the GI x2 hook uses;
        // grid params from the same staging buffer as Update/Resolve) ----
        auto sharc_p = sharc_params.read(0u);
        SharcGridParams sharc_grid{camera->position, kSharcGridLogarithmBase,
                                   sharc_p.sceneScale, sharc_p.levelBias};
        SharcQueryCache<SharcEngineLayout> sharc_cache{
            std::move(sharc_entries), std::move(sharc_resolved), sharc_p.capacity};

        // Per-tap luminance clamp (speckle-fix Phase B): cap any gather tap's
        // luminance at kGatherTapClamp × the reference's, preserving chroma —
        // emitter-adjacent voxels and escaped-ray sun texels can't out-shine
        // the pixel's local light scale and re-introduce fireflies.
        auto clamp_tap = [&](Float3 tap, Float3 ref) noexcept -> Float3 {
            Float lt = luminance(tap);
            Float lr = max(luminance(ref), 1e-6f);
            return ite(lt > kGatherTapClamp * lr,
                       tap * (kGatherTapClamp * lr / max(lt, 1e-8f)), tap);
        };

        // Radiance gathered by one transmission tap: walks up to
        // kGatherGlassLegs trace legs, refracting deterministically through
        // smooth dielectric hits (simplified single-interface replay — no
        // medium list, no Fresnel bookkeeping: the pixel's chain throughput
        // already carries the smooth product), then queries the cache at the
        // final non-glass hit with the Phase-2 footprint gate. Miss / gates
        // rejected / budget exhausted / TIR keep `fallback` (the denoised PSR
        // background = the smooth-mean value, so the cold cache degrades to
        // today's glass, not to black).
        // Dispersion: `disp_channel` is the tap's picked wavelength channel —
        // every leg crossing refracts at that channel's IOR (the wavelength is
        // constant along the path); the x3 channel-basis tint lives at the
        // tap level, not here.
        // `smooth_replay` (Phase 3, smooth dispersive glass): the caller is a
        // delta-refracted tap, not a GGX lobe sample — the footprint gate
        // (which exists to keep lobe queries off content sharper than a
        // voxel, and a delta ray can never pass) is bypassed, and a cache
        // miss / unqueryable hit falls back to a screen-space denoised probe
        // of the final hit before the plain `fallback`.
        auto gather_transmission_ray = [&](Float3 origin, Float3 dir,
                                           Float3 fallback, Float launch_roughness,
                                           UInt disp_channel, Bool smooth_replay) noexcept -> Float3 {
            Float3 radiance = def(fallback);
            Float dist_total = def(0.0f); // lobe spread grows along the path
            Bool done = def(false);
            $for(leg, kGatherGlassLegs) {
                $if(!done) {
                    auto hit = render::trace_closest(accel, make_ray(origin, dir, 0.001f, 1e10f)
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                    $if(hit->miss()) {
                        radiance = clamp_tap(eval_envmap_radiance(dir, env.envmap,
                            env.env_width, env.env_height, env.env_rotation, env_exp),
                            fallback);
                        done = true;
                    } $else {
                        Float tap_t = hit.committed_ray_t;
                        Float3 tap_pos = origin + dir * tap_t;
                        Float3 leg_wo = -dir;
                        dist_total = dist_total + tap_t;

                        Bool tap_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                        tap_proc = hit.is_procedural;
#endif
                        // Mesh-only material lookup (proc hit inst is a TLAS slot,
                        // out of range for instance_buffer — proc taps keep fallback).
                        UInt4 tap_inst = def(make_uint4(0u));
                        Float tap_ior = def(1.5f);
                        Float tap_disp = def(0.f);
                        Bool tap_thin = def(false);
                        Bool tap_glass = def(false);
                        Bool tap_queryable = def(false);
#if NT_ENABLE_PROCEDURAL
                        $if(!tap_proc) {
#endif
                            tap_inst = scene.instance_buffer.read(hit.inst);
                            auto tap_mat = scene.material_buffer.read(Expr{ tap_inst.y & 0xFFu });
                            UInt tap_type = get_effective_bsdf_type(tap_mat);
                            tap_ior = tap_mat.ior;
                            tap_disp = tap_mat.dispersion;
                            tap_thin = (tap_type == 11u);
                            tap_glass = (tap_type == 3u) | (tap_type == 11u);
                            tap_queryable = true;
#if NT_ENABLE_PROCEDURAL
                        };
#endif
                        $if(tap_glass) {
                            // Deterministic crossing (thin: straight through).
                            // Dispersion: channel IOR of the crossed medium.
                            Float3 tap_obj_n = reconstruct_normal(vertex_bindless, tap_inst.z,
                                tap_inst.w, hit.prim, hit.bary);
                            auto tap_xform = scene.instance_transform_buffer.read(hit.inst);
                            Float3 tap_geo_n = transform_normal(tap_xform, tap_obj_n);
                            Bool tap_entering = dot(leg_wo, tap_geo_n) > 0.0f;
                            Float3 tap_refr_n = ite(tap_entering, tap_geo_n, -tap_geo_n);
                            Float tap_cos_i = abs(dot(leg_wo, tap_refr_n));
#if NT_ENABLE_DISPERSION
                            Float tap_ior_eff = ite(tap_disp > 0.0f,
                                dispersed_ior(tap_ior, tap_disp, disp_channel), tap_ior);
#else
                            (void)disp_channel;
                            Float tap_ior_eff = tap_ior;
#endif
                            Float cross_eta = ite(tap_thin, 1.0f,
                                ite(tap_entering, 1.0f / max(tap_ior_eff, 1.0001f), max(tap_ior_eff, 1.0001f)));
                            Float sin2_t = cross_eta * cross_eta * (1.0f - tap_cos_i * tap_cos_i);
                            $if(sin2_t < 1.0f) {
                                Float3 wi_cross = ite(tap_thin, -leg_wo,
                                    refract_dir(leg_wo, tap_refr_n, cross_eta));
                                Float eps_c = max(0.001f * tap_t, 1e-4f);
                                origin = tap_pos - tap_refr_n * eps_c;
                                dir = wi_cross;
                            } $else {
                                done = true; // TIR: keep fallback
                            };
                        } $else {
                            // Final non-glass hit: resolve + query. The
                            // footprint gate serves the ROUGH taps only — a
                            // smooth-replay delta ray has no lobe footprint
                            // and can never pass the spread term, so it is
                            // bypassed there (the sample-count threshold
                            // inside the query still guards empty voxels).
                            Bool tap_served = def(false);
                            $if(tap_queryable) {
                                SurfaceData tap_surface = resolve_surface_from_instance(
                                    _surfaceResolver, vertex_bindless, tex_bindless,
                                    tap_inst, hit.prim, hit.bary,
                                    scene.material_buffer, leg_wo,
                                    scene.instance_transform_buffer.read(hit.inst),
                                                                        0.0f, make_float2(0.0f), 0u, 0u, hit.inst);
                                UInt tap_level = sharc_get_level<SharcEngineLayout>(
                                    tap_surface.position, sharc_grid);
                                Bool tap_eligible = ite(smooth_replay, true,
                                    sharc_query_eligible(
                                        dist_total, launch_roughness, tap_level, sharc_grid));
                                $if(tap_eligible) {
                                    Float3 tap_cached = def(make_float3(0.0f));
                                    Bool tap_hit_cache = sharc_get_cached_radiance(
                                        sharc_cache, sharc_grid, tap_surface.position,
                                        tap_surface.ns, tap_cached);
                                    $if(tap_hit_cache) {
                                        radiance = clamp_tap(tap_cached, fallback);
                                        tap_served = true;
                                        sharc_query_stats.atomic(2u).fetch_add(1u);
                                    } $else {
                                        sharc_query_stats.atomic(3u).fetch_add(1u);
                                    };
                                };
                            };
                            // Smooth-replay fallback tier 2: screen-space
                            // denoised probe of the final hit position. Only
                            // non-glass destinations are probed — the tint
                            // pass writes this image in place, and a non-glass
                            // pixel's write is the identity (its pre-write
                            // content equals its post-write content), so the
                            // in-place read is value-stable; a glass pixel's
                            // write is NOT (attenuation + mirror composite),
                            // so probing one would race this dispatch.
                            $if(smooth_replay & !tap_served) {
                                Float4 probe_clip = camera->view_proj * make_float4(tap_pos, 1.0f);
                                $if(probe_clip.w > 1e-4f) {
                                    Float2 probe_ndc = probe_clip.xy() / probe_clip.w;
                                    Float2 probe_px = (probe_ndc + 1.0f) * 0.5f
                                                    * make_float2(resolution);
                                    Bool probe_in = all(probe_px >= make_float2(0.0f))
                                                  & all(probe_px < make_float2(resolution));
                                    $if(probe_in) {
                                        UInt2 pcoord = min(make_uint2(
                                            cast<UInt>(probe_px.x), cast<UInt>(probe_px.y)),
                                            make_uint2(resolution.x - 1u, resolution.y - 1u));
                                        Bool probe_glass =
                                            (gbuf_vis.read(pcoord).y >> 31u) != 0u;
                                        $if(!probe_glass) {
                                            radiance = clamp_tap(
                                                denoised_input.read(pcoord).xyz(), fallback);
                                        };
                                    };
                                };
                            };
                            done = true;
                        };
                    };
                };
            };
            return radiance;
        };

        // Radiance gathered by one reflection tap: single outward leg; cache
        // hit → blurred scene reflection, miss / gates / glass hit → the
        // per-pixel `fallback` (the stable mirror-env sample the caller
        // computed once — the smooth-glass value; per-tap env samples with
        // the frame-re-seeded tap directions were the speckle source).
        auto gather_reflection_ray = [&](Float3 origin, Float3 wi_r,
                                         Float3 fallback,
                                         Float launch_roughness) noexcept -> Float3 {
            Float3 radiance = def(fallback);
            auto hit = render::trace_closest(accel, make_ray(origin, wi_r, 0.001f, 1e10f)
#if NT_ENABLE_PROCEDURAL
                , proc_bindless
#endif
            );
            // Fallback-tap diagnostic ([4]): counts taps that resolved to the
            // fallback WITHOUT attempting a lookup (escape / glass / proc /
            // footprint-gate reject) — the cases the hit-rate readout's
            // attempted-lookups-only denominator never saw.
            Bool tap_attempted = def(false);
            $if(!hit->miss()) {
                Bool tap_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                tap_proc = hit.is_procedural;
#endif
                Bool tap_queryable = def(false);
                UInt4 tap_inst = def(make_uint4(0u));
                $if(!tap_proc) {
                    tap_inst = scene.instance_buffer.read(hit.inst);
                    auto tap_mat = scene.material_buffer.read(Expr{ tap_inst.y & 0xFFu });
                    UInt tap_type = get_effective_bsdf_type(tap_mat);
                    // Glass tap hits keep the envmap fallback (rough-reflection
                    // through another dielectric is out of v1 scope).
                    tap_queryable = (tap_type != 3u) & (tap_type != 11u);
                };
                $if(tap_queryable) {
                    SurfaceData tap_surface = resolve_surface_from_instance(
                        _surfaceResolver, vertex_bindless, tex_bindless,
                        tap_inst, hit.prim, hit.bary,
                        scene.material_buffer, -wi_r,
                        scene.instance_transform_buffer.read(hit.inst),
                                                0.0f, make_float2(0.0f), 0u, 0u, hit.inst);
                    UInt tap_level = sharc_get_level<SharcEngineLayout>(
                        tap_surface.position, sharc_grid);
                    Bool tap_eligible = sharc_query_eligible(
                        hit.committed_ray_t, launch_roughness, tap_level, sharc_grid);
                    $if(tap_eligible) {
                        tap_attempted = true;
                        Float3 tap_cached = def(make_float3(0.0f));
                        Bool tap_hit_cache = sharc_get_cached_radiance(
                            sharc_cache, sharc_grid, tap_surface.position,
                            tap_surface.ns, tap_cached);
                        $if(tap_hit_cache) {
                            radiance = clamp_tap(tap_cached, fallback);
                            sharc_query_stats.atomic(2u).fetch_add(1u);
                        } $else {
                            sharc_query_stats.atomic(3u).fetch_add(1u);
                        };
                    };
                };
            };
            $if(!tap_attempted) {
                sharc_query_stats.atomic(4u).fetch_add(1u);
            };
            return radiance;
        };

        Float4 rough_info = rough_glass_info.read(coord);
        Bool do_gather = (gather_on != 0u) & (rough_info.x > kRoughGlassEpsilon);
        $if(do_gather) {
            // --- First dielectric interface via the unjittered camera ray
            //     (the same reconstruction the smooth envmap branch uses). ---
            auto cam_ray = camera->generate_ray(Expr{
                (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter });
            auto glass_hit = render::trace_closest(accel, cam_ray
#if NT_ENABLE_PROCEDURAL
                , proc_bindless
#endif
            );
            // Pass through camera-invisible geometry (an invisible light in
            // front of the glass must not hijack the interface replay).
            $for(glass_skip, 4u) {
                Bool glass_hit_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                glass_hit_proc = glass_hit.is_procedural;
#endif
                Bool glass_hit_invisible = def(false);
                $if(!glass_hit->miss() & !glass_hit_proc) {
                    UInt4 gskip = scene.instance_buffer.read(glass_hit.inst);
                    glass_hit_invisible = (gskip.x & kCameraInvisibleFlag) != 0u;
                };
                $if(glass_hit_invisible) {
                    Float gskip_off = max(0.001f * glass_hit.committed_ray_t, 1e-4f);
                    Float3 gskip_pos = cam_ray->origin()
                        + cam_ray->direction() * glass_hit.committed_ray_t;
                    cam_ray = make_ray(gskip_pos + cam_ray->direction() * gskip_off,
                        cam_ray->direction(), 0.0f, 1e10f);
                    glass_hit = render::trace_closest(accel, cam_ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                } $else { $break; };
            };
            $if(!glass_hit->miss()) {
                Float t_first = glass_hit.committed_ray_t;
                Float3 hit_pos = cam_ray->origin() + cam_ray->direction() * t_first;
                Float3 wo = -normalize(cam_ray->direction());

                Float3 glass_ns;
#if NT_ENABLE_PROCEDURAL
                // Procedural hits: glass_hit->inst is the TLAS slot of the proc BLAS
                // (out of range for instance_buffer), and glass_hit->prim is the
                // proc AABB index. Use proc helpers; no instance transform (proc
                // BLAS uses identity TLAS xform — Geometry.cpp:346).
                $if(glass_hit.is_procedural) {
                    UInt packed_proc_prim = (1u << 29u) | glass_hit.local_tri;
                    glass_ns = reconstruct_procedural_normal(
                        proc_bindless, glass_hit->prim, packed_proc_prim,
                        hit_pos, glass_hit.local_bary);
                } $else {
#endif
                    UInt4 inst_data = scene.instance_buffer.read(glass_hit->inst);
                    auto glass_xform = scene.instance_transform_buffer.read(glass_hit->inst);
                    Float3 glass_obj_n = reconstruct_normal(vertex_bindless, inst_data.z, inst_data.w,
                        glass_hit->prim, glass_hit->bary);
                    glass_ns = transform_normal(glass_xform, glass_obj_n);
#if NT_ENABLE_PROCEDURAL
                };
#endif
                Float3 ns = ite(dot(wo, glass_ns) < 0.0f, -glass_ns, glass_ns);

                Float roughness = max(rough_info.x, 0.03f); // kMinRoughness floor
                Float eta = max(rough_info.y, 1e-3f);
                Bool is_thin = rough_info.z > 0.5f;
#if NT_ENABLE_DISPERSION
                // Abbe V of the first interface (side image .w; 0 = off).
                Float abbe = rough_info.w;
                Bool abbe_on = abbe > 0.0f;
                // Glass-side d-line index from the stored first-interface ratio
                // (against air — the common case; nested firsts stay approximate,
                // same as the rest of this replay).
                Float glass_d = ite(eta < 1.0f, 1.0f / max(eta, 1e-3f), eta);
#endif

                Float2 alpha_ggx = roughness_to_alpha(make_float2(roughness));
                Float3x3 basis = make_orthonormal_basis(ns);
                Float3 wo_local = transpose(basis) * wo;
                Float eps = max(0.001f * t_first, 1e-4f);

                // Deterministic per-pixel reflection fallback (speckle-fix
                // Phase A): the stable mirror-env sample, computed once before
                // the tap loops. Reflection taps rejected before a cache hit
                // (escape / glass / footprint gate / lookup miss) resolve to
                // this constant instead of a per-tap env texel — the tap
                // directions re-seed every frame, so those texels flickered
                // (salt-and-pepper). Same value the refl_count == 0 branch and
                // the smooth-glass path use.
                Float3 mirror_env = eval_envmap_radiance(reflect(-wo, ns),
                    env.envmap, env.env_width, env.env_height,
                    env.env_rotation, env_exp);

                // ---- Transmission taps (K GGX-refracted rays through the
                //      first interface; thin: straight-through lobe) ----
                Float3 trans_sum = def(make_float3(0.0f));
                $for(k, trans_taps) {
                    UInt ts = util::xxhash32(make_uint3(
                        coord.x, coord.y + 0x5bd1e995u * cast<UInt>(k), frame_count));
                    auto pcg = util::pcg2d(make_uint2(ts, ts ^ 0x68bc21ebu));
                    Float2 u_tap = make_float2(util::uniform_uint_to_float(pcg.x),
                                               util::uniform_uint_to_float(pcg.y));
#if NT_ENABLE_DISPERSION
                    // Deterministic balanced channel allocation: tap k takes
                    // channel (k + frame) % 3 — K a multiple of 3 gives every
                    // channel exactly K/3 taps each frame (zero channel-count
                    // variance; the old i.i.d. picks left ~58% of frames with
                    // an empty channel — a 3×-amplitude chroma strobe under the
                    // x3 tint on high-contrast content). Non-multiple K stays
                    // balanced ±1 and the rotation equalizes counts over
                    // frames; the per-frame offset also re-pairs channels with
                    // lobe regions so no channel is systematically stuck on
                    // the same slice of the GGX lobe.
                    UInt tap_ch = (k + frame_count) % 3u;
                    Float eta_tap = ite(abbe_on,
                        ite(eta < 1.0f,
                            1.0f / dispersed_ior(glass_d, abbe, tap_ch),
                            dispersed_ior(glass_d, abbe, tap_ch)),
                        eta);
                    // Tint applies to refracted (non-thin) dispersive taps only.
                    Bool tap_dispersed = abbe_on & !is_thin;
                    Float3 tap_tint = ite(tap_dispersed,
                        dispersion_channel_weight(tap_ch), make_float3(1.0f));
#else
                    UInt tap_ch = def(0u);
                    Float eta_tap = eta;
                    Float3 tap_tint = def(make_float3(1.0f));
#endif
                    Float3 wh_local = sample_ggx_wh(wo_local, alpha_ggx, u_tap);
                    Float3 wi_local = def(make_float3(0.0f, 0.0f, -1.0f));
                    Bool tap_ok = def(false);
                    $if(is_thin) {
                        wi_local = -reflect(-wo_local, wh_local);
                        tap_ok = wi_local.z < -1e-4f;
                    } $else {
                        Float cos_i = dot(wo_local, wh_local);
                        Float sin2_t = eta_tap * eta_tap * (1.0f - cos_i * cos_i);
                        tap_ok = sin2_t < 1.0f;
                        wi_local = refract_dir(wo_local, wh_local, eta_tap);
                    };
                    $if(tap_ok) {
                        trans_sum = trans_sum + tap_tint *
                            gather_transmission_ray(
                                hit_pos - ns * eps, basis * wi_local, denoised.xyz(),
                                roughness, tap_ch, false);
                    } $else {
                        // TIR / degenerate tap: smooth-path fallback keeps the
                        // estimator bounded (divisor stays K).
                        trans_sum = trans_sum + denoised.xyz();
                    };
                };
                Float3 gathered_T = trans_sum * (1.0f / max(cast<Float>(trans_taps), 1.0f));

                // ---- Reflection taps (K' GGX-reflected rays; below-horizon
                //      samples are zero-weight and excluded from the divisor) ----
                Float3 refl_sum = def(make_float3(0.0f));
                UInt refl_count = def(0u);
                $for(k, refl_taps) {
                    UInt rs = util::xxhash32(make_uint3(
                        coord.x + 0x85ebca6bu * cast<UInt>(k), coord.y, frame_count ^ 0x9e3779b9u));
                    auto pcg_r = util::pcg2d(make_uint2(rs, rs ^ 0x68bc21ebu));
                    Float2 u_tap = make_float2(util::uniform_uint_to_float(pcg_r.x),
                                               util::uniform_uint_to_float(pcg_r.y));
                    Float3 wh_local = sample_ggx_wh(wo_local, alpha_ggx, u_tap);
                    Float3 wi_r_local = reflect(-wo_local, wh_local);
                    $if(wi_r_local.z > 1e-4f) {
                        refl_sum = refl_sum + gather_reflection_ray(
                            hit_pos + ns * eps, basis * wi_r_local, mirror_env, roughness);
                        refl_count = refl_count + 1u;
                    };
                };
                // All taps below the horizon (grazing silhouette + wide lobe):
                // the smooth mirror envmap value, not black — a zero here
                // flickers against neighboring frames' accepted taps.
                Float3 gathered_R = ite(refl_count > 0u,
                    refl_sum * (1.0f / max(cast<Float>(refl_count), 1.0f)),
                    mirror_env);

                // ---- Compose with the PSR chain quantities (same shape as
                //      the smooth tint: attenuation carries Π(1-F)·absorption,
                //      fresnel the accumulated reflectivity). ----
                Float3 rough_composite = attenuation * gathered_T + fresnel * gathered_R;

                // ---- Temporal accumulation. Reprojection targets the GLASS
                //      surface (prev camera projection of the first-interface
                //      hit — the stored G-buffer motion belongs to the PSR
                //      background and lags the interface under orbit). A
                //      rough-classification check at the history pixel guards
                //      disocclusions; .w is the accumulation length n
                //      (0 = invalid). alpha = max(floor, 1/n): a running mean
                //      over up to 1/floor frames, so the per-frame re-seeded
                //      K-tap estimate actually converges instead of hovering
                //      at a fixed-EWMA noise floor (K=4 taps of a
                //      high-contrast backdrop at fixed alpha 0.2 left a
                //      permanently boiling ~1/3-of-per-frame-variance residue;
                //      2026-09-03 fix). The floor bounds ghosting/lag. ----
                Float2 prev_ndc = camera->project_prev(hit_pos);
                Float2 prev_px = (prev_ndc + 1.0f) * 0.5f * make_float2(resolution);
                Bool hist_in = (gather_reset == 0u)
                    & all(prev_px >= make_float2(0.0f))
                    & all(prev_px < make_float2(resolution));
                Float3 hist_col = def(make_float3(0.0f));
                Bool hist_valid = def(false);
                Float hist_age = def(0.0f);
                $if(hist_in) {
                    UInt2 hcoord = min(make_uint2(cast<UInt>(prev_px.x), cast<UInt>(prev_px.y)),
                                       make_uint2(resolution.x - 1u, resolution.y - 1u));
                    Float4 h = gather_hist_prev.read(hcoord);
                    Float4 h_info = rough_glass_info.read(hcoord);
                    hist_age = h.w;
                    hist_valid = (h.w >= 1.0f)
                        & (h_info.x > kRoughGlassEpsilon)
                        & (abs(h_info.x - rough_info.x) < 0.02f);
                    hist_col = h.xyz();
                };
                Float blend_a = ite(hist_valid,
                    max(gather_alpha, 1.0f / max(hist_age, 1.0f)), 1.0f);
                Float3 accumulated = lerp(hist_col, rough_composite, blend_a);
                // Age in fp16: exact integers up to 2048; the cap only stops
                // growth once 1/n is far below any usable floor.
                Float age_next = ite(hist_valid, min(hist_age + 1.0f, 1024.0f), 1.0f);
                gather_hist_curr.write(coord, make_float4(accumulated, age_next));
                result = accumulated;
            };
        } $else {
#endif
        // Smooth-glass path. Phase 3 (2026-09-06): dispersive pixels
        // additionally replay 3 deterministic per-channel delta-refracted
        // taps post-denoise (below); the replay and the Fresnel mirror share
        // one first-interface trace of the unjittered camera ray.
        // Non-dispersive pixels keep this body bit-identical to the previous
        // shader (the trace condition is exactly `fresnel > 0.001f`).
#if NT_ENABLE_SHARC && NT_ENABLE_DISPERSION
        Bool disp_replay = (gather_on != 0u) & (rough_info.w > 0.0f);
        Bool need_glass_surface = (fresnel > 0.001f) | disp_replay;
#else
        Bool need_glass_surface = (fresnel > 0.001f);
#endif
        $if(need_glass_surface) {

            // Use unjittered ray for stable Fresnel reflection
            auto cam_ray = camera->generate_ray(Expr{
                (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter });
            auto glass_hit = render::trace_closest(accel, cam_ray
#if NT_ENABLE_PROCEDURAL
                , proc_bindless
#endif
            );
            // Pass through camera-invisible geometry (an invisible light in
            // front of the glass must not hijack the interface replay).
            $for(glass_skip, 4u) {
                Bool glass_hit_proc = def(false);
#if NT_ENABLE_PROCEDURAL
                glass_hit_proc = glass_hit.is_procedural;
#endif
                Bool glass_hit_invisible = def(false);
                $if(!glass_hit->miss() & !glass_hit_proc) {
                    UInt4 gskip = scene.instance_buffer.read(glass_hit.inst);
                    glass_hit_invisible = (gskip.x & kCameraInvisibleFlag) != 0u;
                };
                $if(glass_hit_invisible) {
                    Float gskip_off = max(0.001f * glass_hit.committed_ray_t, 1e-4f);
                    Float3 gskip_pos = cam_ray->origin()
                        + cam_ray->direction() * glass_hit.committed_ray_t;
                    cam_ray = make_ray(gskip_pos + cam_ray->direction() * gskip_off,
                        cam_ray->direction(), 0.0f, 1e10f);
                    glass_hit = render::trace_closest(accel, cam_ray
#if NT_ENABLE_PROCEDURAL
                        , proc_bindless
#endif
                    );
                } $else { $break; };
            };

            $if(glass_hit->inst != ~0u) {
                Float t_first = glass_hit.committed_ray_t;
                Float3 hit_pos = cam_ray->origin() + cam_ray->direction() * t_first;
                Float3 wo = -normalize(cam_ray->direction());

                Float3 glass_ns;
#if NT_ENABLE_PROCEDURAL
                // Procedural hits: glass_hit->inst is the TLAS slot of the proc BLAS
                // (out of range for instance_buffer), and glass_hit->prim is the
                // proc AABB index. Use proc helpers; no instance transform (proc
                // BLAS uses identity TLAS xform — Geometry.cpp:346).
                $if(glass_hit.is_procedural) {
                    UInt packed_proc_prim = (1u << 29u) | glass_hit.local_tri;
                    glass_ns = reconstruct_procedural_normal(
                        proc_bindless, glass_hit->prim, packed_proc_prim,
                        hit_pos, glass_hit.local_bary);
                } $else {
#endif
                    UInt4 inst_data = scene.instance_buffer.read(glass_hit->inst);
                    auto glass_xform = scene.instance_transform_buffer.read(glass_hit->inst);
                    Float3 glass_obj_n = reconstruct_normal(vertex_bindless, inst_data.z, inst_data.w,
                        glass_hit->prim, glass_hit->bary);
                    glass_ns = transform_normal(glass_xform, glass_obj_n);
#if NT_ENABLE_PROCEDURAL
                };
#endif
                Float3 ns = ite(dot(wo, glass_ns) < 0.0f, -glass_ns, glass_ns);

#if NT_ENABLE_SHARC && NT_ENABLE_DISPERSION
                // ---- Phase 3: smooth dispersive glass replay. The PSR chain
                //      is d-line (RC1 fix), so the stored background carries
                //      no fringes; this post-denoise pass re-derives them
                //      exactly: 3 deterministic taps (R/G/B) delta-refract
                //      the unjittered camera ray through the first interface
                //      at the channel IOR, gather_transmission_ray replays
                //      later crossings at the same channel and resolves
                //      through the cascade cache → screen-space denoised
                //      probe → d-line denoised. Exact per-channel estimator:
                //      (1/3)·Σ 3·e_c·L_c — the c-th radiance component of the
                //      c-th channel's bent path (fringes read STRONGER than
                //      the old PSR channel-mean; docs/dispersion.md). Zero
                //      per-frame randomness. ----
                $if(disp_replay) {
                    Float abbe = rough_info.w;
                    // Stored d-line first-interface ratio (eta_i/eta_t; the
                    // first crossing always pairs against air).
                    Float eta_d = max(rough_info.y, 1e-3f);
                    Float glass_d = ite(eta_d < 1.0f, 1.0f / eta_d, eta_d);
                    Float cos_i = dot(wo, ns);
                    Float eps_d = max(0.001f * t_first, 1e-4f);
                    Float3 trans_sum = def(make_float3(0.0f));
                    $for(c, 3u) {
                        Float ior_c = dispersed_ior(glass_d, abbe, c);
                        Float eta_c = ite(eta_d < 1.0f, 1.0f / ior_c, ior_c);
                        Float sin2_t = eta_c * eta_c * (1.0f - cos_i * cos_i);
                        Float3 wi_c = refract_dir(wo, ns, eta_c);
                        Float3 L_c = def(denoised.xyz());
                        // First-interface TIR (channel ring): bounded d-line
                        // fallback, same discipline as the gather taps.
                        $if(sin2_t < 1.0f) {
                            L_c = gather_transmission_ray(
                                hit_pos - ns * eps_d, wi_c, denoised.xyz(),
                                0.0f, c, true);
                        };
                        trans_sum = trans_sum + dispersion_channel_weight(c) * L_c;
                    };
                    Float3 dispersed_T = trans_sum * (1.0f / 3.0f);

                    // ---- Temporal accumulation keyed to the GLASS surface
                    //      (same protocol as the rough gather: prev-frame
                    //      projection of the first-interface hit — the stored
                    //      G-buffer motion belongs to the PSR background and
                    //      lags the interface under orbit; class + Abbe + eta
                    //      validation at the history pixel; alpha =
                    //      max(floor, 1/age)). The replay is deterministic,
                    //      so this is a denoising EWMA (screen-probe judder
                    //      under motion), not a mean over random samples. ----
                    Float2 prev_ndc = camera->project_prev(hit_pos);
                    Float2 prev_px = (prev_ndc + 1.0f) * 0.5f * make_float2(resolution);
                    Bool hist_in = (gather_reset == 0u)
                        & all(prev_px >= make_float2(0.0f))
                        & all(prev_px < make_float2(resolution));
                    Float3 hist_col = def(make_float3(0.0f));
                    Bool hist_valid = def(false);
                    Float hist_age = def(0.0f);
                    $if(hist_in) {
                        UInt2 hcoord = min(make_uint2(cast<UInt>(prev_px.x), cast<UInt>(prev_px.y)),
                                           make_uint2(resolution.x - 1u, resolution.y - 1u));
                        Float4 h = gather_hist_prev.read(hcoord);
                        Float4 h_info = rough_glass_info.read(hcoord);
                        hist_age = h.w;
                        // Same smooth dispersive interface: still smooth,
                        // still dispersive, same Abbe, same d-line ratio.
                        hist_valid = (h.w >= 1.0f)
                            & (h_info.x <= kRoughGlassEpsilon)
                            & (h_info.w > 0.0f)
                            & (abs(h_info.w - rough_info.w) < 1e-3f)
                            & (abs(h_info.y - rough_info.y) < 1e-2f);
                        hist_col = h.xyz();
                    };
                    Float blend_a = ite(hist_valid,
                        max(gather_alpha, 1.0f / max(hist_age, 1.0f)), 1.0f);
                    Float3 accumulated = lerp(hist_col, dispersed_T, blend_a);
                    Float age_next = ite(hist_valid, min(hist_age + 1.0f, 1024.0f), 1.0f);
                    gather_hist_curr.write(coord, make_float4(accumulated, age_next));
                    result = attenuation * accumulated;
                };
#endif
                // Add Fresnel-weighted envmap reflection (d-line by design —
                // dispersion in reflection is negligible; docs/dispersion.md).
                $if(fresnel > 0.001f) {
                    Float3x3 env_rot = env.env_rotation;
                    Float3 env_reflected = eval_envmap_radiance(reflect(-wo, ns),
                        env.envmap, env.env_width, env.env_height, env_rot, env_exp);

                    result = result + fresnel * env_reflected;
                };
            };
        };
#if NT_ENABLE_SHARC
        };
#endif

        out_frame.write(coord, make_float4(result, 1.0f));
    });

#if NT_ENABLE_SHARC
    // Zero-fill for the gather-history ping-pong (.w = accumulation length;
    // fresh/resize/toggle-restart must never read garbage as "valid" — age 0
    // is invalid by construction).
    _roughGlassClearShader = device.compile<2>([&](ImageFloat target) noexcept {
        set_name("rough_glass_clear");
        target.write(dispatch_id().xy(), make_float4(0.0f));
    });
#endif

    // OIT composite: McGuire weighted blended transparency over denoised output
    _oitCompositeShader = device.compile<2>([&](
        ImageFloat output,
        ImageFloat oit_accum,
        ImageFloat oit_log_reveal
    ) noexcept {
        set_name("oit_composite");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(any(coord >= resolution)) { $return(); };

        Float4 accum = oit_accum.read(coord);
        $if(accum.w > 1e-8f) {
            Float  log_rev = oit_log_reveal.read(coord).x;

            Float  revealage = exp(log_rev);   // recover Π(1-α)
            Float3 trans_color = accum.xyz() / accum.w;
            Float3 denoised = output.read(coord).xyz();
            Float3 result = trans_color * (1.0f - revealage) + denoised * revealage;
            output.write(coord, make_float4(result, 1.0f));
        };
    });

    if (_requireExplicitBlit) {
        _blitFltShader = device.compile<2>([&](
            ImageFloat  out_frame,
            ImageFloat  src_frame
            ) noexcept {
            set_name("blit_flt");
            UInt2 coord = dispatch_id().xy();
            out_frame.write(coord, src_frame.read(coord));
        });
        _blitUIntShader = device.compile<2>([&](
            ImageUInt  out_frame,
            ImageUInt  src_frame
            ) noexcept {
            set_name("blit_uint");
            UInt2 coord = dispatch_id().xy();
            out_frame.write(coord, src_frame.read(coord));
        });
    }

    // Bilinear stretch blit (upscaler emergency fallback) — compiled
    // unconditionally: unlike the explicit-blit pair above, the fallback must
    // exist whenever an upscaler mode is selected. Luisa images are
    // texel-read only, so the filtering is an explicit 4-tap.
    _stretchBlitShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageFloat src_frame,
        UInt       src_width,
        UInt       src_height
        ) noexcept {
        set_name("blit_stretch");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        Float2 src_size_f = make_float2(cast<Float>(src_width), cast<Float>(src_height));
        Float2 srcf = (make_float2(coord) + 0.5f) / make_float2(resolution)
                    * src_size_f - 0.5f;
        UInt2 p0 = make_uint2(
            cast<UInt>(max(floor(srcf.x), 0.0f)),
            cast<UInt>(max(floor(srcf.y), 0.0f)));
        UInt2 p1 = make_uint2(min(p0.x + 1u, src_width - 1u),
                              min(p0.y + 1u, src_height - 1u));
        Float2 f = clamp(srcf - make_float2(p0), 0.0f, 1.0f);
        Float4 c00 = src_frame.read(p0);
        Float4 c10 = src_frame.read(make_uint2(p1.x, p0.y));
        Float4 c01 = src_frame.read(make_uint2(p0.x, p1.y));
        Float4 c11 = src_frame.read(p1);
        out_frame.write(coord,
            lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y));
    });

    // Progressive (offline) accumulation: running average of the composite
    // HDR output. Samples are clamped like the shade writes (radiance clamp
    // [0,256]) so a stochastic firefly cannot poison the average with NaN/Inf.
    _progressiveAccumShader = device.compile<2>([&](
        ImageFloat out_accum,   // this frame's accumulator slot
        ImageFloat src_sample,  // current composite HDR sample
        ImageFloat prev_accum,  // previous frame's accumulator slot
        UInt       frame_idx    // 0 = seed
    ) noexcept {
        set_name("progressive_accum");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(any(coord >= resolution)) { $return(); };

        Float4 cur = clamp(src_sample.read(coord), 0.0f, 256.0f);
        Float4 outv = def(cur);
        $if(frame_idx > 0u) {
            Float4 prev = clamp(prev_accum.read(coord), 0.0f, 256.0f);
            Float  n = cast<float>(frame_idx);
            outv = (prev * n + cur) / (n + 1.0f);
        };
        out_accum.write(coord, outv);
    });

    // Tone map shader: HDR FLOAT4 → BYTE4 RGBA8 with selectable tone curve
    // Modes: 0=ACES, 1=Hejl, 2=Reinhard, 3=Lottes, 4=Uchimura
    _toneMapBlitShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageFloat hdr_input,
        UInt       tone_map_mode,
        Float      exposure,
        Float      gamma
    ) noexcept {
        //set_block_size(16u, 16u, 1u);
        set_name("tone_map_blit");
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        // Guard against partial-block threads
        $if(any(coord >= resolution)) { $return(); };

        Float4 hdr = hdr_input.read(coord);
        Float3 color = hdr.xyz() * exposure;
        
        // ACES (Stephen Hill's fit)
        $if(tone_map_mode == 0u) {
            Float a = def(2.51f);
            Float b = def(0.03f);
            Float c = def(2.43f);
            Float d = def(0.59f);
            Float e = def(0.14f);
            color = clamp((color * (a * color + b)) / (color * (c * color + d) + e),
                          make_float3(0.0f), make_float3(1.0f));
            color = pow(color, 1.0f / gamma);
        }
        // Hejl 2015 (already gamma-corrected)
        $elif(tone_map_mode == 1u) {
            Float3 x = max(color - make_float3(0.004f), make_float3(0.0f));
            color = (x * (6.2f * x + make_float3(0.5f))) /
                    (x * (6.2f * x + make_float3(1.7f)) + make_float3(0.06f));
        }
        // Reinhard extended (with white point = 4.0)
        $elif(tone_map_mode == 2u) {
            Float white2 = def(4.0f * 4.0f);  // white point squared
            auto reinhard = [&white2](Float c) noexcept {
                return (c * (1.0f + c / white2)) / (1.0f + c);
            };
            color = make_float3(reinhard(color.x), reinhard(color.y), reinhard(color.z));
            color = pow(color, 1.0f / gamma);
        }
        // Lottes 2016 (AMD GDC "Advanced Techniques and Optimization of HDR Color Pipelines")
        // Rational function: (x*(a*x+b))/(x*(a*x+c)+b) with a=c coefficient in both
        // numerator/denominator x^2 terms, so ratio → 1 for bright values.
        // Parameters derived from: f(0)=0, f(hdrMax)=1, f(midIn)=midOut
        $elif(tone_map_mode == 3u) {
            Float a      = def(1.6f);
            Float hdrMax = def(4.0f);
            Float midIn  = def(0.18f);
            // When midOut == midIn, simplified:
            Float lottes_b = a * (1.0f - midIn) * hdrMax / (hdrMax - 1.0f);
            Float lottes_c = a * (1.0f - midIn);

            // Operate on max channel to preserve hue
            Float maxC   = max(color.x, max(color.y, color.z));
            Float mapped = (maxC * (a * maxC + lottes_b)) /
                           (maxC * (a * maxC + lottes_c) + lottes_b);
            color = color * (mapped / max(maxC, 1e-6f));
            color = pow(color, 1.0f / gamma);
        }
        // Uchimura 2017
        $else {
            // Uchimura: P=a*l^b / (c*l^d + 1) with parameters
            Float maxBrightness = def(1.0f);
            Float contrast = def(1.0f);
            Float lMin = def(0.0f);
            Float lMax = def(1.0f);
            Float sigma = maxBrightness * 0.01f;

            auto uchimura_curve = [&maxBrightness, &contrast, &sigma](Float x) noexcept {
                // Rewritten to avoid pow() issues with negative values
                Float l0 = ((-sigma) * contrast + maxBrightness) / (maxBrightness - 1.0f);
                Float L = maxBrightness;
                Float S0 = 0.0f;
                Float S1 = l0;
                Float C = 1.0f / l0;

                // Polynomial interpolation between S0 and S1
                Float l = x;
                Float S2 = S1 + C * (l - S0);  // linear segment below l0
                Float l2 = L + C * (l - S0);

                Float cS0 = ite(l < S0, 0.0f, 1.0f);
                Float cS1 = ite(l < S1, 1.0f, 0.0f);
                Float cS2 = ite(l < S1, 0.0f, 1.0f);

                // Simpler form: tone curve below l0, linear above
                Float t = l / max(l0, 1e-6f);
                Float curved = l0 * t / (1.0f - t + sigma);
                Float result = min(curved, l) + max(l - l0, 0.0f) * C;
                return clamp(result, 0.0f, 1.0f);
            };
            color = make_float3(
                uchimura_curve(color.x),
                uchimura_curve(color.y),
                uchimura_curve(color.z));
            color = pow(color, 1.0f / gamma);
        };

        out_frame.write(coord, make_float4(color, 1.0f));
    });

    // LUT-based tonemapping: separate shader to avoid extending the $if/$else chain
    _toneMapLutShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageFloat hdr_input,
        Float      exposure,
        Float      gamma,
        VolumeVar<float> lut_volume
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("tone_map_lut");
        UInt2 coord      = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        // Guard against partial-block threads
        $if(any(coord >= resolution)) { $return(); };

        Float4 hdr = hdr_input.read(coord);
        Float3 color = clamp(pow(hdr.xyz() * exposure, 1.0f / gamma), 0.0f, 1.0f);

        // Map [0,1] → [0, size-1] for trilinear fetch
        UInt3 lutSize = lut_volume.size();
        Float3 lutMax = make_float3(
            cast<float>(lutSize.x) - 1.0f,
            cast<float>(lutSize.y) - 1.0f,
            cast<float>(lutSize.z) - 1.0f);
        Float3 scaled = color * lutMax;
        UInt3 p0 = make_uint3(
            cast<uint>(scaled.x),
            cast<uint>(scaled.y),
            cast<uint>(scaled.z));
        UInt3 one = make_uint3(1u);
        UInt3 p1 = min(p0 + one, lutSize - one);
        Float3 frac = scaled - make_float3(
            cast<float>(p0.x),
            cast<float>(p0.y),
            cast<float>(p0.z));
        
        // Read 8 corners
        // Trilinear interpolation
        Float3 c00 = luisa::compute::lerp(
            lut_volume->read(make_uint3(p0.x, p0.y, p0.z)).xyz(), 
            lut_volume->read(make_uint3(p1.x, p0.y, p0.z)).xyz(), frac.x);
        Float3 c01 = luisa::compute::lerp(
            lut_volume->read(make_uint3(p0.x, p0.y, p1.z)).xyz(), 
            lut_volume->read(make_uint3(p1.x, p0.y, p1.z)).xyz(), frac.x);
        Float3 c10 = luisa::compute::lerp(
            lut_volume->read(make_uint3(p0.x, p1.y, p0.z)).xyz(), 
            lut_volume->read(make_uint3(p1.x, p1.y, p0.z)).xyz(), frac.x);
        Float3 c11 = luisa::compute::lerp(
            lut_volume->read(make_uint3(p0.x, p1.y, p1.z)).xyz(), 
            lut_volume->read(make_uint3(p1.x, p1.y, p1.z)).xyz(), frac.x);
        Float3 c0  = luisa::compute::lerp(c00, c10, frac.y);
        Float3 c1  = luisa::compute::lerp(c01, c11, frac.y);
        color      = luisa::compute::lerp(c0, c1, frac.z);

        out_frame.write(coord, make_float4(color, 1.0f));
    });
}

#if NT_DEBUG_VIZ
void Pipeline::_compileDebugShaders() {
    auto& device = Renderer::device();
    _debugVisShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageUInt  gbuf_vis ) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("DebugVis");
        UInt2 coord      = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();

        // Guard against partial-block threads
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        UInt4  vis     = gbuf_vis.read(coord);
        UInt   inst_id = vis.x;
        UInt   prim_id = vis.y & 0x3FFFFFFFu;
        out_frame.write(coord, make_float4(cast<float>(inst_id % 256)   / 256.f,
                                           cast<float>(prim_id % 65535) / 65535.f, 0.f, 1.f));
    });

    // DebugTag::Motion: motion-vector field. motion_px uses the same NDC ->
    // pixel conversion as the temporal pass so magnitudes read 1:1.
    _debugMotionShader = device.compile<2>([&](
        ImageFloat out_frame,
        ImageFloat gbuf_bary_motion) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("DebugMotion");
        UInt2 coord      = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();

        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float2 motion = gbuf_bary_motion.read(coord).zw();
        Float2 px     = motion * make_float2(resolution) * 0.5f;
        Float3 col = make_float3(
            clamp(0.5f + px.x * (1.0f / 64.0f), 0.0f, 1.0f),
            clamp(0.5f + px.y * (1.0f / 64.0f), 0.0f, 1.0f),
            clamp(luisa::compute::length(px) * (1.0f / 64.0f), 0.0f, 1.0f));
        out_frame.write(coord, make_float4(col, 1.0f));
    });

    // DI debug: visualize reservoir state.
    // In checkerboard mode the reservoir buffer is half-width — dispatchs at
    // half-width and uses the same rsv->coord mapping as the shade shader so
    // the buffer read lands on the correct slot for the active screen pixel.
    _diDebugReservoirShader = device.compile<2>([&](
        ImageFloat output,
        BufferVar<Reservoir> reservoir_buffer,
        ImageFloat gbuf_depth,
        ImageUInt  gbuf_vis,
        UInt       cbField
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("DI_DebugReservoir");
        UInt2 rsv = dispatch_id().xy();
        UInt2 rsv_res = dispatch_size().xy();
        UInt2 coord;
        UInt2 resolution;
        UInt pixel_index;
        if (_checkerboardEnabled) {
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

        Float depth = gbuf_depth.read(coord).x;
        UInt4 vis = gbuf_vis.read(coord);
        UInt inst_id = vis.x;

        // Sky pixel: dark blue
        $if(depth > 1e10f) {
            output.write(coord, luisa::make_float4(0.0f, 0.0f, 0.2f, 1.0f));
            $return();
        };

        // No geometry: black
        $if(inst_id == ~0u) {
            output.write(coord, luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f));
            $return();
        };

        Var<Reservoir> r = reservoir_buffer.read(pixel_index);
        // P1-9: split the debug states. After the empty-but-alive semantics
        // change (M>0 no longer implies a selected sample), a single red
        // "invalid" channel would conflate three different situations:
        //   green  = valid sample (M>0, light selected): G = min(W,1), B = M/64
        //   orange = alive-but-empty (M>0, no sample — transient, expected at
        //            AfterCandidate on env-only surfaces awaiting reuse rescue)
        //   red    = M == 0 (no candidates considered / boiling kill — should
        //            be rare on geometry pixels after the fix)
        Bool has_sample = r->is_valid() & (r.light_idx != ~0u);
        Bool alive_empty = r->is_valid() & (r.light_idx == ~0u);

        Float green = ite(has_sample, luisa::compute::min(r->weight(), 1.0f), 0.0f);
        Float blue  = ite(has_sample, cast<float>(r->M()) / 64.0f, 0.0f);
        Float red   = ite(alive_empty, 1.0f, ite(has_sample, 0.0f, 0.8f));
        Float orange_mix = ite(alive_empty, 0.5f, 0.0f);

        output.write(coord, make_float4(red, max(green, orange_mix), blue, 1.0f));
    });

    // GI debug: visualize GI reservoir state.
    // R = invalid flag (0.8 if invalid), G = W (capped at 1), B = M/64, A = age/30
    // Same checkerboard handling as DI debug above.
    _giDebugReservoirShader = device.compile<2>([&](
        ImageFloat output,
        BufferVar<GIReservoir> reservoir_buffer,
        ImageFloat gbuf_depth,
        ImageUInt  gbuf_vis,
        UInt       cbField
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("GI_DebugReservoir");
        UInt2 rsv = dispatch_id().xy();
        UInt2 rsv_res = dispatch_size().xy();
        UInt2 coord;
        UInt2 resolution;
        UInt pixel_index;
        if (_checkerboardEnabled) {
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

        Float depth = gbuf_depth.read(coord).x;
        UInt4 vis = gbuf_vis.read(coord);
        UInt inst_id = vis.x;

        // Sky pixel: dark blue
        $if(depth > 1e10f) {
            output.write(coord, luisa::make_float4(0.0f, 0.0f, 0.2f, 1.0f));
            $return();
        };

        // No geometry: black
        $if(inst_id == ~0u) {
            output.write(coord, luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f));
            $return();
        };

        Var<GIReservoir> r = reservoir_buffer.read(pixel_index);
        Bool valid = r->is_valid();

        Float green = ite(valid, luisa::compute::min(r->weight() * (1.0f / 20.0f), 1.0f), 0.0f);
        Float blue  = ite(valid, cast<float>(r->M()) / 64.0f, 0.0f);
        Float alpha = ite(valid, cast<float>(r->age()) / 30.0f, 0.0f);
        Float red   = ite(valid, 0.0f, 0.8f);

        output.write(coord, make_float4(red, green, blue, alpha));
    });
}
#endif // NT_DEBUG_VIZ

//==============================================================================
// Custom Material Callable DLL (Debug builds — machinery compiled into every
// _DEBUG binary so the prebuilt Debug lib serves Debug_Runtime consumers)
//==============================================================================

#ifdef _DEBUG
void Pipeline::loadCallableDLL() {
    if (isBuilt()) {
        CI_LOG_W("Pipeline::loadCallableDLL called after buildScene() - custom "
            "material callables are baked into the $switch dispatch at shader "
            "compile time. Call _recompileAllShaders() after registering new "
            "callables, or call loadCallableDLL() before buildScene().");
    }
    // Host-side registration functions passed to the DLL.
    // These bridge the DLL's callable registration into the Pipeline's
    // Polymorphic<SurfaceResolver> (avoids per-DLL singleton issues).
    // Uses a static pointer so the function pointers can be plain C function ptrs.
    static SurfaceResolverPoly* s_poly = nullptr;
    s_poly = &_surfaceResolver;
    static render::MaterialPool* s_params_pool = nullptr;
    s_params_pool = _materialPool.get();
    static auto hostRegister = [](const char* name, std::any resolve) -> std::uint32_t {
        if (!resolve.has_value() || !s_poly) return ~0u;
        const auto& fn = std::any_cast<const SurfaceResolveFn&>(resolve);
        return s_poly->create<CustomSurfaceResolver<SurfaceResolveFn>>(fn);
    };
    // ABI v2: params descriptors -> MaterialPool (name-keyed, values survive
    // reloads). Returns the callable's float4 base for the resolver to capture.
    static auto hostRegisterParams =
        [](const char* callable, const runtime::HostResolverParamDesc* descs,
           std::uint32_t count) -> std::uint32_t {
        if (!s_params_pool || !descs || count == 0u) return ~0u;
        luisa::vector<render::MaterialPool::ResolverParamDesc> hostDescs;
        hostDescs.reserve(count);
        for (std::uint32_t i = 0u; i < count; i++) {
            hostDescs.push_back({descs[i].name, descs[i].min_v, descs[i].max_v, descs[i].def_v});
        }
        return s_params_pool->registerResolverParams(callable, hostDescs);
    };
    static auto hostClear = []() {
        if (!s_poly) return;
        // Reset and re-register built-ins (0-13). Params entries are NOT
        // cleared: current values survive the reload (plan contract); the
        // re-registration re-maps by name.
        *s_poly = SurfaceResolverPoly{};
        for (uint i = 0; i <= 13u; ++i) {
            s_poly->create<IdentitySurfaceResolver>();
        }
    };

    _callableDLL.setHostFunctions(hostRegister, hostClear, hostRegisterParams);

    std::string dllPath    = "build/Runtime/x64/Debug_Runtime/CustomMaterialShader.dll";
    std::string sourcePath = "../runtime_shaders/CustomMaterialShader/CustomMaterialShader.cpp";

    if (_callableDLL.load(dllPath, sourcePath)) {
        CI_LOG_I("Pipeline: Custom material callable DLL loaded successfully");
    } else {
        CI_LOG_W("Pipeline: Failed to load custom material callable DLL: "
                 << _callableDLL.getLastError());
    }
}
#endif

//==============================================================================
// Shader Recompilation (for callable DLL hot-reload)
//==============================================================================

void Pipeline::_recompileAllShaders() {
    auto& device = Renderer::device();

    CI_LOG_I("Pipeline: Recompiling resolver-dependent shaders...");

    // Only resolver-dependent sub-shaders need recompile on a custom-callable
    // DLL reload. Resolver-free shaders (utility, debug, presample, boiling,
    // upsample, denoiser prepass/temporal/history/atrous) hit the disk cache
    // unchanged and are skipped here.
    const bool transparent = _geom != nullptr && _geom->has_transparent_shadow_casters();
    auto f_di       = std::async(std::launch::async, [&] { _passDI.recompileCallables(device, _surfaceResolver); });
    auto f_shade    = std::async(std::launch::async, [&] { _compileShadeShader(); });
    auto f_gi       = std::async(std::launch::async, [&] { _passGI.recompileCallables(device, _surfaceResolver, transparent); });
    auto f_denoiser = std::async(std::launch::async, [&] { _denoiser.recompileCallables(device, _surfaceResolver); });
    f_di.get(); f_shade.get(); f_gi.get(); f_denoiser.get();

    CI_LOG_I("Pipeline: Shader recompilation complete");
}

void Pipeline::_refreshSpecialization() {
    if (!isBuilt()) return;
    auto& device = Renderer::device();

    const bool transparent = _geom != nullptr && _geom->has_transparent_shadow_casters();
    const bool di_changed    = _passDI.specializationChanged();
    const bool gi_changed    = _passGI.specializationChanged(transparent);
    const bool shade_changed =
        _shadeBakedTransparentShadowCasters != (transparent ? 1u : 0u);
    if (!di_changed && !gi_changed && !shade_changed) return;

    CI_LOG_I("Pipeline: Specialized flag flip detected (DI="
        << di_changed << " GI=" << gi_changed << " Shade=" << shade_changed
        << ") - recompiling affected shaders...");

    // Same recompile path as a callable DLL hot-reload (must run on the
    // render thread / before dispatches — callers guarantee that). Only the
    // passes whose flags changed pay the DXC compile; warm shader-cache
    // entries make revisited variants fast.
    if (di_changed && shade_changed && gi_changed) {
        auto f_di    = std::async(std::launch::async, [&] { _passDI.recompileCallables(device, _surfaceResolver); });
        auto f_shade = std::async(std::launch::async, [&] { _compileShadeShader(); });
        auto f_gi    = std::async(std::launch::async, [&] { _passGI.recompileCallables(device, _surfaceResolver, transparent); });
        f_di.get(); f_shade.get(); f_gi.get();
    } else {
        if (di_changed)    _passDI.recompileCallables(device, _surfaceResolver);
        if (shade_changed) _compileShadeShader();
        if (gi_changed)    _passGI.recompileCallables(device, _surfaceResolver, transparent);
    }

    CI_LOG_I("Pipeline: Specialization recompile complete");
}

void Pipeline::_applyCheckerboardReconfigure() {
    // Full drain before destroying images: beginFrame() only synced the
    // render stream, but the recreated G-buffer/reservoir images are touched
    // by every stream tier (resize()'s envelope — destroy-under-flight TDRs,
    // docs/resize_tdr_root_cause.md).
    Renderer::stream() << synchronize();
    _computeStream       << synchronize();
    _bufferStream        << synchronize();
#if NT_ALLOW_RASTER_FEATURES
    if (_rasterContext) _rasterContext->rasterStream() << synchronize();
#endif
    _frameSubmitted = false;

    CI_LOG_I("Pipeline: checkerboard " << (_checkerboardBaked ? "on" : "off")
        << " -> " << (_checkerboardEnabled ? "on" : "off")
        << " - recompiling DI/GI/shade and recreating reservoir images"
        << (_checkerboardEnabled ? "" :
            " (DLSS-RR requires dense per-pixel samples - DLSS-RR "
            "Integration Guide §3.5 lists checkerboard among the practices "
            "to avoid)") << "...");

    _checkerboardBaked = _checkerboardEnabled;

    auto& device = Renderer::device();
    const bool transparent =
        _geom != nullptr && _geom->has_transparent_shadow_casters();
    {
        // Full compile (not recompileCallables): the checkerboard layout is
        // baked into the boiling/upsample/presample kernels as well, which
        // the resolver-only fast path skips.
        auto f_di    = std::async(std::launch::async, [&] { _passDI.compile(device, *_geom, _surfaceResolver, _checkerboardEnabled); });
        auto f_shade = std::async(std::launch::async, [&] { _compileShadeShader(); });
        auto f_gi    = std::async(std::launch::async, [&] { _passGI.compile(device, _surfaceResolver, _checkerboardEnabled, transparent); });
        f_di.get(); f_shade.get(); f_gi.get();
    }

    // Reservoir buffers are half-width in checkerboard mode — recreate at the
    // new layout (zero-filled inside; same sequence as resize()). Dims are
    // unchanged, so feature images / temp pool / upscaler contexts are
    // unaffected.
    _createImages(_width, _height);
    _initSeedImage();
    _denoiser.resetFrameIdx();
    requestUpscalerReset();
    requestAccumReset();
    requestProgressiveReset();
    CI_LOG_I("Pipeline: checkerboard reconfigure complete");
}

//==============================================================================
// Features
//==============================================================================

void Pipeline::addFeature(std::unique_ptr<IFeature> feature) {
    if (!isBuilt()) {
        CI_LOG_W("Pipeline::addFeature called before buildScene() - onInit "
            "may touch resources that have not been created yet. Recommended "
            "order: buildScene() -> addFeature().");
    }
    auto pt = feature->point();
    auto idx = static_cast<size_t>(pt);
    CI_LOG_D("Pipeline: Adding feature at point " << idx);
    feature->onInit(Renderer::device());
    // Feature-owned images must exist before the first onExecute without
    // relying on a resize event: the startup resize is a no-op now (see
    // Pipeline::resize), so the first size-dependent allocation happens here.
    feature->onResize(Renderer::device(), _width, _height);
    _features[idx].push_back(std::move(feature));
}

Image<float>& Pipeline::requestTempImage(PixelStorage format, uint width, uint height, std::string_view id) {
    if (!id.empty()) {
        for (auto& entry : _tempImages) {
            if (entry.id == id) {
                if (entry.format != format || entry.width != width || entry.height != height) {
                    entry.format = format;
                    entry.width = width;
                    entry.height = height;
                    entry.image = Renderer::device().create_image<float>(format, width, height);
                }
                return entry.image;
            }
        }
    }
    auto& device = Renderer::device();
    _tempImages.push_back({format, width, height, std::string(id), device.create_image<float>(format, width, height)});
    return _tempImages.back().image;
}

void Pipeline::_dispatchFeatures(Stream& stream, FeaturePoint point,
                                  const FrameContext& frame, Image<float>& target) {
    auto& list = _features[static_cast<size_t>(point)];
    if (list.empty()) return;
    FeatureContext fctx{frame, target, Renderer::device(), *this, point};

    // Non-perspective projection guard: features that cannot express the
    // projection (fixed-function raster clip space, view_proj overlays) are
    // skipped while a panoramic/room-rig camera is active.
    const bool perspectiveOnly =
        frame.camera.projection != static_cast<uint32_t>(util::CameraProjection::Perspective);

#if NT_ALLOW_RASTER_FEATURES
    if (point == FeaturePoint::AfterGBuffer) {
        // Collect active RasterBase features for batch dispatch
        luisa::vector<feature::RasterBase*> rasterFeatures;
        for (auto& feat : list) {
            if (!feat->enabled()) continue;
            if (perspectiveOnly && feat->requiresPerspectiveProjection()) {
                static bool sWarnedRasterProjection = false;
                if (!sWarnedRasterProjection) {
                    sWarnedRasterProjection = true;
                    CI_LOG_W("Non-perspective projection active: raster features "
                             "(point cloud / trails / OIT) are disabled this session "
                             "(docs/non_perspective_camera_report.md raster guard)");
                }
                continue;
            }
            if (auto* rb = dynamic_cast<feature::RasterBase*>(feat.get())) {
                if (rb->shouldExecute())
                    rasterFeatures.push_back(rb);
            } else {
                feat->onExecute(stream, fctx);
            }
        }
        if (!rasterFeatures.empty())
            _rasterContext->dispatchBatch(stream, fctx, rasterFeatures);
        return;
    }
#endif

    for (auto& feature : list) {
        if (feature->enabled()) {
            if (perspectiveOnly && feature->requiresPerspectiveProjection())
                continue;
            feature->onExecute(stream, fctx);
        }
    }
}

} // namespace newtype::core
