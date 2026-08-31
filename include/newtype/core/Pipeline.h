#pragma once
#include "newtype/core/Config.h"
#include "cinder/Json.h"
#include "newtype/scene/Geometry.h"
#include "newtype/scene/LightShape.h"
#include "newtype/scene/InstancedMesh.h"
#include "newtype/scene/VoxelGrid.h"
#include "newtype/core/IVoxelGridUser.h"
#include "newtype/render/GIShading.h"
#include "newtype/render/PassDenoiser.h"
#include "newtype/render/PassGI.h"
#include "newtype/core/BindingGroups.h"
#include "newtype/render/PassDI.h"
#include "newtype/render/PassSSS.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/FeaturePoint.h"
#include "newtype/core/IFeature.h"
#include "newtype/util/Profiler.h"
#include "newtype/render/SurfaceResolver.h"
#include "Renderer.h"
#include <luisa/core/fiber.h>
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif
#if NT_ALLOW_RASTER_FEATURES
#include "newtype/feature/RasterContext.h"
#endif

#ifdef RT_RUNTIME
#include "newtype/runtime/CallableDLLLoader.h"
#endif

namespace newtype {
namespace core {
using namespace newtype::render;

class Pipeline;
typedef luisa::unique_ptr<Pipeline> PipelinePtr;

//==========================================================================
// Pipeline-level Shader Type Aliases
//==========================================================================

// ReSTIR shade: read reservoir -> shadow ray -> raw output
using ShadeShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (raw shade color)
    luisa::compute::Buffer<Reservoir>,  // 1: reservoir_buffer
    luisa::compute::Image<float>,      // 2: gbuf_depth
    luisa::compute::Image<uint>,       // 3: gbuf_vis
    luisa::compute::Image<float>,      // 4: gbuf_bary_motion (reads .xy() for barycentrics)
    luisa::uint,                       // 5: frame_count
    luisa::compute::Accel,             // 6: TLAS (shadow rays)
    newtype::util::CameraData,         // 7: camera
    SceneGeometryResources,            // 8: instance + transform + material buffers
    luisa::compute::BindlessArray,     // 9: vertex_bindless
    luisa::compute::BindlessArray,     // 10: tex_bindless (material textures)
    LightSamplingResources,            // 11: triangle lights + alias table + emissive counts
    luisa::compute::Buffer<GIReservoir>, // 12: GI reservoirs
    EnvLightResources,                 // 13: envmap + CDFs + rotation
    float,                             // 14: env_exposure
    luisa::uint,                       // 15: cbField (checkerboard, 0=off)
    luisa::uint,                       // 16: debugVizMode (0=off)
    luisa::compute::Image<float>,      // 17: specular output (.rgb=spec radiance, .a=hitDist)
    luisa::compute::Image<float>,      // 18: glass_throughput (.rgb=attenuation, .a=fresnel_accum)
    luisa::compute::Image<float>,      // 19: albedo_output (xyz=diffuse demod factor, w=metallic*0.49 or 1.0 for emissive)
    luisa::compute::Image<float>,      // 19b: spec_factor_output (xyz=NRD specular demod factor)
    luisa::uint,                       // 20: visMaxAge (visibility reuse max age)
    float,                             // 21: visMaxDistance (max screen-space px for vis reuse)
    float,                             // 22: endVisMaxDistance (max screen-space px for env vis reuse)
    luisa::uint                        // 23: hasTransparentShadowCasters (0=any-hit fast path, 1=full transparent loop)
#if NT_ALLOW_RASTER_FEATURES
    ,
    newtype::scene::VoxelGrid::Resources,  // 24: voxel grid binding group
    luisa::compute::Image<float>,          // 25: shadow cache prev (read)
    luisa::compute::Image<float>           // 26: shadow cache (write)
#endif
    ,
    luisa::compute::Buffer<GIReservoir>,   // 24/27: GI initial-reservoir snapshot (frozen, for MIS)
    float,                                 // 25/28: giMISRoughness (0=disable MIS, RTXDI default 0.3)
    luisa::uint                            // 26/28: deltaBranchNEEEnabled (1=run NEE at mirror-hit x2)
#if NT_ENABLE_BSSRDF
    ,
    luisa::compute::Image<float>           // SSS radiance (HALF4, per-channel demodulated)
#endif
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray           // procedural bindless
#endif
>;

// Compositing blit: (denoised diffuse + specular) * albedo for geometry, envmap for sky
using CompositeBlitType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (display)
    luisa::compute::Image<float>,      // 1: denoised diffuse (HALF4)
    luisa::compute::Image<float>,      // 2: gbuf_depth (R32F)
    EnvLightResources,                 // 3: envmap + CDFs + rotation
    newtype::util::CameraData,         // 4: camera (for ray direction)
    float,                             // 5: env_exposure
    luisa::compute::Image<float>,      // 6: albedo (HALF4, .rgb=diffuse demod factor, .w=emissive flag)
    luisa::compute::Image<float>,      // 7: spec demod factor (HALF4)
    luisa::compute::Image<float>,      // 8: denoised specular (HALF4)
    luisa::uint,                       // 9: solid_bg_enabled (0 or 1)
    luisa::float3                      // 10: solid_bg_color
>;

#if NT_DEBUG_VIZ
// Debug shader: visualize visibility buffer (inst_id, prim_id)
using DebugVisibilityShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output
    luisa::compute::Image<uint>        // 1: visibility buffer input
>;

// DI debug: visualize reservoir state (valid=green, invalid=red, weight=brightness)
using DIDebugReservoirType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output
    luisa::compute::Buffer<Reservoir>, // 1: reservoir buffer
    luisa::compute::Image<float>,      // 2: gbuf_depth
    luisa::compute::Image<uint>,       // 3: gbuf_vis
    uint                               // 4: cbField
>;

// GI debug: visualize GI reservoir state.
// R = invalid flag (0.8 if invalid), G = W (capped at 1), B = M/64, A = age/30
using GIDebugReservoirType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,         // 0: output
    luisa::compute::Buffer<GIReservoir>,  // 1: GI reservoir buffer
    luisa::compute::Image<float>,         // 2: gbuf_depth
    luisa::compute::Image<uint>,          // 3: gbuf_vis
    uint                                  // 4: cbField
>;
#endif

using BlitShaderFltType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output
    luisa::compute::Image<float>       // 1: src
>;

using BlitShaderUIntType = luisa::compute::Shader<2,
    luisa::compute::Image<uint>,      // 0: output
    luisa::compute::Image<uint>       // 1: src
>;

// Glass tint: apply glass attenuation + Fresnel reflection on denoised output
using GlassTintShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (display)
    luisa::compute::Image<float>,      // 1: denoised input (read+write in-place)
    luisa::compute::Image<float>,      // 2: glass_throughput (RGBA16F)
    luisa::compute::Image<float>,      // 3: gbuf_depth (R32F)
    luisa::compute::Image<uint>,       // 4: gbuf_vis (RG32U)
    luisa::compute::Image<float>,      // 5: gbuf_bary_motion (RGBA16F)
    newtype::util::CameraData,         // 6: camera
    EnvLightResources,                 // 7: envmap + CDFs + rotation
    float,                             // 8: env_exposure
    SceneGeometryResources,            // 9: instance + transform + material buffers
    luisa::compute::BindlessArray,     // 10: vertex_bindless
    luisa::compute::Accel              // 11: TLAS (trace camera ray for glass normal)
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // procedural bindless
#endif
>;

// OIT composite: McGuire weighted blended transparency over denoised output
using OITCompositeShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (denoised + transparent, in-place)
    luisa::compute::Image<float>,      // 1: oit_accum (HALF4: RGB=α·c·w, A=α·w)
    luisa::compute::Image<float>       // 2: oit_log_reveal (FLOAT: Σ log(1-α))
>;

// Tone map: HDR float → RGBA8 with selectable tone curve
using ToneMapShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (BYTE4 storage)
    luisa::compute::Image<float>,      // 1: HDR input (FLOAT4)
    luisa::uint,                       // 2: tone_map_mode (0-4)
    float,                             // 3: exposure
    float                              // 4: gamma
>;

// Tone map LUT: HDR float → RGBA8 via 3D LUT with trilinear sampling
using ToneMapLutShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (BYTE4 storage)
    luisa::compute::Image<float>,      // 1: HDR input (FLOAT4)
    float,                             // 2: exposure
    float,                             // 3: gamma
    luisa::compute::Volume<float>  // 4: LUT 3D texture
>;

//==============================================================================
// Pipeline Class
//==============================================================================

class Pipeline {
public:
    // Sentinel for environment light (shared between init and render TUs)
    static constexpr uint kEnvLightSentinel = ~1u;  // 0xFFFFFFFE

    enum class DebugTag {
        None, Depth, Visibility, BaryCentric, Motion, PDF, Jacobian, Normal, Glass, GIReservoir
    };

    // DI debug: bypass after a specific pass, write intermediate output to screen
    enum class DIDebugPass {
        Off,             // normal rendering
        AfterCandidate,  // show reservoir state after candidate generation
        AfterTemporal,   // show reservoir state after temporal reuse
        AfterBoiling,    // show reservoir state after boiling filter
        AfterSpatial,    // show reservoir state after spatial reuse
        AfterShade,      // show raw shade output (no denoiser)
        Count
    };

    // Tone mapping curve selection (for RGBA8 output mode)
    enum class ToneMapMode : uint32_t { ACES = 0, Hejl = 1, Reinhard = 2, Lottes = 3, Uchimura = 4, LUT = 5 };

private:
    //==============================================================================
    // Light Sampling Mode (compile-time toggle)
    //==============================================================================
    static constexpr bool kUniformLightSampling = false;

    // --- Scene ---
    scene::GeomPtr                          _geom;
    std::unique_ptr<render::MaterialPool>   _materialPool;
    std::unique_ptr<render::LightSampler>   _lightSampler;

    luisa::compute::Stream                  _computeStream;
    luisa::compute::Stream                  _bufferStream;

    // --- G-Buffer images (16 bytes/pixel) ---
    luisa::compute::Image<float>  _gbufDepth;      // R32F    - linear ray t
    luisa::compute::Image<uint>   _gbufVis;        // RG32U   - (inst_id, prim_id)
    luisa::compute::Image<float>  _gbufBaryMotion; // RGBA16F - RG=barycentrics, BA=motion vectors
    luisa::compute::Image<float>  _glassThroughput; // RGBA16F - RGB=attenuation*(1-F), A=Fresnel reflectivity

    // --- Accumulation ---
    luisa::compute::Image<float>  _accumBuffer;    // FLOAT4 (raw shade output, diffuse)
    luisa::compute::Image<float>  _specularBuffer; // HALF4 (.rgb=specular radiance, .a=hitDist)
    luisa::compute::Image<uint>   _seedImage;      // INT1

    // --- ReLAX Denoiser (self-contained class) ---
    RelaxDenoiser _denoiser;

    // --- PassDI (self-contained class) ---
    PassDI _passDI;

    // --- PassGI (self-contained class) ---
    PassGI _passGI;

    // --- PassSSS (self-contained class, BSSRDF probe — gated by NT_ENABLE_BSSRDF) ---
#if NT_ENABLE_BSSRDF
    PassSSS _passSSS;
#endif

    // --- Voxel Grid (shared by raster features with IVoxelGridUser) ---
#if NT_ALLOW_RASTER_FEATURES
    luisa::unique_ptr<scene::VoxelGrid> _voxelGrid;
    luisa::unique_ptr<feature::RasterContext> _rasterContext;
#endif

#if NT_ENABLE_PROCEDURAL
    // Procedural primitive system
    luisa::unique_ptr<scene::ProceduralGeometry> _procGeom;
    luisa::compute::BindlessArray _procBindless;
    // Placeholder buffers (used only when _procGeom is null)
    luisa::compute::Buffer<scene::ProcInstanceData> _procInstanceBuf;
    luisa::compute::Buffer<luisa::float4> _vatPosBuf;
    luisa::compute::Buffer<compute::Triangle> _vatIdxBuf;
    luisa::compute::Buffer<compute::AABB> _procAabbBuf;
    luisa::compute::Buffer<luisa::float4> _vatNormalsBuf;
    luisa::compute::Buffer<scene::ProcDeformState> _deformStateBuf;
#endif

    // --- Async feature G-buffer event (e.g., PointCloud raster merge) ---
    luisa::compute::TimelineEvent _featureGbufEvent;
    uint64_t _featureGbufFence = 0u;

    // --- Render-ready event: Renderer::stream() signals after G-buffer + presample
    //     so _rasterStream can safely read shared buffers (env CDF, material, etc.) ---
    luisa::compute::TimelineEvent _renderReadyEvent;
    uint64_t _renderReadyFence = 0u;

    // --- Frame-tail event for async compute/render overlap ---
    luisa::compute::TimelineEvent _frameTailEvent;
    uint64_t _frameTailFence = 0u;

    // --- Light-sampler-ready event: Renderer::stream() signals after the shade
    //     pass finishes reading _triangle_lights / _alias_table; _computeStream
    //     waits before update_weights() rewrites them. Avoids the cross-queue
    //     write-while-read race that previously caused DX12 device removal. ---
    luisa::compute::TimelineEvent _lightSamplerReadyEvent;
    uint64_t _lightSamplerReadyFence = 0u;

    // --- Previous frame G-Buffer (for denoiser disocclusion detection) ---
    luisa::compute::Image<float>  _gbufDepthPrev;     // FLOAT1
    luisa::compute::Image<uint>   _gbufVisPrev;       // INT2
    luisa::compute::Image<float>  _denoiseNormalPrev; // HALF4 (previous frame world normal)

#if NT_ALLOW_RASTER_FEATURES
    // --- Temporal shadow cache (DDA env shadow attenuation) ---
    luisa::compute::Image<float>  _shadowCache;       // HALF2 (attenuation, depth)
    luisa::compute::Image<float>  _shadowCachePrev;   // HALF2 previous frame
#endif

    // --- Compiled shaders (Shade stays in Pipeline — reads both DI and GI reservoirs) ---
    ShadeShaderType             _shadeShader;

    // Utility shaders
    CompositeBlitType           _compositeBlitShader;  // for no-denoiser fallback: raw shade + envmap sky
    GlassTintShaderType         _glassTintShader;      // post-denoiser glass Fresnel + absorption
    OITCompositeShaderType      _oitCompositeShader;   // McGuire weighted blended OIT composite
    BlitShaderFltType           _blitFltShader;        // Dx blit
    BlitShaderUIntType          _blitUIntShader;

#if NT_DEBUG_VIZ
    DebugVisibilityShaderType   _debugVisShader;
    DIDebugReservoirType        _diDebugReservoirShader;
    GIDebugReservoirType        _giDebugReservoirShader;
#endif

    // Tone mapping
    ToneMapShaderType           _toneMapBlitShader;
    ToneMapLutShaderType        _toneMapLutShader;

    // --- DI debug: stop after a specific pass and visualize reservoir state ---
    DIDebugPass _diDebugPass = DIDebugPass::Off;
    DebugTag    _debugTag    = DebugTag::None;

    // --- Shade debug viz ---
    int  _shadeDebugVizMode = 0;    // 0=off, 1=W+target_pdf+is_env+spec_lum, 2=M+w_sum+is_env+spec_lum,
                                    // 4=glass, 5=glass detail, 6=shadow, 7=shadow detail, 8=split lum,
                                    // 9=GI W/target_pdf/M/age, 10=GI weight_sum/rad/indirect, 11=GI dist/vis

    // --- Frame state ---
    uint  _frameCount = 0;
    bool  _accumReset = false;
    uint  _width      = 0;
    uint  _height     = 0;
    bool  _frameSubmitted = false;      // true if previous frame's GPU tail is still executing
    FrameResource* _currentFrame = nullptr; // current frame resource (set in beginFrame)

    // --- Fiber scheduler (MARL-based, for future CPU/GPU overlap) ---
    std::unique_ptr<luisa::fiber::scheduler> _fiberScheduler;

    // --- PreUpdate dummy target ---
    luisa::compute::Image<float>  _dummyTarget;   // 1x1 FLOAT4 for PreUpdate feature dispatch

    // --- Shared tunable parameters ---
    float _boilingFilterStrength  = 0.2f;  // 0..1, lower = more aggressive outlier rejection
    uint  _rasterLightSamples     = 4u;    // local light samples for point/trail shading

    // --- Feature toggles ---
    bool _checkerboardEnabled = true;
    bool _requireExplicitBlit = false;
    bool      _solidBackgroundEnabled = false;
    luisa::float3 _solidBackgroundColor = luisa::make_float3(0.02f);

    // --- Tone mapping settings ---
    ToneMapMode _toneMapMode     = ToneMapMode::ACES;
    float       _toneMapExposure = 1.0f;
    float       _toneMapGamma    = 2.2f;

    // --- Tone mapping LUT ---
    luisa::compute::Volume<float>  _toneMapLut;           // 3D texture (FLOAT4, N^3)
    bool                           _toneMapLutEnabled = false;
    std::string                    _toneMapLutPath;        // path to loaded .cube file

    // Auxiliary images for edge-stopping (HALF4)
    luisa::compute::Image<float>  _denoiseAlbedo;       // material albedo (no lighting)
    luisa::compute::Image<float>  _denoiseSpecFactor;   // NRD specular demod factor
    luisa::compute::Image<float>  _denoiseNormal;       // world-space shading normal

    // --- Custom material callable DLL (Debug_Runtime only) ---
#ifdef RT_RUNTIME
    runtime::CallableDLLLoader _callableDLL;
#endif

    // --- Polymorphic surface resolver (built-in + custom materials) ---
    render::SurfaceResolverPoly _surfaceResolver;

    // --- Features (injectable render passes) ---
    std::array<std::vector<std::unique_ptr<IFeature>>, static_cast<size_t>(FeaturePoint::Count)> _features;

    // --- Temp image pool (shared by features) ---
    struct TempImageEntry {
        luisa::compute::PixelStorage format;
        uint width = 0, height = 0;
        std::string id;
        luisa::compute::Image<float> image;
    };
    std::vector<TempImageEntry> _tempImages;

public:
    explicit Pipeline(Renderer& renderer) noexcept;
    Pipeline(Pipeline&&) noexcept = delete;
    Pipeline(const Pipeline&) noexcept = delete;
    Pipeline& operator=(Pipeline&&) noexcept = delete;
    Pipeline& operator=(const Pipeline&) noexcept = delete;
    ~Pipeline() noexcept;

public:
    [[nodiscard]] static PipelinePtr create(Renderer& renderer) noexcept;

    // --- Accessors ---
    [[nodiscard]] auto geometry()      const noexcept { return _geom.get(); }
    [[nodiscard]] auto material()      const noexcept { return _materialPool.get(); }
    [[nodiscard]] auto lightsampler()  const noexcept { return _lightSampler.get(); }
    [[nodiscard]] const auto& surfaceResolver() const noexcept { return _surfaceResolver; }
    [[nodiscard]] auto& surfaceResolver() noexcept { return _surfaceResolver; }
    [[nodiscard]] uint width()         const noexcept { return _width; }
    [[nodiscard]] uint height()        const noexcept { return _height; }
    /// True after buildScene() has been called. Used to warn when callers
    /// mutate the pipeline in ways that are silently ignored post-build
    /// (setProceduralGeometry, addEnvMap, loadCallableDLL, custom resolver reg).
    [[nodiscard]] bool isBuilt()       const noexcept { return _geom && _geom->is_built(); }
    [[nodiscard]] scene::VoxelGrid* voxelGrid() const noexcept {
#if NT_ALLOW_RASTER_FEATURES
        return _voxelGrid.get();
#else
        return nullptr;
#endif
    }

#if NT_ALLOW_RASTER_FEATURES
    [[nodiscard]] feature::RasterContext* rasterContext() const noexcept {
        return _rasterContext.get();
    }
#endif

#if NT_ENABLE_PROCEDURAL
    [[nodiscard]] scene::ProceduralGeometry* proceduralGeom() const noexcept { return _procGeom.get(); }
    /// Transfer ownership of procedural geometry. Call before buildScene().
    void setProceduralGeometry(luisa::unique_ptr<scene::ProceduralGeometry> geom) noexcept;
#endif
    [[nodiscard]] luisa::compute::TimelineEvent& featureGbufEvent() noexcept {
        return _featureGbufEvent;
    }
    [[nodiscard]] auto& computeStream() noexcept { return _computeStream; }
    void setFeatureGbufFence(uint64_t fence) noexcept { _featureGbufFence = fence; }

    [[nodiscard]] luisa::compute::TimelineEvent& renderReadyEvent() noexcept {
        return _renderReadyEvent;
    }
    [[nodiscard]] uint64_t renderReadyFence() const noexcept { return _renderReadyFence; }

    // --- Scene Construction API ---
    [[nodiscard]] uint  addMaterial(const std::string& name, render::MaterialData data);
    /// Borrowed transform: the caller must keep non-static transforms alive for
    /// the shape's lifetime (they are polled every frame).
    [[nodiscard]] scene::ShapeId addShape     (luisa::unique_ptr<scene::MeshShape> shape, scene::Transform* xform);
    /// Owning transform: the pipeline stores and polls it — mutate later via
    /// getShapeTransform(id)->set_*(). nullptr = identity StaticTransform.
    [[nodiscard]] scene::ShapeId addShape     (luisa::unique_ptr<scene::MeshShape> shape,
                                               luisa::unique_ptr<scene::Transform> xform = nullptr);
    [[nodiscard]] scene::ShapeId addLightShape(luisa::unique_ptr<scene::LightShape> light, scene::Transform* xform);
    [[nodiscard]] scene::ShapeId addLightShape(luisa::unique_ptr<scene::LightShape> light,
                                               luisa::unique_ptr<scene::Transform> xform = nullptr);
    void  addEnvMap     (ci::Surface32fRef envmap);
    void  buildScene();

    // --- Prototype Instancing API ---
    struct PrototypeHandle { scene::ShapeId id; };

    /// Register a prototype mesh (NOT added to TLAS). Call before buildScene().
    [[nodiscard]] PrototypeHandle addPrototype(luisa::unique_ptr<scene::MeshShape> prototype);

    /// Add a single TLAS instance referencing a prototype's BLAS.
    [[nodiscard]] scene::ShapeId addPrototypeInstance(PrototypeHandle proto, const luisa::float4x4 &transform);

    /// Batch: add N instances of a prototype with different transforms.
    void addPrototypeInstances(PrototypeHandle proto,
                                luisa::span<const luisa::float4x4> transforms,
                                luisa::vector<scene::ShapeId> &out_ids);

    /// Get prototype MeshShape by handle.
    [[nodiscard]] scene::MeshShape* getPrototype(PrototypeHandle proto) noexcept;

    /// Create an InstancedMesh (prototype + transform buffer) for easy batch updates.
    [[nodiscard]] luisa::unique_ptr<scene::InstancedMesh> createInstancedMesh(
        PrototypeHandle proto, uint instance_count);

    // --- Runtime Shape Manipulation (call before update()) ---
    void setShapeTransform(scene::ShapeId id, const luisa::float4x4 &matrix,
                            scene::Change changeHint = scene::Change::Scale) noexcept;
    void setShapeVisibility(scene::ShapeId id, bool visible) noexcept;
    void setShapeMaterial(scene::ShapeId id, uint32_t material_layers) noexcept;
    bool removeShape(scene::ShapeId id) noexcept;
    [[nodiscard]] scene::MeshShape*     getShape(scene::ShapeId id) noexcept;
    /// Transform update route for a shape: the pipeline-owned transform if the
    /// owning addShape overload was used, else the shape's internal transform.
    /// Mutations apply on the next update().
    [[nodiscard]] scene::Transform*     getShapeTransform(scene::ShapeId id) noexcept;
    [[nodiscard]] scene::DeformableMesh* getDeformable(scene::ShapeId id) noexcept;

    // --- Rendering ---
    void beginFrame(Renderer& renderer) noexcept;
    void update(float time, float dt) noexcept;
    void render(Renderer& renderer, const util::CameraData& data, DebugTag debugTag = DebugTag::None, float dt = 1.0f / 60.0f) noexcept;

    // --- Frame Control ---
    void resize(uint width, uint height);
    void requestAccumReset() noexcept { _accumReset = true; }

    // Binding group accessors for invariant resources
    [[nodiscard]] SceneGeometryResources scene_gpu_resources() const noexcept {
        return { _geom->instance_buffer(), _geom->instance_transform_buffer(), _geom->instance_transform_prev_buffer(), _materialPool->buffer(), _materialPool->simKeyBuffer() };
    }
    [[nodiscard]] EnvLightResources env_gpu_resources() const noexcept {
        return { _lightSampler->envmap_image(), _lightSampler->env_cdf_marginal(),
                 _lightSampler->env_cdf_conditional(), _lightSampler->env_integral(),
                 _lightSampler->env_width(), _lightSampler->env_height(),
                 _lightSampler->env_rotation_matrix() };
    }
    [[nodiscard]] LightSamplingResources light_gpu_resources() const noexcept {
        return { _lightSampler->triangle_buffer(), _lightSampler->vertex_buffer(),
                 _lightSampler->alias_table(), _lightSampler->emissive_triangle_count(),
                 _lightSampler->total_power_inv(), _lightSampler->emissive_count_inv(),
                 _lightSampler->instance_to_light_base() };
    }
    void setDenoiserEnabled(bool enabled) noexcept { _denoiser.setEnabled(enabled); }
    void setGIEnabled(bool enabled) noexcept { _passGI.setEnabled(enabled); }
    void setCheckerboardEnabled(bool enabled) noexcept { _checkerboardEnabled = enabled; }
    void setRasterLightSamples(uint n) noexcept { _rasterLightSamples = std::max(1u, n); }
    [[nodiscard]] uint rasterLightSamples() const noexcept { return _rasterLightSamples; }

    // --- Custom Material Callables ---
#ifdef RT_RUNTIME
    /// Load custom material callable DLL (Debug_Runtime only).
    /// Must be called after create() but before buildScene().
    void loadCallableDLL();
#endif

    // --- Features ---
    /// Register an injectable render feature. Calls onInit() immediately.
    /// Should be called after buildScene().
    void addFeature(std::unique_ptr<IFeature> feature);

    /// Request a temp image from the pool. Same id returns existing image.
    /// Empty id always creates a new unique image.
    [[nodiscard]] luisa::compute::Image<float>& requestTempImage(
        luisa::compute::PixelStorage format, uint width, uint height,
        std::string_view id = "");

    // ui
    void drawUi();

    // --- Config ---
    void loadConfig(const std::filesystem::path& path);
    void saveConfig(const std::filesystem::path& path);
    void applyConfig(const ci::Json& config);
    [[nodiscard]] ci::Json captureConfig() const;

private:
    void _compileShadeShader();
#if NT_DEBUG_VIZ
    void _compileDebugShaders();
#endif
    void _compileUtilityShaders();
    void _createImages(uint width, uint height);
    void _initSeedImage();
    void _loadToneMapLut(const std::filesystem::path& path);
    void _createIdentityLut();

    /// Recompile all shaders (used after callable DLL hot-reload).
    void _recompileAllShaders();

    /// Dispatch all features registered at the given injection point.
    void _dispatchFeatures(luisa::compute::Stream& stream, FeaturePoint point,
                           const FrameContext& frame, luisa::compute::Image<float>& target);
};

}
}
