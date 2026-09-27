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
#if NT_ENABLE_SHARC
#include "newtype/render/PassSharc.h"
#endif
#include "newtype/core/FrameContext.h"
#include "newtype/core/FeaturePoint.h"
#include "newtype/core/IFeature.h"
#include "newtype/upscal/UpscalerBackend.h"
#include "newtype/ngx/DlssRayReconstruction.h"
#include "newtype/util/Profiler.h"
#include "newtype/render/SurfaceResolver.h"
#include "Renderer.h"
#include <luisa/core/fiber.h>

#include <functional>
#include <optional>
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif
#if NT_ALLOW_RASTER_FEATURES
#include "newtype/feature/RasterContext.h"
#endif

#ifdef _DEBUG
#include "newtype/runtime/CallableDLLLoader.h"
#endif

namespace newtype::util { struct LutData; }  // fwd — _uploadToneMapLut param

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
    float                              // 22: endVisMaxDistance (max screen-space px for env vis reuse)
    // Formerly arg 23: hasTransparentShadowCasters (0=any-hit fast path,
    // 1=full transparent loop) is compile-time baked into the shade shader
    // (see _compileShadeShader / _refreshSpecialization).
#if NT_ALLOW_RASTER_FEATURES
    ,
    newtype::scene::VoxelGrid::Resources,  // 24: voxel grid binding group
    luisa::compute::Image<float>,          // 25: shadow cache prev (read)
    luisa::compute::Image<float>           // 26: shadow cache (write)
#endif
    ,
    luisa::compute::Buffer<GIReservoir>,   // 24/27: GI initial-reservoir snapshot (frozen, for MIS)
    float,                                 // 25/28: giMISRoughness (0=disable MIS, RTXDI default 0.3)
    luisa::uint,                           // 26/28: deltaBranchNEEEnabled (1=run NEE at mirror-hit x2)
    luisa::uint,                           // 27/29: dispShadowInterfaces (glass crossings per channel sub-walk)
    float                                  // 28/30: dispShadowSplit (RGB-split saturation, 1=exact)
    ,
    luisa::compute::Image<float>           // SSS radiance (HALF4, per-channel demodulated)
    ,
    // Stagnancy decorrelation (RTXDI 3.1): smoothed stagnancy + firefly flag
    // (R16F, shaded-space/compacted coords) and the swap parameters.
    luisa::compute::Image<float>,          // smoothed stagnancy
    luisa::compute::Image<float>,          // boiling firefly flag
    luisa::uint,                           // decorrelation mode (0=None 1=Uniform 2=Stagnancy)
    float,                                 // decorrelation factor
    float,                                 // stagnancy exponent
    luisa::uint,                           // firefly replacement active
    float                                  // firefly bias-reduction bound
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

// DLSS-RR input format pass: rewrites the denoiser G-buffer signals into the
// float diffuse-albedo / specular-F0 / packed normal+roughness textures NGX
// expects (the sample's DLSSRRInputFormattingPass analog). The diffuse and
// specular guides are recovered from the NRD demod factors (see the shader
// body — the RR guide's appendix wants EnvBRDFApprox(F0), which is exactly
// what the spec demod factor encodes).
using RrInputFormatType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: RR diffuse albedo out (HALF4, .rgb)
    luisa::compute::Image<float>,      // 1: RR specular F0 out (HALF4, .rgb)
    luisa::compute::Image<float>,      // 2: RR normal+roughness out (HALF4, .xyz world normal, .w roughness)
    luisa::compute::Image<float>,      // 3: _denoiseAlbedo in (HALF4, .w = emissive flag / metallic*0.49)
    luisa::compute::Image<float>,      // 4: _denoiseSpecFactor in (HALF4, .xyz = NRD spec demod factor)
    luisa::compute::Image<float>       // 5: _denoiseNormal in (HALF4, .w = matID + roughness packed)
>;

// Denoiser backends at the Pass-9 slot. ReLAX (in-house NRD 4.17 port) is the
// default; DLSS Ray Reconstruction (NGX) replaces it on RTX systems by
// consuming the noisy AfterShade signals 1:1 at render res — the upscale
// stage and everything after the denoiser are unaffected.
enum class DenoiserMode { Relax = 0, DlssRR = 1 };

#if NT_DEBUG_VIZ
// Debug shader: visualize visibility buffer (inst_id, prim_id)
using DebugVisibilityShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output
    luisa::compute::Image<uint>        // 1: visibility buffer input
>;

// Debug shader: motion-vector field (gbuf_bary_motion.zw, NDC).
// R/G = signed motion x/y in px (0.5 = zero, full scale at +/-64 px);
// B = magnitude (saturates at 64 px). Black = no motion.
using DebugMotionShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output
    luisa::compute::Image<float>       // 1: gbuf_bary_motion (motion in .zw, NDC)
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

// Scaling blit (bilinear stretch): emergency upscaler fallback that keeps the
// tonemap input whole when the vendor backend cannot run (e.g. DLSS feature
// creation failed at unsupported dimensions). src dims are runtime args so one
// shader serves every render/display pairing.
using StretchBlitType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: output (dispatch dims)
    luisa::compute::Image<float>,      // 1: src (srcWidth x srcHeight)
    luisa::uint,                       // 2: srcWidth
    luisa::uint                        // 3: srcHeight
>;

// Progressive (offline) accumulation: running average of the composite HDR
// output into a ping-pong pair at render resolution. Frame 0 seeds with the
// raw sample; subsequent frames average (prev * n + cur) / (n + 1).
using ProgressiveAccumType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: dst (this frame's accumulator)
    luisa::compute::Image<float>,      // 1: src (current composite HDR sample)
    luisa::compute::Image<float>,      // 2: prev accumulator
    uint                               // 3: frame index (0 = seed)
>;

#if NT_ENABLE_SHARC
// Zero-fill one image (gather-history validity semantics: A = 0 = invalid)
using ClearImageShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>       // 0: output
>;
#endif

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
#if NT_ENABLE_SHARC
    ,
    luisa::compute::BindlessArray,     // 12/13: tex_bindless (tap surface resolves)
    luisa::compute::Image<float>,      // 13/14: rough-glass info (x=roughness, y=eta, z=is_thin)
    luisa::compute::Buffer<SharcParams>,             // SHARC params (grid params)
    luisa::compute::Buffer<SharcKeyHost>,            // hash entries (read-only)
    luisa::compute::Buffer<render::SharcPackedData>, // resolved radiance (read-only)
    luisa::compute::Buffer<luisa::uint>,             // query stats [4]: GI hit/miss, gather hit/miss
    luisa::compute::uint,              // gatherOn (rough-glass gather enabled)
    luisa::compute::uint,              // transmission tap count K
    luisa::compute::uint,              // reflection tap count K'
    luisa::compute::Image<float>,      // gather history prev (read)
    luisa::compute::Image<float>,      // gather history curr (write)
    luisa::compute::uint,              // frame_count
    float,                             // gather temporal blend alpha
    luisa::compute::uint               // gather history reset (skip reprojection)
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
        None, Depth, Visibility, BaryCentric, Motion, PDF, Jacobian, Normal, Glass, GIReservoir,
        Dispersion   // per-channel first-interface refraction / TIR rings
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
    // InstancedMeshes created via createInstancedMesh (non-owning — they are
    // flushed at the top of update() and must outlive the Pipeline or be
    // released via releaseInstancedMesh()).
    luisa::vector<scene::InstancedMesh *>   _instancedMeshes;
    std::unique_ptr<render::MaterialPool>   _materialPool;
    std::unique_ptr<render::LightSampler>   _lightSampler;

    luisa::compute::Stream                  _computeStream;
    luisa::compute::Stream                  _bufferStream;

    // --- G-Buffer images (16 bytes/pixel) ---
    luisa::compute::Image<float>  _gbufDepth;      // R32F    - linear ray t
    luisa::compute::Image<uint>   _gbufVis;        // RG32U   - (inst_id, prim_id)
    luisa::compute::Image<float>  _gbufBaryMotion; // RGBA16F - RG=barycentrics, BA=motion vectors
    luisa::compute::Image<float>  _glassThroughput; // RGBA16F - RGB=attenuation*(1-F), A=Fresnel reflectivity

    // --- Upscaler inputs (render resolution, written by the G-buffer store) ---
    luisa::compute::Image<float>  _gbufVelocity;      // R16G16F - unjittered motion, render-res pixels
    luisa::compute::Image<float>  _gbufDepthUpscale;  // R32F    - NDC z/w of the virtual hit (non-inverted)

    // --- Upscaler output (display resolution, FLOAT4 HDR). The upscaler stage
    //     writes it; the tonemap block reads it instead of render_target when
    //     _lastFrameUpscaled is set. Only allocated while an upscaler is active. ---
    luisa::compute::Image<float>  _upscaledHdr;

    // --- DLSS Ray Reconstruction inputs (render resolution, RGBA16F; only
    //     allocated while DenoiserMode::DlssRR is active). _rrColor holds the
    //     noisy composited HDR (remod + emission + sky, via the composite
    //     blit); albedo/F0/normal+roughness come from the format pass. ---
    luisa::compute::Image<float>  _rrColor;
    luisa::compute::Image<float>  _rrAlbedo;
    luisa::compute::Image<float>  _rrF0;
    luisa::compute::Image<float>  _rrNormalRoughness;

    // --- Accumulation ---
    luisa::compute::Image<float>  _accumBuffer;    // HALF4 (raw shade output, diffuse)
    luisa::compute::Image<float>  _specularBuffer; // HALF4 (.rgb=specular radiance, .a=hitDist)
    luisa::compute::Image<uint>   _seedImage;      // INT1

    // --- ReLAX Denoiser (self-contained class) ---
    RelaxDenoiser _denoiser;

    // --- DLSS Ray Reconstruction (NGX denoiser replacement at Pass 9) ---
    DenoiserMode  _denoiserMode = DenoiserMode::Relax;
    bool          _denoiserModeUnavailableLogged = false;
    bool          _rrDenoiserRan = false;      // RR dispatched this frame
    std::optional<PassGI::GiDecorrelationSettings> _giDecorrelationPreRr;
    ngx::DlssRrDenoiser _rrDenoiser;

    // --- PassDI (self-contained class) ---
    PassDI _passDI;

    // --- PassGI (self-contained class) ---
    PassGI _passGI;

    // --- PassSSS (self-contained class, ray-traced SSS probe) ---
    PassSSS _passSSS;

    // --- PassSharc (self-contained class, SHARC update/resolve — gated by NT_ENABLE_SHARC) ---
#if NT_ENABLE_SHARC
    PassSharc _passSharc;
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

    // --- Geometry-update event: _computeStream signals after any mid-update
    //     write of render-consumed state (TLAS/BLAS builds, instance/transform
    //     uploads, light-sampler weight/transform rewrites, env rotation
    //     buffer). render() waits on Renderer::stream() before the first
    //     dispatch of the frame — GPU-GPU ordering instead of the former
    //     per-animated-frame _computeStream << synchronize() CPU stall. ---
    luisa::compute::TimelineEvent _geomUpdateEvent;
    uint64_t _geomUpdateFence = 0u;
    bool _geomUpdatePending = false;

    void _signalGeomUpdate() noexcept {
        ++_geomUpdateFence;
        _computeStream << _geomUpdateEvent.signal(_geomUpdateFence);
        _geomUpdatePending = true;
    }

    // --- Previous frame G-Buffer (for denoiser disocclusion detection) ---
    luisa::compute::Image<float>  _gbufDepthPrev;     // FLOAT1
    luisa::compute::Image<uint>   _gbufVisPrev;       // INT2
    luisa::compute::Image<float>  _denoiseNormalPrev; // HALF4 (previous frame world normal)

#if NT_ENABLE_SHARC
    // --- Phase 3 rough-glass gather (docs/sharc_rough_glass_plan.md §7) ---
    luisa::compute::Image<float>  _roughGlassInfo;    // HALF4: roughness, eta, is_thin, -
                                                      // (written by the PSR G-buffer pass)
    luisa::compute::Image<float>  _roughGlassHist[2]; // HALF4 ping-pong gather history
                                                      // (RGB = accumulated gather, A = age, 0 = invalid)
    ClearImageShaderType          _roughGlassClearShader;    // zero-fill (age-0 = invalid semantics)
    uint                          _roughGlassHistIdx = 0u;    // read slot this frame
    uint                          _roughGlassLastFrame = ~0u; // gap detection (toggle/resize reset)
#endif

#if NT_ALLOW_RASTER_FEATURES
    // --- Temporal shadow cache (DDA env shadow attenuation) ---
    luisa::compute::Image<float>  _shadowCache;       // HALF2 (attenuation, depth)
    luisa::compute::Image<float>  _shadowCachePrev;   // HALF2 previous frame
#endif

    // --- Compiled shaders (Shade stays in Pipeline — reads both DI and GI reservoirs) ---
    ShadeShaderType             _shadeShader;

    // Utility shaders
    CompositeBlitType           _compositeBlitShader;  // for no-denoiser fallback: raw shade + envmap sky
    RrInputFormatType           _rrInputFormatShader;  // DLSS-RR guide-buffer formatting
    GlassTintShaderType         _glassTintShader;      // post-denoiser glass Fresnel + absorption
    OITCompositeShaderType      _oitCompositeShader;   // McGuire weighted blended OIT composite
    BlitShaderFltType           _blitFltShader;        // Dx blit
    StretchBlitType             _stretchBlitShader;    // bilinear stretch (upscaler emergency fallback)
    BlitShaderUIntType          _blitUIntShader;
    ProgressiveAccumType        _progressiveAccumShader; // offline running-average accumulation

#if NT_DEBUG_VIZ
    DebugVisibilityShaderType   _debugVisShader;
    DebugMotionShaderType       _debugMotionShader;
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
    // _width/_height are RENDER dimensions (= display * renderScale);
    // _displayWidth/_displayHeight are the window/presentation dimensions.
    uint  _width          = 0;
    uint  _height         = 0;
    uint  _displayWidth   = 0;
    uint  _displayHeight  = 0;
    float _renderScale    = 1.0f;
    bool  _frameSubmitted = false;      // true if previous frame's GPU tail is still executing
    bool  _updateTailWaited = false;    // beginUpdate() already enqueued this frame's tail wait
    FrameResource* _currentFrame = nullptr; // current frame resource (set in beginFrame)

    // --- Upscaler state (docs/upscaling_feasibility_report.md Phase 0/1) ---
    upscal::UpscalerMode _upscalerMode = upscal::UpscalerMode::None;
    float                _upscalerSharpness = 0.0f;  // RCAS, 0 = off
    bool                 _lastFrameUpscaled = false; // upscaler ran this frame
    bool                 _upscalerReset = false;     // one-shot history reset request
    bool                 _upscalerUnavailableLogged = false;
    std::optional<float> _pendingRenderScale;        // UI slider buffer (applied on release)
    luisa::float3        _upscalerPrevCamPos = luisa::make_float3(0.0f);
    bool                 _upscalerHavePrevCamPos = false;
    std::unique_ptr<upscal::IUpscalerBackend> _upscalerBackend;
    Renderer*            _renderer = nullptr;        // for setRenderScale (config path)

    // --- Progressive (offline) accumulation state ---
    // Ping-pong lives in the temp-image pool ("prog_accum_a"/"prog_accum_b",
    // FLOAT4 at render res — recreated on resize by requestTempImage).
    bool _progressiveAccum       = false;
    bool _progressiveResetPending = true;
    uint _progressiveFrame       = 0u;
    uint _progressiveAccumIdx    = 0u;

    // --- Fiber scheduler (MARL-based, for future CPU/GPU overlap) ---
    std::unique_ptr<luisa::fiber::scheduler> _fiberScheduler;

    // --- PreUpdate dummy target ---
    luisa::compute::Image<float>  _dummyTarget;   // 1x1 FLOAT4 for PreUpdate feature dispatch

    // --- Shared tunable parameters ---
    float _boilingFilterStrength  = 0.2f;  // 0..1, lower = more aggressive outlier rejection
    uint  _rasterLightSamples     = 4u;    // local light samples for point/trail shading

    // --- Feature toggles ---
    bool _checkerboardEnabled = true;
    // Value baked into the compiled DI/GI/shade kernels and the reservoir
    // image layout (half-width while on). Diverges from _checkerboardEnabled
    // when a runtime toggle (DLSS-RR engage/leave, config load) changes the
    // desired state; render() applies the flip at its thread-safe point via
    // _applyCheckerboardReconfigure.
    bool _checkerboardBaked = true;
    // Checkerboard state to restore when leaving DLSS-RR (mirrors
    // _giDecorrelationPreRr).
    std::optional<bool> _checkerboardPreRr;
    // Primary-ray sub-pixel jitter (Halton TAA). When false, Pipeline::render
    // zeroes CameraData::jitter/prev_jitter before the FrameContext is built,
    // making the primary ray / G-buffer fully deterministic (see
    // docs/primary_jitter_plan.md). Runtime float field — no shader recompiles.
    bool _primaryJitterEnabled = true;
    // perf R2 item 14: let the Halton jitter through while the upscaler runs
    // at renderScale < 1 — FSR expects jittered input and resolves the
    // sub-pixel pattern at display res (the full-res TAA toggle above stays
    // independent). The unjittered-bary hybrid speckle risk
    // (docs/primary_jitter_plan.md) is what the A/B gates; flip via the
    // "fsrJitterEnabled" config key / Upscaler UI.
    bool _fsrJitterEnabled = true;
    // perf R2 item 14: skip the FXAA AfterToneMap pass when the frame went
    // through the upscaler at renderScale < 1 (FSR already reconstructs +
    // RCAS-sharpens; FXAA on top is a display-res double-AA pass).
    bool _fxaaSkipWhenUpscaled = true;
    bool _requireExplicitBlit = false;
    bool      _solidBackgroundEnabled = false;
    luisa::float3 _solidBackgroundColor = luisa::make_float3(0.02f);

    // --- Tone mapping settings ---
    ToneMapMode _toneMapMode     = ToneMapMode::ACES;
    float       _toneMapExposure = 1.0f;
    float       _toneMapGamma    = 2.2f;

    // True for frames where the pipeline tone-mapped HDR -> LDR display target
    // (RGBA8 output mode). AfterToneMap features consult this to keep their
    // post-tonemap contract (e.g. FXAA skips in HDR float display mode).
    bool _lastFrameTonemapped = false;

    // True for frames where tonemap wrote the persistent "fxaa_temp" BYTE4
    // temp instead of the display target (perf R2 item 13): the sole enabled
    // AfterToneMap feature reads the temp and writes the display itself, so
    // the display target is stale until that feature runs.
    bool _tonemapRoutesToTemp = false;

    // --- Tone mapping LUT ---
    luisa::compute::Volume<float>  _toneMapLut;           // 3D texture (FLOAT4, N^3)
    bool                           _toneMapLutEnabled = false;
    std::string                    _toneMapLutPath;        // path to loaded .cube file

    // Auxiliary images for edge-stopping (HALF4)
    luisa::compute::Image<float>  _denoiseAlbedo;       // material albedo (no lighting)
    luisa::compute::Image<float>  _denoiseSpecFactor;   // NRD specular demod factor
    luisa::compute::Image<float>  _denoiseNormal;       // world-space shading normal

    // --- Custom material callable DLL (Debug builds; inert unless the DLL
    //     is present at the expected runtime_shaders path) ---
#ifdef _DEBUG
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

    // Shade debug viz ('v' hotkey; mode 10 = GI weight_sum/rad/indirect).
    void setShadeDebugVizMode(int m) noexcept { _shadeDebugVizMode = m; }
    [[nodiscard]] int shadeDebugVizMode() const noexcept { return _shadeDebugVizMode; }
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
    /// `material_layers` (B1): optional per-instance 4 × 8-bit layer pack —
    /// instances of one prototype can render different materials. Default
    /// inherits the prototype's layers.
    [[nodiscard]] scene::ShapeId addPrototypeInstance(PrototypeHandle proto, const luisa::float4x4 &transform,
                                                      uint32_t material_layers = scene::kInheritMaterialLayers);
    /// Borrowed transform: polled every frame when the chain is not fully
    /// static — mutate it directly (set_local_* / set_*) and the instance
    /// follows. Caller keeps the transform alive.
    [[nodiscard]] scene::ShapeId addPrototypeInstance(PrototypeHandle proto, scene::Transform* xform,
                                                      uint32_t material_layers = scene::kInheritMaterialLayers);
    /// Owning transform: the pipeline stores and polls it — mutate later via
    /// getShapeTransform(id)->set_*().
    [[nodiscard]] scene::ShapeId addPrototypeInstance(PrototypeHandle proto,
                                                      luisa::unique_ptr<scene::Transform> xform,
                                                      uint32_t material_layers = scene::kInheritMaterialLayers);

    /// Batch: add N instances of a prototype with different transforms.
    void addPrototypeInstances(PrototypeHandle proto,
                                luisa::span<const luisa::float4x4> transforms,
                                luisa::vector<scene::ShapeId> &out_ids);
    /// Batch with per-instance material layers (parallel to `transforms`;
    /// entries may be scene::kInheritMaterialLayers to inherit).
    void addPrototypeInstances(PrototypeHandle proto,
                                luisa::span<const luisa::float4x4> transforms,
                                luisa::span<const uint32_t> material_layers,
                                luisa::vector<scene::ShapeId> &out_ids);
    /// Batch with borrowed transforms (same registration rule as the single
    /// addPrototypeInstance(Transform*) overload).
    void addPrototypeInstances(PrototypeHandle proto,
                                luisa::span<scene::Transform *const> transforms,
                                luisa::vector<scene::ShapeId> &out_ids);

    /// Get prototype MeshShape by handle.
    [[nodiscard]] scene::MeshShape* getPrototype(PrototypeHandle proto) noexcept;

    /// Create an InstancedMesh (prototype + transform buffer) for easy batch
    /// updates. The Pipeline flushes its transforms automatically at the top
    /// of update(); destroy it only after releaseInstancedMesh() or Pipeline
    /// teardown.
    [[nodiscard]] luisa::unique_ptr<scene::InstancedMesh> createInstancedMesh(
        PrototypeHandle proto, uint instance_count);

    /// Stop flushing this InstancedMesh in update() (call before destroying
    /// one created via createInstancedMesh if it does not outlive the Pipeline).
    void releaseInstancedMesh(scene::InstancedMesh *mesh) noexcept;

    // --- Runtime Shape Manipulation (call before update()) ---
    void setShapeTransform(scene::ShapeId id, const luisa::float4x4 &matrix,
                            scene::Change changeHint = scene::Change::Scale) noexcept;
    void setShapeVisibility(scene::ShapeId id, bool visible) noexcept;
    /// Camera-path visibility only: primary rays, mirror reflections, and
    /// glass replays pass through the shape while it keeps illuminating
    /// (light sampling power, shadow, and GI rays are unaffected).
    void setShapeCameraVisibility(scene::ShapeId id, bool camera_visible) noexcept;
    void setShapeMaterial(scene::ShapeId id, uint32_t material_layers) noexcept;

    /// Per-instance custom data (track B2): write one float4 slot (0..3) of
    /// an instance's 64 B user-params row, readable by custom material
    /// callables via instance_params(tex, s.instance_index, i). The first
    /// write materializes the GPU-side buffer (rows cover every instance,
    /// zeros elsewhere); uploads flush on the next update().
    void setInstanceUserData(scene::ShapeId id, uint slot, luisa::float4 value) noexcept;
    /// Read back one authored slot (zeros when never written).
    [[nodiscard]] luisa::float4 instanceUserData(
        scene::ShapeId id, uint slot) const noexcept;

    /// Opt-in (A1): release CPU mesh mirrors for static, non-emissive,
    /// non-physics meshes after buildScene() — see
    /// Geometry::unload_static_cpu_data for the exact skip rules.
    void unloadStaticMeshCPUData() noexcept;
    bool removeShape(scene::ShapeId id) noexcept;
    [[nodiscard]] scene::MeshShape*     getShape(scene::ShapeId id) noexcept;
    /// Transform update route for a shape: the pipeline-owned transform if an
    /// owning addShape/addPrototypeInstance overload was used, the stored
    /// borrowed transform for prototype instances, else the shape's internal
    /// transform (nullptr for matrix-only prototype instances). Mutations
    /// apply on the next update().
    [[nodiscard]] scene::Transform*     getShapeTransform(scene::ShapeId id) noexcept;
    [[nodiscard]] scene::DeformableMesh* getDeformable(scene::ShapeId id) noexcept;

    // --- GPU-owned transform rows (device-side TLAS transform updates) ---
    // Thin delegations to Geometry; contract documented on the Geometry
    // methods (docs/TLAS-instance-transform-updates.md Part 2).
    /// Dense TLAS row of a shape (~0u for invalid/prototype ids).
    [[nodiscard]] uint geometryTlasRow(scene::ShapeId id) const noexcept;
    /// Register rows for a device writer (one contiguous run required).
    bool registerGpuTransformRows(luisa::span<const scene::ShapeId> ids) noexcept;
    /// Device writer rewrote the registered rows this frame.
    void notifyGpuTransformsDirty() noexcept;
    [[nodiscard]] uint topologyGeneration() const noexcept;

    // --- Rendering ---
    void beginFrame(Renderer& renderer) noexcept;
    /// Order the scene phase against the previous frame's render tail:
    /// enqueues the _frameTailEvent wait on the compute stream so any
    /// scene-update dispatch (TetSolve transform rows, deformable vertex
    /// uploads + BLAS rebuilds) executes after the render stream's in-flight
    /// reads of those same buffers. Call BEFORE the scene's update() — the
    /// wait inside update() alone is too late for dispatches enqueued ahead
    /// of it in stream order. No-op when no render is in flight.
    void beginUpdate() noexcept;
    void update(float time, float dt) noexcept;
    void render(Renderer& renderer, const util::CameraData& data, DebugTag debugTag = DebugTag::None, float dt = 1.0f / 60.0f) noexcept;

    // --- Video publish window ---
    /// Hook fired at the top of beginFrame, right after the previous GPU
    /// frame's synchronize — the only point in the tick where the GPU is
    /// drained, so cross-API shared-texture writes are cheap and race-free.
    /// The engine wires the media player's publish_due_frame() here. Called
    /// every frame; must be non-blocking.
    void setVideoPublishHook(std::function<void()> hook) noexcept {
        _videoPublishHook = std::move(hook);
    }

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
    void setPrimaryJitterEnabled(bool enabled) noexcept { _primaryJitterEnabled = enabled; }
    [[nodiscard]] bool primaryJitterEnabled() const noexcept { return _primaryJitterEnabled; }
    // perf R2 item 14 (see the member comments): FSR-input jitter + FXAA
    // upscaler guard. The jitter toggle resets temporal/upscaler histories
    // (the pattern change invalidates both).
    void setFsrJitterEnabled(bool enabled) noexcept;
    [[nodiscard]] bool fsrJitterEnabled() const noexcept { return _fsrJitterEnabled; }
    void setFxaaSkipWhenUpscaled(bool v) noexcept { _fxaaSkipWhenUpscaled = v; }
    [[nodiscard]] bool fxaaSkipWhenUpscaled() const noexcept { return _fxaaSkipWhenUpscaled; }
    [[nodiscard]] bool lastFrameTonemapped() const noexcept { return _lastFrameTonemapped; }
    [[nodiscard]] bool lastFrameUpscaled() const noexcept { return _lastFrameUpscaled; }
    // perf R2 item 13: afterToneMap features read this to know the tonemap
    // output already lives in "fxaa_temp" (skip the input copy).
    [[nodiscard]] bool tonemapRoutedToTemp() const noexcept { return _tonemapRoutesToTemp; }
    void setRasterLightSamples(uint n) noexcept { _rasterLightSamples = std::max(1u, n); }
    [[nodiscard]] uint rasterLightSamples() const noexcept { return _rasterLightSamples; }

    // --- Upscaler (render-res -> display-res HDR, pre-tonemap) ---
    // Selects the backend and (re)creates the upscaler context + images.
    // setRenderScale enforces: scale < 1 requires an active upscaler, and an
    // upscaler requires RGBA8 display mode. Both are destructive (image
    // recreation) — same envelope as resize().
    void setUpscalerMode(upscal::UpscalerMode mode);
    [[nodiscard]] upscal::UpscalerMode upscalerMode() const noexcept { return _upscalerMode; }
    void setRenderScale(float scale);

    // Denoiser backend selection (ReLAX default; DLSS-RR replaces it at the
    // Pass-9 slot when NGX reports it available). Falls back to ReLAX with a
    // one-time log otherwise. Also engages/restores the PassGI RR-compat
    // decorrelation preset on mode switches.
    void setDenoiserMode(DenoiserMode mode);
    [[nodiscard]] DenoiserMode denoiserMode() const noexcept { return _denoiserMode; }
    [[nodiscard]] bool rrDenoiserActive() const noexcept;
    [[nodiscard]] float renderScale() const noexcept { return _renderScale; }

    // --- Non-perspective projection support (docs/non_perspective_camera_report.md) ---
    // Decouple the render resolution from the window (equirect 2:1, room-rig
    // atlases). Renderer::presentWidth/Height follow the override, so the
    // recorder captures at render resolution and the present path letterboxes.
    // Implies renderScale = 1 and disables the upscaler path. Destructive
    // (image recreation) — same envelope as resize().
    void setRenderResolutionOverride(std::optional<luisa::uint2> size);

    // --- Progressive (offline) accumulation ---
    // Running average of raw (denoiser-off) frames at render resolution —
    // the stills/animation export path for non-perspective projections
    // (ReLAX reconstruction assumes a pinhole frustum there) and for
    // noise-free perspective stills. Camera motion should call
    // requestProgressiveReset().
    void setProgressiveAccumulation(bool enabled) noexcept;
    [[nodiscard]] bool progressiveAccumulation() const noexcept { return _progressiveAccum; }
    [[nodiscard]] uint progressiveFrameIndex() const noexcept { return _progressiveFrame; }
    void requestProgressiveReset() noexcept { _progressiveResetPending = true; }

    [[nodiscard]] uint  displayWidth()  const noexcept { return _displayWidth; }
    [[nodiscard]] uint  displayHeight() const noexcept { return _displayHeight; }
    void setUpscalerSharpness(float s) noexcept { _upscalerSharpness = std::clamp(s, 0.0f, 1.0f); }
    void requestUpscalerReset() noexcept { _upscalerReset = true; }
    // True when the upscaler stage should run this frame (mode set, backend
    // context alive, RGBA8 output). Does NOT imply scale < 1 — NativeAA runs
    // at 1:1 for pure temporal AA.
    [[nodiscard]] bool upscalerActive() const noexcept;

    // --- Custom Material Callables ---
#ifdef _DEBUG
    /// Load custom material callable DLL (Debug builds; warns and continues
    /// when the DLL is absent). Must be called after create() but before
    /// buildScene().
    void loadCallableDLL();
#endif

    // --- Features ---
    /// Register an injectable render feature. Calls onInit() and onResize()
    /// immediately (first image allocation must not wait for a resize event).
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

    // --- Scene content (serialized by the app into scene.json) ---
    /// Non-builtin materials as a JSON array (MaterialPool layout). The app
    /// owns the "materials" key, same as it owns "camera"/feature keys.
    [[nodiscard]] ci::Json captureMaterials() const;
    void applyMaterials(const ci::Json& json);

private:
    // Fired at the top of beginFrame after the frame synchronize (see
    // setVideoPublishHook). Default-constructed = no-op.
    std::function<void()> _videoPublishHook;

    void _compileShadeShader();
#if NT_DEBUG_VIZ
    void _compileDebugShaders();
#endif
    void _compileUtilityShaders();
    void _createImages(uint width, uint height);
    void _initSeedImage();
    void _loadToneMapLut(const std::filesystem::path& path);
    /// Upload an already-parsed LUT (shared by file loads and the embedded default).
    void _uploadToneMapLut(const util::LutData& lut, const std::string& label);
    void _createIdentityLut();

    /// Recompile all shaders (used after callable DLL hot-reload).
    void _recompileAllShaders();

    // --- Compile-time flag specialization -----------------------------------
    // oneBounce/giScale/sharcQueryOn (PassGI), diBiasCorrectionEnabled
    // (PassDI) and hasTransparentShadowCasters (PassDI/PassGI/shade) are
    // baked into their kernels at compile time. This check compares each
    // pass's baked snapshot against the current desired values and recompiles
    // the affected passes through the same path as a callable DLL hot-reload.
    // Must run on the render thread (or before the render loop starts) —
    // the same guarantee _recompileAllShaders relies on.
    void _refreshSpecialization();
    uint _shadeBakedTransparentShadowCasters = 0u;

    /// Checkerboard flip application (render-thread safe point): recompiles
    /// the DI/GI/shade kernels with the new flag and recreates the reservoir
    /// images (half-width <-> full-width layout). Called from render() when
    /// _checkerboardBaked diverges from _checkerboardEnabled.
    void _applyCheckerboardReconfigure();

    /// Dispatch all features registered at the given injection point.
    void _dispatchFeatures(luisa::compute::Stream& stream, FeaturePoint point,
                           const FrameContext& frame, luisa::compute::Image<float>& target);

    /// One-shot upscaler history reset decision: requested reset, accum
    /// reset, or a camera-position jump (teleport heuristic).
    bool _consumeUpscalerReset(const FrameContext& ctx);
};

}
}
