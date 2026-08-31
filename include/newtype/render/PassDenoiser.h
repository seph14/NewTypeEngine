#pragma once
#include <luisa/luisa-compute.h>
#include "cinder/Json.h"
#include "newtype/render/RelaxDenoiser.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/SurfaceResolver.h"
#include "newtype/render/LightSampler.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/Camera.h"
#include "newtype/util/CommandBuffer.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Config.h"
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif

namespace newtype {
namespace core {

using namespace luisa::compute;

//==========================================================================
// Denoiser Shader Type Aliases
//==========================================================================

// Denoise PreFilter: G-Buffer -> clean albedo + world normal images
using DenoisePreFilterType = Shader<2,
    Image<float>,      // 0: albedo output (HALF4, rgb = diffuse demod factor)
    Image<float>,      // 0b: spec demod factor output (HALF4)
    Image<float>,      // 1: normal output (HALF4)
    Image<float>,      // 2: gbuf_depth
    Image<uint>,       // 3: gbuf_vis
    Image<float>,      // 4: gbuf_bary_motion (reads .xy() for barycentrics)
    util::CameraData,  // 5: camera
    Buffer<luisa::uint4>,           // 6: instance_buffer
    Buffer<luisa::float4x4>,        // 7: instance_transform_buffer
    Buffer<render::MaterialData>,   // 8: material_buffer
    BindlessArray,                   // 9: vertex_bindless
    BindlessArray                    // 10: tex_bindless (material textures)
#if NT_ENABLE_PROCEDURAL
    , BindlessArray                    // 11: proc_bindless (instances, positions, indices, AABBs, normals)
#endif
>;

// ReLAX ClassifyTiles: 16x16 tile sky detection (groupshared reduction)
using RelaxClassifyTilesType = Shader<2,
    Image<float>,      // 0: output tiles (1.0=sky tile)
    Image<float>,      // 1: gIn_ViewZ (R32F)
    Buffer<render::RelaxConstants> // 2: constants
>;

// ReLAX Prepass: Poisson 8-tap spatial blur for noisy input (diffuse + specular)
using RelaxPrepassType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: io_Diff (RGBA16F, in/out)
    Image<float>,      // 2: io_Spec (RGBA16F, in/out)
    Image<float>,      // 3: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 4: gIn_ViewZ (R32F)
    Image<float>       // 5: gIn_Tiles (R8)
>;

// ReLAX Hit Distance Reconstruction: 3x3 bilateral smooth for stable virtual motion.
// Early-outs to raw passthrough when checkerboard is active (NRD: enableHitDistanceReconstruction
// requires checkerboardMode == OFF).
using RelaxHitDistReconstructType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants (gDiffCheckerboard gates the early-out)
    Image<float>,      // 1: gIn_SpecInput (RGBA16F, read .w = raw hit distance)
    Image<float>,      // 2: gIn_ViewZ (R32F)
    Image<float>,      // 3: gIn_Normal_Roughness (RGBA16F)
    Image<float>       // 4: out_SpecHitDistSmoothed (R32F)
>;

// ReLAX TemporalAccumulation: bilinear reprojection + dual-rate EMA (diffuse + specular)
using RelaxTemporalAccumulationType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: gIn_ViewZ (R32F, current frame)
    Image<float>,      // 2: gIn_Normal_Roughness (RGBA16F, current frame)
    Image<float>,      // 3: gIn_Diff_Radiance_HitDist (RGBA16F, noisy input)
    Image<float>,      // 4: gbuf_bary_motion (reads .zw() for motion vectors)
    Image<uint>,       // 5: gbuf_vis (is_point flag for motion vector masking)
    Image<float>,      // 6: out_Diff (RGBA16F)
    Image<float>,      // 7: out_DiffFast (RGBA16F)
    Image<float>,      // 8: out_HistoryLength (R16F)
    Image<float>,      // 9: out_Normal_Roughness (RGBA16F, for next frame)
    Image<float>,      // 10: out_ViewZ (R32F, for next frame)
    Image<float>,      // 11: gIn_Tiles (R8)
    Image<float>,      // 12: gPrev_ViewZ (R32F, disocclusion check)
    Image<float>,      // 13: gPrev_Normal_Roughness (RGBA16F, backface check)
    Image<float>,      // 14: gPrev_DiffHistory (RGBA16F, weighted bilinear read)
    Image<float>,      // 15: gPrev_DiffFastHistory (RGBA16F, weighted bilinear read)
    Image<float>,      // 16: gPrev_HistoryLength (R16F, weighted bilinear read)
    // Specular channels (16-24)
    Image<float>,      // 17: gIn_Spec_Radiance_HitDist (RGBA16F, noisy specular)
    Image<float>,      // 18: out_Spec (RGBA16F)
    Image<float>,      // 19: out_SpecFast (RGBA16F)
    Image<float>,      // 20: gPrev_SpecHistory (RGBA16F)
    Image<float>,      // 21: gPrev_SpecFastHistory (RGBA16F)
    Image<float>,      // 22: gPrev_ViewZ2 (alias of 11, for spec disocclusion)
    Image<float>,      // 23: gPrev_SpecHitDist (R32F, virtual motion)
    Image<float>,      // 24: out_SpecHitDist (R32F, virtual motion)
    Image<float>       // 25: gIn_SpecHitDistSmoothed (R32F, bilateral-smoothed)
>;

// ReLAX HistoryFix: 5x5 sparse cross-bilateral for disoccluded pixels
using RelaxHistoryFixType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: io_Diff (RGBA16F, in/out)
    Image<float>,      // 2: io_DiffFast (RGBA16F, in/out)
    Image<float>,      // 3: io_HistoryLength (R16F, in/out)
    Image<float>,      // 4: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 5: gIn_ViewZ (R32F)
    Image<float>,      // 6: gIn_Tiles (R8)
    // Specular channels (7-8)
    Image<float>,      // 7: io_Spec (RGBA16F, in/out)
    Image<float>       // 8: io_SpecFast (RGBA16F, in/out)
>;

// ReLAX HistoryClamping: YCoCg color-box clamp + anti-lag (diffuse + specular)
using RelaxHistoryClampingType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: io_Diff (RGBA16F, in/out = main history)
    Image<float>,      // 2: io_DiffFast (RGBA16F, in/out = fast history)
    Image<float>,      // 3: io_HistoryLength (R16F, in/out)
    Image<float>,      // 4: gIn_DiffNoisy (RGBA16F, original noisy input)
    Image<float>,      // 5: gIn_ViewZ (R32F, sky rejection)
    Image<float>,      // 6: gIn_Tiles (R8)
    // Specular channels (7-9)
    Image<float>,      // 7: io_Spec (RGBA16F, in/out)
    Image<float>,      // 8: io_SpecFast (RGBA16F, in/out)
    Image<float>       // 9: gIn_SpecNoisy (RGBA16F, noisy specular input)
>;

// ReLAX Atrous: 3x3 cross-bilateral with configurable step size (diffuse + specular)
// Same signature used by both regular and smem variants
using RelaxAtrousSmemType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: output Diff+variance (RGBA16F)
    Image<float>,      // 2: input Diff+variance (RGBA16F)
    Image<float>,      // 3: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 4: gIn_ViewZ (R32F)
    Image<float>,      // 5: gIn_HistoryLength (R16F)
    uint,              // 6: stepSize (1, 2, 4, 8, 16)
    uint,              // 7: isLastPass (1 if last atrous iteration, 0 otherwise)
    Image<float>,      // 8: gIn_Tiles (R8)
    // Specular channels (9-10)
    Image<float>,      // 9: output Spec+variance (RGBA16F)
    Image<float>       // 10: input Spec+variance (RGBA16F)
>;

using RelaxAtrousType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: output Diff+variance (RGBA16F)
    Image<float>,      // 2: input Diff+variance (RGBA16F)
    Image<float>,      // 3: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 4: gIn_ViewZ (R32F)
    Image<float>,      // 5: gIn_HistoryLength (R16F)
    uint,              // 6: stepSize (1, 2, 4, 8, 16)
    uint,              // 7: isLastPass (1 if last atrous iteration, 0 otherwise)
    Image<float>,      // 8: gIn_Tiles (R8)
    // Specular channels (9-10)
    Image<float>,      // 9: output Spec+variance (RGBA16F)
    Image<float>       // 10: input Spec+variance (RGBA16F)
>;

// ReLAX Anti-firefly: RCRS 3x3 min/max luminance clamping (diffuse + specular)
// Material-ID gated to prevent firefly bleed across material boundaries (NRD CompareMaterials).
using RelaxAntiFireflyType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants (gDiffMinMaterial, gSpecMinMaterial)
    Image<float>,      // 1: output Diff+variance (RGBA16F)
    Image<float>,      // 2: input Diff+variance (RGBA16F)
    Image<float>,      // 3: gIn_Tiles (R8)
    // Specular channels (4-6)
    Image<float>,      // 4: output Spec+variance (RGBA16F)
    Image<float>,      // 5: input Spec+variance (RGBA16F)
    Image<float>,      // 6: gIn_Tiles2 (alias of 3)
    Image<float>       // 7: gIn_Normal_Roughness (RGBA16F, .w = packed matID+rough)
>;

// ReLAX Motion Copy: unpacks packed motion from rasterDepth.G into gbufBaryMotion.zw
using RelaxMotionCopyType = Shader<2,
    Image<float>,      // 0: rasterDepth (FLOAT2, read packed motion from .y)
    Image<uint>,       // 1: gbufVis (read, check is_point flag)
    Image<float>       // 2: gbufBaryMotion (HALF4, read-modify-write .zw with motion)
>;

// Utility shaders used by denoiser internally
using DenoiserBlitType = Shader<2, Image<float>, Image<float>>;
using DenoiserCompositeBlitType = Shader<2,
    Image<float>,      // 0: output (display)
    Image<float>,      // 1: denoised diffuse (HALF4)
    Image<float>,      // 2: gbuf_depth (R32F)
    Image<float>,      // 3: envmap
    util::CameraData,  // 4: camera
    uint, uint,        // 5-6: env_width, env_height
    Buffer<float3x3>,  // 7: env_rotation
    float,             // 8: env_exposure
    Image<float>,      // 9: albedo (HALF4, rgb = diffuse demod factor)
    Image<float>,      // 9b: spec demod factor (HALF4)
    Image<float>,      // 10: denoised specular (HALF4)
    uint,              // 11: solid_bg_enabled (0 or 1)
    luisa::float3      // 12: solid_bg_color
>;
//==========================================================================
// RelaxDenoiser Class
//==========================================================================

class RelaxDenoiser {
public:
    explicit RelaxDenoiser() noexcept = default;
    RelaxDenoiser(RelaxDenoiser&&) noexcept = default;
    RelaxDenoiser(const RelaxDenoiser&) = delete;
    RelaxDenoiser& operator=(const RelaxDenoiser&) = delete;
    RelaxDenoiser& operator=(RelaxDenoiser&&) = delete;
    ~RelaxDenoiser() noexcept = default;

    // --- Lifetime ---
    void compile(Device& device, const render::SurfaceResolverPoly& resolver);
    // Recompile only the resolver-dependent sub-shader (prefilter). Resolver-free
    // sub-shaders (prepass, temporal, history, atrous, etc.) are unaffected by
    // custom-callable DLL changes and skip recompilation.
    void recompileCallables(Device& device, const render::SurfaceResolverPoly& resolver);
    void createImages(Device& device, uint width, uint height);
    void zeroInitImages(uint width, uint height);
    void release();

    // --- Prefilter (extracted for early dispatch after G-Buffer) ---
    // Writes _denoiseAlbedo + _denoiseNormal from G-Buffer data.
    // Must be called after feature G-buffer modifications are complete.
    void renderPrefilter(CommandList& cmdlist, const FrameContext& ctx);

    // --- Main render ---
    // Returns true if denoiser rendered to renderTarget, false if bypassed (no denoiser)
    // rasterDepth is optional — when provided, packed motion is unpacked into gbufBaryMotion.zw
    bool render(Stream& stream, const FrameContext& ctx,
                const Image<float>& renderTarget, bool debugTagNone,
                const Image<float>* rasterDepth = nullptr);

    // --- Accessors ---
    bool enabled() const noexcept { return _enabled; }
    void setEnabled(bool v) noexcept { _enabled = v; }

    uint relaxFrameIdx() const noexcept { return _relaxFrameIdx; }

    // Expose settings for UI
    render::RelaxSettings& settings() noexcept { return _relaxSettings; }

#if NT_DEBUG_VIZ
    // Debug state
    int  debugStage() const noexcept { return _relaxDebugStage; }
    void setDebugStage(int v) noexcept { _relaxDebugStage = v; }
    bool disableAntilag() const noexcept { return _relaxDisableAntilag; }
    void setDisableAntilag(bool v) noexcept { _relaxDisableAntilag = v; }
    bool disableClamp() const noexcept { return _relaxDisableClamp; }
    void setDisableClamp(bool v) noexcept { _relaxDisableClamp = v; }
    int  histClampDebugViz() const noexcept { return _relaxHistClampDebugViz; }
    void setHistClampDebugViz(int v) noexcept { _relaxHistClampDebugViz = v; }
#endif

    void advanceFrame() noexcept { _relaxFrameIdx++; }
    void resetFrameIdx() noexcept { _relaxFrameIdx = 0u; }

    void drawUi();

    // --- Config serialization ---
    void toJson(ci::Json& j) const;
    void fromJson(const ci::Json& j);

private:
    // --- Compiled Shaders ---
    DenoisePreFilterType            _denoisePreFilterShader;
    RelaxClassifyTilesType          _relaxClassifyTiles;
    RelaxPrepassType                _relaxPrepass;
    RelaxHitDistReconstructType     _relaxHitDistReconstruct;
    RelaxTemporalAccumulationType   _relaxTemporalAccumulation;
    RelaxHistoryFixType             _relaxHistoryFix;
    RelaxHistoryClampingType        _relaxHistoryClamping;
    RelaxAtrousType                 _relaxAtrous;
    RelaxAtrousSmemType             _relaxAtrousSmem;
    RelaxAntiFireflyType            _relaxAntiFirefly;
    RelaxMotionCopyType             _relaxMotionCopy;

    // --- ReLAX Persistent Resources (ping-pong) ---
    Buffer<render::RelaxConstants>  _relaxConstantsBuf;
    BindlessArray                   _relaxHeap;

    Image<float>  _relaxDiffHistory[2];       // RGBA16F
    Image<float>  _relaxDiffFastHistory[2];   // RGBA16F
    Image<float>  _relaxSpecHistory[2];       // RGBA16F
    Image<float>  _relaxSpecFastHistory[2];   // RGBA16F
    Image<float>  _relaxNormalRoughnessPrev[2]; // RGBA16F
    Image<float>  _relaxViewZPrev[2];         // R32F
    Image<float>  _relaxHistoryLengthPrev[2]; // RG16F (.x=historyLength, .y=specConf)
    Image<float>  _relaxSpecHitDistPrev[2];   // R32F
    Image<float>  _relaxSpecHitDistSmoothed;  // R32F

    // Transient textures (reused across passes within a frame)
    Image<float>  _relaxTiles;                // R8
    Image<float>  _relaxDiff[2];              // RGBA16F (ping-pong between passes)
    Image<float>  _relaxSpec[2];              // RGBA16F
    Image<float>  _relaxDiffFast;             // RGBA16F
    Image<float>  _relaxSpecFast;             // RGBA16F
    Image<float>  _relaxHistoryLength;        // R16F

    uint _relaxFrameIdx = 0u;

    // --- Settings ---
    render::RelaxSettings _relaxSettings;
    bool _enabled = true;

    // --- Frame state (set per-frame by Pipeline before render) ---
    uint _frameCount = 0;
    uint _width = 0;
    uint _height = 0;
    bool _accumReset = false;

    // --- Debug state ---
#if NT_DEBUG_VIZ
    int  _relaxDebugStage = 0;
    bool _relaxDisableAntilag = false;
    bool _relaxDisableClamp = false;
    int  _relaxHistClampDebugViz = 0;
#endif

    // Internal utility shaders (compiled by RelaxDenoiser)
    DenoiserBlitType            _blitShader;
    DenoiserBlitType            _clearImageShader;
    DenoiserCompositeBlitType   _compositeBlitShader;
#if NT_DEBUG_VIZ
    DenoiserCompositeBlitType   _debugTaBlitShader;
#endif

    void _populateRelaxConstantsStruct(render::RelaxConstants& consts,
                                        const util::CameraData& cam,
                                        uint width, uint height,
                                        uint frameCount, uint cbField,
                                        bool accumReset,
                                        float dt);
    void _populateRelaxConstants(Stream& stream,
                                  const util::CameraData& cam,
                                  uint width, uint height,
                                  uint frameCount, uint cbField,
                                  bool accumReset,
                                  float dt);

    void compilePrefilterAndClassifyTiles(Device& device, const render::SurfaceResolverPoly& resolver);
    void compilePrepass       (Device& device);
    void compileHitAndTemporal(Device& device);
    void compileHistory       (Device& device);
    void compileAtrous        (Device& device);
    void compileAtrousSmem    (Device& device);
    void compileUtility       (Device& device);
};

} // namespace core
} // namespace newtype
