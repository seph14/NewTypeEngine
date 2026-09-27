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
    Image<float>,      // 5: gIn_Tiles (R8, sky-tile early out — perf R2 item 9)
    util::CameraData,  // 6: camera
    Buffer<luisa::uint4>,           // 7: instance_buffer
    Buffer<luisa::float4x4>,        // 8: instance_transform_buffer
    Buffer<render::MaterialData>,   // 9: material_buffer
    BindlessArray,                   // 10: vertex_bindless
    BindlessArray                    // 11: tex_bindless (material textures)
#if NT_ENABLE_PROCEDURAL
    , BindlessArray                    // 12: proc_bindless (instances, positions, indices, AABBs, normals)
#endif
>;

// ReLAX ClassifyTiles: 16x16 tile sky detection (groupshared reduction).
// Denoising range rides as a direct scalar arg so the dispatch has no
// constants-buffer dependency (it runs ahead of the setup CL — item 9).
using RelaxClassifyTilesType = Shader<2,
    Image<float>,      // 0: output tiles (1.0=sky tile)
    Image<float>,      // 1: gIn_ViewZ (R32F)
    float              // 2: denoisingRange
>;

// ReLAX Prepass: Poisson 8-tap spatial blur for noisy input (diffuse + specular)
using RelaxPrepassType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: io_Diff (RGBA16F, in/out)
    Image<float>,      // 2: io_Spec (RGBA16F, in/out)
    Image<float>,      // 3: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 4: gIn_ViewZ (R32F)
    Image<float>      // 5: gIn_Tiles (R8)
>;

// Raw specular hit-distance visualization (debug mode 21): grayscale .w ramp
// (0-10m -> black-white), taken pre-denoiser.
using RelaxRawHitDistVizType = Shader<2,
    Image<float>,      // 0: output
    Image<float>       // 1: raw specular buffer (.w = hit distance)
>;

// ReLAX TemporalAccumulation: bilinear reprojection + dual-rate EMA (diffuse + specular)
// (The former HitDistReconstruct pass — a 3x3 bilateral hit-dist blur feeding TA — was
// removed with RC2: NRD's reconstruction is zero-fill for probabilistic holes and OFF
// by default; TA now consumes the PrePass stochastic-min specular .w directly.)
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
    // Specular channels (17-24)
    Image<float>,      // 17: gIn_Spec_Radiance_HitDist (RGBA16F, noisy specular; .w = PrePass stochastic min)
    Image<float>,      // 18: out_Spec (RGBA16F)
    Image<float>,      // 19: out_SpecFast (RGBA16F)
    Image<float>,      // 20: gPrev_SpecHistory (RGBA16F)
    Image<float>,      // 21: gPrev_SpecFastHistory (RGBA16F)
    Image<float>,      // 22: gPrev_ViewZ2 (alias of 12, for spec disocclusion)
    Image<float>,      // 23: gPrev_SpecHitDist (R32F, virtual motion)
    Image<float>,      // 24: out_SpecHitDist (R32F, virtual motion)
    // Split-kernel VMB prepass outputs (e55) — see _relaxSpecVmbPrepass.
    // FetchA/B/C are only written/read for pixels whose VMB footprint test
    // passed; TA's def() defaults reproduce the not-found path bit-exactly
    // (virtualHistoryAmount = 0 zeroes the VMB side of every blend).
    Image<float>,      // 25: vmbIn_Info   (uv.xy, curvature, |virtualWorldPos|)
    Image<float>,      // 26: vmbIn_Info2  (found, domF, hitDist, prevRoughness)
    Image<float>,      // 27: vmbIn_FetchA (prevSpecVirtual)
    Image<float>,      // 28: vmbIn_FetchB (prevSpecFastVirtual.rgb, .w = fast hitT)
    Image<float>      // 29: vmbIn_FetchC (prevNormalRoughnessVMBPacked.rgb, .w = prevReflectionHitTVMB)
>;

// ReLAX Specular VMB prepass (e55): the virtual-motion production chain
// (dominance/curvature/thin-lens offset/projection/tap validation/history
// fetch) split out of the TA kernel. The TA kernel's compiled code computed
// different values for the same variable at different use sites (see the
// analysis doc, e49-e54); this small kernel compiles correctly and TA
// consumes its outputs as plain textures.
using RelaxSpecVmbPrepassType = Shader<2,
    Buffer<render::RelaxConstants>, // 0: constants
    Image<float>,      // 1: gIn_ViewZ (R32F)
    Image<float>,      // 2: gIn_Normal_Roughness (RGBA16F)
    Image<float>,      // 3: gbuf_bary_motion (.zw = motion vectors)
    Image<float>,      // 4: gIn_Spec (.w = PrePass stochastic-min hit dist)
    Image<float>,      // 5: gPrev_SpecHistory (RGBA16F)
    Image<float>,      // 6: gPrev_SpecFastHistory (RGBA16F)
    Image<float>,      // 7: gPrev_SpecHitDist (R32F)
    Image<float>,      // 8: gPrev_Normal_Roughness (RGBA16F)
    Image<float>,      // 9: gPrev_ViewZ2 (R32F)
    Image<float>,      // 10: gIn_Tiles (R8, sky-tile early out)
    Image<float>,      // 11: out_Info   (uv.xy, curvature, |virtualWorldPos|) — always written
    Image<float>,      // 12: out_Info2  (found, domF, hitDist, prevRoughness) — always written
    Image<float>,      // 13: out_FetchA (prevSpecVirtual / mode-27 viz) — written only when found
    Image<float>,      // 14: out_FetchB (prevSpecFastVirtual.rgb, .w) — written only when found
    Image<float>      // 15: out_FetchC (prevNormalRoughnessVMBPacked.rgb, .w = prevReflectionHitTVMB) — written only when found
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
    Image<float>      // 8: io_SpecFast (RGBA16F, in/out)
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
    Image<float>      // 10: input Spec+variance (RGBA16F)
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
    Image<float>      // 10: input Spec+variance (RGBA16F)
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

    // --- ClassifyTiles (perf R2 item 9) ---
    // Dispatched with (and ahead of) the prefilter so prefilter's sky-tile
    // early-out sees THIS frame's classification — gbufDepth is final at the
    // Pass 1.6 batch (post raster-merge wait). Previously ran inside render()
    // and was one frame stale for the prefilter. render() re-dispatches as a
    // fallback when the prefilter batch did not run this frame.
    void renderClassifyTiles(CommandList& cmdlist, const FrameContext& ctx);

    // --- Main render ---
    // Returns true if denoiser rendered to renderTarget, false if bypassed (no denoiser)
    // rasterDepth is optional — when provided, packed motion is unpacked into gbufBaryMotion.zw
    bool render(Stream& stream, const FrameContext& ctx,
                const Image<float>& renderTarget, bool debugTagNone,
                const Image<float>* rasterDepth = nullptr);

    // --- Accessors ---
    bool enabled() const noexcept { return _enabled; }
    void setEnabled(bool v) noexcept { _enabled = v; }

    // --- Compile-time projection specialization (docs §9 perf investigation) ---
    // The projection-sensitive kernels bake this value as a C++ constant at
    // trace time, so the active projection's paths fold away in DXC — the
    // perspective build keeps the pre-projection instruction sequence. A
    // camera-type flip calls setBakedProjection + recompileProjectionKernels
    // (Pipeline::render does this at the same safe point as
    // _refreshSpecialization; revisits hit the shader disk cache).
    [[nodiscard]] uint bakedProjection() const noexcept { return _bakedProjection; }
    void setBakedProjection(uint p) noexcept { _bakedProjection = p; }
    void recompileProjectionKernels(Device& device);

    uint relaxFrameIdx() const noexcept { return _relaxFrameIdx; }

    // Expose settings for UI
    render::RelaxSettings& settings() noexcept { return _relaxSettings; }

    // Debug state — always present (toJson/fromJson serialize it and callers
    // use the accessors unconditionally); only consumed by the debug-viz
    // shader paths when NT_DEBUG_VIZ is on.
    int  debugStage() const noexcept { return _relaxDebugStage; }
    void setDebugStage(int v) noexcept { _relaxDebugStage = v; }
    bool disableAntilag() const noexcept { return _relaxDisableAntilag; }
    void setDisableAntilag(bool v) noexcept { _relaxDisableAntilag = v; }
    bool disableClamp() const noexcept { return _relaxDisableClamp; }
    void setDisableClamp(bool v) noexcept { _relaxDisableClamp = v; }
    int  histClampDebugViz() const noexcept { return _relaxHistClampDebugViz; }
    void setHistClampDebugViz(int v) noexcept { _relaxHistClampDebugViz = v; }

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
    RelaxTemporalAccumulationType   _relaxTemporalAccumulation;
    RelaxSpecVmbPrepassType         _relaxSpecVmbPrepass;
    RelaxHistoryFixType             _relaxHistoryFix;
    RelaxHistoryClampingType        _relaxHistoryClamping;
    RelaxAtrousType                 _relaxAtrous;
    RelaxAtrousSmemType             _relaxAtrousSmem;
    RelaxAntiFireflyType            _relaxAntiFirefly;
    RelaxMotionCopyType             _relaxMotionCopy;
    RelaxRawHitDistVizType          _relaxRawHitDistViz;
    // Spec-temporal viz display (modes 13-15): RAW passthrough of the TA output
    // + build-epoch marker. Must NOT read the atrous output — the blur smears
    // the debug encoding.
    DenoiserBlitType                _relaxSpecVizBlit;

    // --- ReLAX Persistent Resources (ping-pong) ---
    Buffer<render::RelaxConstants>  _relaxConstantsBuf;
    // CPU staging (member so the pointer outlives CommandList submission —
    // the upload now rides the setup CL instead of a direct stream <<).
    render::RelaxConstants          _relaxConstantsCpu{};
    // Non-perspective state rides ELEMENT 1 of _relaxConstantsBuf (§9 perf
    // fix: keeps the kernel signatures — and thus the perspective shader
    // hashes — identical to the pre-projection build). Aliased members:
    // gFrustum{Right,Up,Forward} = raw camera basis (current frame),
    // gPrevFrustum* = previous frame, gCameraDelta.xy = (fisheye half-fov
    // rad, tan(fov/2)).
    render::RelaxConstants                   _relaxPanoConstsCpu{};
    render::RelaxConstants                   _relaxConstantsUpload[2]{};
    BindlessArray                   _relaxHeap;

    Image<float>  _relaxDiffHistory[2];       // RGBA16F
    Image<float>  _relaxDiffFastHistory[2];   // RGBA16F
    Image<float>  _relaxSpecHistory[2];       // RGBA16F
    Image<float>  _relaxSpecFastHistory[2];   // RGBA16F
    Image<float>  _relaxNormalRoughnessPrev[2]; // RGBA16F
    Image<float>  _relaxViewZPrev[2];         // R32F
    Image<float>  _relaxHistoryLengthPrev[2]; // RG16F (.x=historyLength, .y=specConf)
    Image<float>  _relaxSpecHitDistPrev[2];   // R32F
    // Split-kernel VMB prepass outputs (single-buffered — consumed by TA
    // in the same frame, no ping-pong needed). Info/Info2 are RGBA32F;
    // FetchA/B/C are HALF4 (see PassDenoiser::createImages).
    Image<float>  _relaxVmbInfo;              // uv.xy, curvature, |virtualWorldPos| (always written)
    Image<float>  _relaxVmbInfo2;             // found, domF, hitDist, prevRoughness (always written)
    Image<float>  _relaxVmbFetchA;            // prevSpecVirtual / mode-27 viz (written only when found)
    Image<float>  _relaxVmbFetchB;            // prevSpecFastVirtual.rgb, .w (written only when found)
    Image<float>  _relaxVmbFetchC;            // prevNormalRoughnessVMBPacked.rgb, .w = hitT (written only when found)

    // Transient textures (reused across passes within a frame)
    Image<float>  _relaxTiles;                // R8
    Image<float>  _relaxDiff[2];              // RGBA16F (ping-pong between passes)
    Image<float>  _relaxSpec[2];              // RGBA16F
    Image<float>  _relaxDiffFast;             // RGBA16F
    Image<float>  _relaxSpecFast;             // RGBA16F
    Image<float>  _relaxHistoryLength;        // R16F

    uint _relaxFrameIdx = 0u;

    // Compile-time projection tag baked into the projection-sensitive
    // kernels (see bakedProjection()). 0 = perspective = pre-projection codegen.
    uint _bakedProjection = 0u;

    // Set by renderClassifyTiles, consumed (and cleared) by render() — lets
    // render() re-dispatch classify when the Pass 1.6 batch was skipped
    // (debug-viz early returns) so every tile consumer still sees fresh data.
    bool _tilesClassifiedThisFrame = false;

    // --- Settings ---
    render::RelaxSettings _relaxSettings;
    bool _enabled = true;

    // --- Frame state (set per-frame by Pipeline before render) ---
    uint _frameCount = 0;
    uint _width = 0;
    uint _height = 0;
    bool _accumReset = false;

    // --- Debug state (always present; see the accessors above) ---
    int  _relaxDebugStage = 0;
    bool _relaxDisableAntilag = false;
    bool _relaxDisableClamp = false;
    int  _relaxHistClampDebugViz = 0;

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
    void _populateRelaxConstants(CommandList& cmdlist,
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
