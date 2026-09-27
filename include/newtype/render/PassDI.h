#pragma once
#include <luisa/luisa-compute.h>
#include "cinder/Json.h"
#include "newtype/core/Config.h"
#include "newtype/render/ReSTIR.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/SurfaceResolver.h"
#include "newtype/render/LightSampler.h"
#include "newtype/core/BindingGroups.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/Camera.h"
#include "newtype/core/FrameContext.h"
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif

namespace newtype {
namespace core {

using namespace newtype::render;

//==========================================================================
// DIParams - runtime-tunable DI parameters uploaded per frame
//==========================================================================
// Mirrors PassDenoiser/RelaxConstants pattern. Fields populated from the
// CPU-side PassDI members each frame via _populateDiParams(). Loop-bound
// counts (kCandidateCount, kBrdfCandidateCount, kEnvCandidateCount,
// kSpatialNeighborCount, kDisocclusionBoostSamples) stay compile-time and
// are NOT in this struct. diBiasCorrectionEnabled (and
// hasTransparentShadowCasters) were folded from kernel args into
// compile-time specialization (see the baked snapshot below).
struct DIParams {
	uint  temporalMaxM;
	uint  spatialMaxM;
	uint  visMaxAge;
	float spatialRadius;
	float spatialNormalThresh;
	float spatialDepthThresh;
	float wSumCap;
	float targetPdfFloor;
	float brdfCutoff;
	float brdfCandidateRoughnessCutoff;
	float mFactorExponent;
	float mFactorThreshold;
	// Neighbor counts are runtime-tunable to prevent DXC from unrolling the
	// spatial-reuse loops (the #1 cause of the 55s compile). The shader-side
	// bound is `spatialNeighborCount + disocclusionBoostSamples`, read from
	// this buffer at kernel start. Defaults come from kSpatialNeighborCount /
	// kDisocclusionBoostSamples; the spiral-offset Constant is sized for the
	// compile-time sum so any runtime value <= that sum is safe.
	uint  spatialNeighborCount;
	uint  disocclusionBoostSamples;
	// Local-light candidate count for the initial kernel, runtime-tunable
	// (perf review R2 item 4): a runtime M_total stops DXC unrolling the
	// 24-iteration candidate loop AND enables 8/12/16 A/B profiling without
	// recompiles (RTXDI Medium preset: 8 local + 1 infinite + 1 env).
	uint  localLightCandidateCount;
	// CGNS — RTXDI 3.1 compatibility-guided spatial neighbor selection
	// (docs/rtxdi31_cgns_decorrelation_plan.md). candidateCount is a runtime
	// bound so DXC cannot unroll the selection loop (55s-compile hazard).
	// The selection itself is compiled out entirely when CGNS is off
	// (_bakedCgnsEnabled specialization).
	uint  cgnsCandidateCount;
	float cgnsRadius;             // candidate disk radius (px; RTXDI 50)
	float cgnsNormalExponent;     // normalSim = pow(saturate(dot), exp) (RTXDI 8)
	float cgnsPosOmega;           // sigma = sqrt(omega * depth^2 / pi) (RTXDI 0.05)
	float cgnsGoodScore;          // early-out "good candidate" score (RTXDI 0.5)
	uint  cgnsMaterialGate;       // 0/1: reject dissimilar materials before WRS
	float cgnsMatSimRoughness;    // material gate thresholds (GI defaults =
	float cgnsMatSimF0;           //   RTXDI RAB_AreMaterialsSimilar defaults)
	float cgnsMatSimAlbedo;
};

//==========================================================================
// DI Shader Type Aliases
//==========================================================================

// G-Buffer shader: trace primary rays -> write visibility buffer
// With PSR trace-through for dielectric surfaces and stochastic alpha cutout.
// Fully deterministic (no RNG state since the dispersion channel pick was
// removed — the PSR chain is d-line; docs/dispersion_speckle_fix_plan.md).
using GBufShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: gbuf_depth (linear ray t)
    luisa::compute::Image<uint>,       // 1: gbuf_vis   (inst_id, prim_id)
    luisa::compute::Image<float>,      // 2: gbuf_bary_motion (RGBA16F: RG=bary, BA=motion)
    luisa::compute::Image<float>,      // 3: glass_throughput (RGBA16F: RGB=attenuation, A=Fresnel)
    luisa::compute::Image<float>,      // 4: gbuf_velocity (R16G16F, unjittered, render-res px)
    luisa::compute::Image<float>,      // 5: gbuf_depth_upscale (R32F, NDC z/w non-inverted)
    newtype::util::CameraData,         // 6: camera
    luisa::compute::Accel,             // 7: TLAS (for PSR trace-through)
    SceneGeometryResources,            // 8: instance/transform/material buffers
    luisa::compute::BindlessArray,     // 9: vertex_bindless (normal reconstruction)
    luisa::compute::BindlessArray,     // 10: tex_bindless (material textures)
    luisa::uint                       // 11: psr_bounce_budget (runtime PSR loop bound —
                                      //     stops DXC unrolling the 8× glass body; the
                                      //     comma for later args comes from the #ifs below)
    // Formerly arg 11: has_glass is compile-time specialized (baked in
    // compileImpl; flips recompile via recompileCallables at the
    // render-thread safe point).
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // 12: procedural bindless
#endif
#if NT_ENABLE_SHARC
    ,
    luisa::compute::Image<float>       // 13: rough-glass info (x=roughness, y=eta, z=is_thin)
#endif
>;

// ReSTIR candidate generation: G-Buffer -> M=32 light/BRDF/env candidates -> reservoir
using CandidateShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<DIParams>,              // 0: di_params (runtime-tunable)
    luisa::compute::Buffer<Reservoir>,             // 1: reservoir_buffer
    luisa::compute::Image<float>,      // 1: gbuf_depth
    luisa::compute::Image<uint>,       // 2: gbuf_vis
    luisa::compute::Image<float>,      // 3: gbuf_bary_motion (reads .xy() for barycentrics)
    luisa::compute::Image<uint>,       // 4: seed
    luisa::compute::Accel,             // 5: TLAS (for BRDF ray tracing)
    newtype::util::CameraData,         // 6: camera
    SceneGeometryResources,            // 7: instance/transform/material buffers
    luisa::compute::BindlessArray,     // 8: vertex_bindless
    luisa::compute::BindlessArray,     // 9: tex_bindless (material textures)
    LightSamplingResources,            // 10: triangle lights + alias table + emissive counts
    EnvLightResources,                 // 11: envmap + CDFs + rotation
    float,                             // 12: env_exposure
    luisa::uint,                       // 13: env_candidate_count
    // Presampled light tiles (14-16)
    luisa::compute::Buffer<PresampledCandidate>,    // 14: presample_local_tiles
    luisa::compute::Buffer<PresampledCandidate>,    // 15: presample_env_tiles
    luisa::uint,                                     // 16: frame_count (per-frame window re-roll)
    luisa::uint,                                     // 17: cbField (checkerboard, 0=off)
    luisa::compute::Image<float>                     // 18: glass_throughput (.rgb=attenuation, .a=fresnel)
    // Formerly arg 18b: hasTransparentShadowCasters is compile-time
    // specialized (baked in compileImpl; flips recompile via
    // recompileCallables at the render-thread safe point).
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray                    // 18: procedural bindless
#endif
>;

// Presample local light candidates: fill the fixed pool with alias table samples
using PresampleLocalShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<PresampledCandidate>,    // 0: pool output buffer
    luisa::uint,                                     // 1: frame_count
    LightSamplingResources                           // 2: triangle lights + alias table + emissive counts
>;

// Presample env light candidates: fill the fixed pool with envmap CDF samples
using PresampleEnvShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<PresampledCandidate>,    // 0: pool output buffer
    luisa::uint,                                     // 1: frame_count
    EnvLightResources                                // 2: envmap + CDFs + rotation
>;

// Reservoir zero-fill: recreated reservoir buffers land in recycled pooled-heap
// memory with no zero-fill (LC DX backend). A zeroed entry reads M()==0 ==
// invalid, so every consumer (temporal taps, spatial neighbors, shade) treats
// it as "no history" — the same semantics as a fresh frame. Required on EVERY
// recreation: the first post-resize temporal frame reads prev-slot pixels that
// were never written at the new size (checkerboard: the other field; plain:
// reprojection taps outside the freshly-written region) and garbage entries
// drove degenerate shadow rays into the RT core → GPU wedge (resize TDR
// 2026-09-16, see docs/resize_tdr_root_cause.md).
using ZeroReservoirShaderType = luisa::compute::Shader<1,
    luisa::compute::Buffer<Reservoir>,               // buffer to zero
    luisa::uint>;                                    // element count

// ReSTIR temporal reuse: merge current reservoir with previous frame (with Jacobian reweighting)
using TemporalReuseShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<DIParams>,              // 0: di_params (runtime-tunable)
    luisa::compute::Buffer<Reservoir>,             // 1: reservoir_buffer (current, in/out)
    luisa::compute::Buffer<Reservoir>,             // 1: reservoir_prev (previous frame, read-only)
    luisa::compute::Image<float>,      // 2: gbuf_depth
    luisa::compute::Image<uint>,       // 3: gbuf_vis
    luisa::compute::Image<float>,      // 4: gbuf_bary_motion (RGBA16F: RG=bary, BA=motion)
    luisa::uint,                       // 5: frame_count
    newtype::util::CameraData,         // 6: camera
    SceneGeometryResources,            // 7: instance/transform/material buffers
    luisa::compute::BindlessArray,     // 8: vertex_bindless
    luisa::compute::BindlessArray,     // 9: tex_bindless
    LightSamplingResources,            // 10: triangle lights + alias table + emissive counts
    EnvLightResources,                 // 11: envmap + CDFs + rotation
    float,                             // 12: env_exposure
    luisa::uint,                       // 13: cbField (checkerboard, 0=off)
    luisa::compute::Image<float>,      // 14: glass_throughput
    luisa::compute::Image<float>,      // 15: gbuf_depth_prev
    luisa::compute::Image<uint>,       // 16: gbuf_vis_prev
    luisa::compute::Image<float>,      // 17: gbuf_normal_prev
    luisa::uint,                       // 18b: diTemporalBiasRayTraced (RAY_TRACED temporal correction)
    luisa::compute::Accel              // 18c: TLAS for the temporal visibility re-trace
    // Formerly args 18/18d: diBiasCorrectionEnabled and
    // hasTransparentShadowCasters are compile-time specialized (baked in
    // compileImpl; flips recompile via recompileCallables at the
    // render-thread safe point).
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // 19: procedural bindless
#endif
>;

// Boiling filter: discard outlier reservoirs within 16x16 blocks
using BoilingFilterDIType = luisa::compute::Shader<2,
    luisa::compute::Buffer<Reservoir>,             // 0: reservoir_buffer (in/out)
    float,                                          // 1: strength (0..1, lower = more aggressive)
    luisa::uint,                                    // 2: cbField (checkerboard, 0=off)
    luisa::compute::Buffer<float>                   // 3: boiling_stats [sum, count] (P3-1 global avg)
>;

// Boiling stats pre-pass (P3-1): single-block shared-memory reduction
// computing the frame-global w_sum/M average over W>0 reservoirs (stats[0] =
// average, stats[1] = subsampled count). No atomics — the grid-wide
// float-atomic variant cost ~0.45 ms/frame on the DX emulation path.
using BoilingStatsDIType = luisa::compute::Shader<1,
    luisa::compute::Buffer<Reservoir>,              // reservoir_buffer (read, 1/256 subsample)
    luisa::compute::Buffer<float>,                  // stats [avg, count]
    luisa::uint                                     // total reservoir count
>;

// ReSTIR spatial reuse: share reservoirs with neighbors (ping-pong: read snapshot, write output)
// NOTE: No TLAS parameter — spatial reuse doesn't trace rays. Binding the TLAS would generate
// D3D12 UAV barriers that conflict with the GI stream's concurrent ray tracing operations.
using SpatialReuseShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<DIParams>,              // 0: di_params (runtime-tunable)
    luisa::compute::Buffer<Reservoir>,             // 1: reservoir_output (write)
    luisa::compute::Buffer<Reservoir>,             // 1: reservoir_input  (read snapshot)
    luisa::compute::Image<float>,      // 2: gbuf_depth
    luisa::compute::Image<uint>,       // 3: gbuf_vis
    luisa::compute::Image<float>,      // 4: gbuf_bary_motion (reads .xy() for barycentrics)
    luisa::uint,                       // 5: frame_count
    SceneGeometryResources,            // 6: instance/transform/material buffers
    luisa::compute::BindlessArray,     // 7: vertex_bindless
    luisa::compute::BindlessArray,     // 8: tex_bindless
    LightSamplingResources,            // 9: triangle lights + alias table + emissive counts
    newtype::util::CameraData,         // 10: camera
    EnvLightResources,                 // 11: envmap + CDFs + rotation
    float,                             // 12: env_exposure
    luisa::uint,                       // 13: cbField (checkerboard, 0=off)
    luisa::compute::Image<float>,      // 14: glass_throughput
    luisa::compute::Image<float>       // 15: denoise_normal (CGNS candidate
                                       //     scoring; prefilter world normals)
    // Formerly arg 15: diBiasCorrectionEnabled is compile-time specialized
    // (baked in compileImpl; flips recompile via recompileCallables at the
    // render-thread safe point).
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // 16: procedural bindless
#endif
>;

//==========================================================================
// PassDI Class
//==========================================================================

class PassDI {
public:
    explicit PassDI() noexcept = default;
    PassDI(PassDI&&) noexcept = default;
    PassDI(const PassDI&) = delete;
    PassDI& operator=(const PassDI&) = delete;
    PassDI& operator=(PassDI&&) = delete;
    ~PassDI() noexcept = default;

    void release();

    // --- Lifetime ---
    void compile(luisa::compute::Device& device, scene::Geometry& geom,
                 const render::SurfaceResolverPoly& resolver, bool checkerboard = false);
    // Recompile only the resolver-dependent sub-shaders (gbuf, candidate,
    // temporal, spatial). Resolver-free sub-shaders (presample local/env,
    // boiling) are unaffected by custom-callable DLL changes.
    void recompileCallables(luisa::compute::Device& device,
                            const render::SurfaceResolverPoly& resolver);
    void createImages(luisa::compute::Device& device, uint width, uint height);

    // Zero-fill both reservoir ping-pong slots (M()==0 == invalid). Call after
    // createImages on a live stream; ordered by stream FIFO before any frame
    // work submitted afterwards. Requires compile() to have run.
    void zeroInitBuffers();

    // --- Render passes (append commands to CommandList for batched submission) ---
    void renderGBuffer(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderPresampleLocal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderPresampleEnv(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderCandidate(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength);
    void renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void flipReservoir() noexcept { _resIdx = 1u - _resIdx; }

    // --- Presample regeneration gating ---
    // Presampled tiles are valid importance samples of the CURRENT light
    // distribution as long as it hasn't changed: candidates store
    // (light_idx, bary, inv_source_pdf) while emission/geometry are read
    // fresh at consumption. Regeneration is therefore dirty-gated instead of
    // re-running over every screen tile every frame. Markers are set by
    // Pipeline where the light/env state changes.
    void markPresampleLocalDirty() noexcept { _presampleLocalDirty = true; }
    void markPresampleEnvDirty() noexcept { _presampleEnvDirty = true; }

    // --- Accessors ---
    luisa::compute::Buffer<Reservoir>& reservoirBuffer() noexcept { return _resBuf[_resIdx]; }
    luisa::compute::Buffer<Reservoir>& reservoirPrevBuffer() noexcept { return _resBuf[1u - _resIdx]; }
    // Presampled env tiles (shared with PassGI for the multi-candidate x2 NEE).
    const luisa::compute::Buffer<PresampledCandidate>& presample_env_tiles() const noexcept { return _presampleEnvTiles; }
    uint visMaxAge() const noexcept { return _visMaxAge; }
    float visMaxDistance() const noexcept { return _visMaxDistance; }
    float envVisMaxDistance() const noexcept { return _envVisMaxDistance; }
    uint dispShadowInterfaces() const noexcept { return _dispShadowInterfaces; }
    float dispShadowSplit() const noexcept { return _dispShadowSplit; }

    // FPS-aware coordination: when enabled, visMaxAge is auto-derived each
    // frame from a seconds-valued accumulation time (with /15 ratio — vis
    // reuse is much shorter than GI accumulation by design). Pipeline syncs
    // these from PassGI's giAccumulationTime knob before invoking DI render.
    // See _populateDiParams.
    void setVisAgeDerivation(bool enabled, float accumTimeSec) noexcept {
        _visAgeAutoDerived   = enabled;
        _visAgeAccumTimeSec  = accumTimeSec;
    }
    bool visAgeAutoDerived() const noexcept { return _visAgeAutoDerived; }

    // --- UI ---
    void drawUi();

    // --- Config serialization ---
    void toJson(ci::Json& j) const;
    void fromJson(const ci::Json& j);

private:
    // --- Scene reference (set at compile time, not owned) ---
    scene::Geometry* _geom = nullptr;

#if NT_ENABLE_PROCEDURAL
    // Procedural bindless array (non-owning pointer, wired by Pipeline)
    const luisa::compute::BindlessArray* _procBindlessPtr = nullptr;
public:
    void set_procedural_buffers(
        const luisa::compute::BindlessArray& proc_bindless) noexcept {
        _procBindlessPtr = &proc_bindless;
    }
private:
#endif

    // --- Compile-time feature toggles ---
    bool _checkerboard = true;

    // --- Compile helpers ---
    // compileImpl: full compile body shared by compile() and recompileCallables().
    // When resolverOnly=true, skips the resolver-free sub-shaders (presample local/env,
    // boiling) since their IR is unchanged by a custom-callable DLL reload.
    void compileImpl(luisa::compute::Device& device,
                     const render::SurfaceResolverPoly& resolver,
                     bool resolverOnly);

    // --- Compiled shaders ---
    GBufShaderType              _gbufShader;
    PresampleLocalShaderType    _presampleLocalShader;
    PresampleEnvShaderType      _presampleEnvShader;
    CandidateShaderType         _candidateShader;
    TemporalReuseShaderType     _temporalReuseShader;
    BoilingFilterDIType         _boilingFilterDI;
    BoilingStatsDIType          _boilingStatsDI;
    SpatialReuseShaderType      _spatialReuseShader;
    ZeroReservoirShaderType     _zeroReservoirShader;

    // Boiling stats (P3-1): [0] = frame-global avg of w_sum/M over W>0
    // reservoirs (written directly by the single-block stats kernel), [1] =
    // subsampled W>0 count. Lazily allocated with the boiling shaders.
    luisa::compute::Buffer<float> _boilingStatsBuf;

    // --- Runtime-tunable params buffer (uploaded once per frame) ---
    luisa::compute::Buffer<DIParams> _diParamsBuf;
    DIParams _diParamsCpu{};  // member so the pointer outlives cmdlist submission
    void _populateDiParams(luisa::compute::CommandList& cmdlist) noexcept;

    // --- ReSTIR DI reservoirs (ping-pong) ---
    std::array<luisa::compute::Buffer<Reservoir>, 2u> _resBuf;
    uint _resIdx = 0u;

    // --- Light presampling pools (fixed, resolution-independent) ---
    // Phase 2 (docs/di-tile-block-artifact-fix-plan.md): pools decoupled from
    // screen resolution — a fixed set of kPresamplePoolTileCount windows, each
    // consumer block (16×16 screen tile) picks its window per frame by hashing
    // its tile coords + frame index (block-coherent: one window serves the
    // wavefront as an L1 broadcast, mirroring RTXDI's random per-frame tile
    // selection). Replaces the screen-tile-indexed frozen windows that turned
    // a window with zero usable entries into a deterministic per-tile lottery
    // (the 16×16 stuck-block artifact).
    static constexpr uint kPresamplePoolTileCount = 128u;  // power of two (hash mask); tunable 64–256
    static constexpr uint kPresampleTileSize      = 256u;  // local-light entries per window
    static constexpr uint kPresampleEnvTileSize   = 256u;  // env entries per window (was 64: window-miss
                                                           // probability (1−f)^64 froze per tile; ^256 +
                                                           // per-frame re-roll makes it negligible)
    static constexpr uint kPresampleBlockSize     = 16u;   // consumer block size (screen tile)

    luisa::compute::Buffer<PresampledCandidate> _presampleLocalTiles;
    luisa::compute::Buffer<PresampledCandidate> _presampleEnvTiles;
    // Regeneration gates (see markPresampleLocalDirty/markPresampleEnvDirty).
    bool _presampleLocalDirty = true;
    bool _presampleEnvDirty = true;

    // --- Light sampling mode (must match other passes) ---
    static constexpr bool kUniformLightSampling = false;

    // --- DI tunable parameters (captured at shader compile time) ---
    // PSR glass trace-through bounce budget (perf review R2 item 8): class
    // scope so renderGBuffer can pass it as the runtime loop bound (a
    // constant bound lets DXC unroll the ~200-line body — the 55s-compile
    // hazard class). Glassless scenes additionally specialize the whole
    // glass path out via _bakedHasGlass.
    static constexpr uint kMaxGlassBounces = 8u;
    static constexpr uint kCandidateCount     = 16u;
    static constexpr uint kBrdfCandidateCount = 1u;
    static constexpr uint kEnvCandidateCount  = 8u;
    // Tile-coherent candidate entry RNG (perf review R2 item 3; RTXDI
    // Doc/ShaderAPI.md "coherentRng" pattern, RTXDI_TILE_SIZE_IN_PIXELS=16):
    // all pixels of a 16×16 block derive the per-candidate presample entry
    // pick from one seed-image texel, so the wavefront reads the same tile
    // entry per candidate — one cache line serves the group and the
    // remaining per-pixel light-buffer reads become coherent broadcasts.
    // Candidate SETS correlate within a block (RIS stays unbiased; RTXDI
    // ships this as default). Flip to false to restore the per-pixel pick.
    static constexpr bool kTileCoherentCandidateRng = true;
    // Runtime local-light candidate count (perf review R2 item 4); default
    // preserves today's kCandidateCount - kBrdfCandidateCount behavior.
    uint  _localLightCandidateCount = kCandidateCount - kBrdfCandidateCount;
    uint  _temporalMaxM          = 512u;    // RTXDI grows M ~historyLength*candidates (cap 16383); 128 saturated within a few frames
    // RTXDI Medium preset: 1 spatial neighbor normally (+8 boost on disocclusion).
    // With RAY_TRACED temporal correction + M growth, spatial carries far less of
    // the quality burden — each neighbor costs fetch_light_sample + p_hat (+ a full
    // surface resolve per neighbor in the bias-correction second pass).
    static constexpr uint kSpatialNeighborCount = 2u;
    uint  _spatialMaxM           = 256u;   // cap M growth from spatial reuse
    uint  _spatialRadius         = 20u;
    static constexpr uint kDisocclusionBoostSamples = 4u;  // extra spatial neighbors when M < temporalMaxM
    float _spatialNormalThresh   = 0.5f;
    float _spatialDepthThresh    = 0.1f;
    float _wSumCap               = 1e6f;
    float _targetPdfFloor        = 1e-8f;
    float _brdfCutoff            = 1e-4f;   // discard BRDF hits beyond sqrt((1/cutoff - 1) * pdf)
    uint  _visMaxAge             = 4u;      // max frames to reuse confirmed visibility (RTXDI-style)
    float _visMaxDistance        = 16.0f;   // max screen-space distance (px) for visibility reuse (local lights)
    // Env-light shade-time visibility reuse radius (px). 0 disables — every
    // env-lit pixel then re-traces its full shadow ray every frame (a
    // dispersive walk in glass scenes). MEASURED 2026-09-15 (perf review R2
    // item 2): 8 px is perf-NEUTRAL on hit/cornell @1440² static camera —
    // the shade-ray amortization is offset by the temporal re-trace cycle
    // (visibility age grows until the RAY_TRACED re-trace resets it). RTXDI
    // reuses final visibility for ALL lights at 16 px, but our temporal
    // chain cycles differently; revisit together with boundary-selective
    // vis-aware weighting (docs/boundary-selective-vis-reuse-plan.md) or in
    // env-heavy scenes under motion. Default stays 0 (no behavior change
    // without a measured win); the knob remains UI/JSON-tunable.
    float _envVisMaxDistance     = 0.0f;
    // Dispersive shadow walk dials (docs/Dispersive-Glass-Shadows.md Phase 2;
    // consumed by trace_shadow in the shade kernel).
    uint  _dispShadowInterfaces  = 2u;      // glass crossings per channel sub-walk after the fan-out (dial (a); a sphere needs 1, stacked glass 2 — deeper stacks truncate keeping the accumulated attenuation)
    float _dispShadowSplit       = 1.0f;    // RGB-split saturation: 1 = exact per-channel estimator, 0 = d-line gray (softens primary-hard fringe hues, zero cost)
    float _brdfCandidateRoughnessCutoff = 0.1f; // skip BRDF candidates below this roughness (narrow-lobe variance source)
    float _mFactorExponent       = 6.0f;   // RTXDI MFactor pow exponent (ref uses 8 w/ pairwise MIS; 6 compensates for single-direction)
    float _mFactorThreshold      = 0.001f; // skip neighbor reservoirs whose MFactor falls below this (0.1% effective M)
    bool  _diBiasCorrectionEnabled = true; // DI BASIC piSum MIS (parallel to _giTier3BiasCorrectionEnabled)
    bool  _diTemporalBiasRayTraced = true; // RAY_TRACED temporal correction: re-trace prev's reused sample (RTXDI Medium preset)

    // --- CGNS: RTXDI 3.1 compatibility-guided spatial neighbor selection ---
    // Off by default (bit-identical spiral path); the flip is a compile-time
    // specialization (_bakedCgnsEnabled) so the selection phase and its
    // texture reads compile out entirely when disabled.
    static constexpr uint kCgnsNeighborCount = 4u;   // WRS pick slots (RTXDI SPATIAL_HEURISTIC_MAX_NEIGHBORS)
    bool  _cgnsEnabled           = false;
    uint  _cgnsCandidateCount    = 32u;    // RTXDI PT_NEIGHBOR_SELECTION_CANDIDATE_COUNT
    float _cgnsRadius            = 50.0f;  // px (RTXDI PT_NEIGHBOR_SELECTION_SEARCH_RADIUS)
    float _cgnsNormalExponent    = 8.0f;   // RTXDI NeighborCompatibilityScore beta
    float _cgnsPosOmega          = 0.05f;  // world-pos tolerance scale (RTXDI)
    float _cgnsGoodScore         = 0.5f;   // early-out "good" score (RTXDI)
    bool  _cgnsMaterialGate      = true;   // engine extension: sim-key reject before WRS
    float _cgnsMatSimRoughness   = 0.25f;  // GI/RTXDI RAB_AreMaterialsSimilar defaults
    float _cgnsMatSimF0          = 0.5f;
    float _cgnsMatSimAlbedo      = 0.5f;

    // FPS-aware visMaxAge derivation state. Synced from PassGI via
    // setVisAgeDerivation(). When _visAgeAutoDerived is true, _populateDiParams
    // overwrites _visMaxAge each frame from _visAgeAccumTimeSec * liveFps / 15.
    bool  _visAgeAutoDerived   = false;
    float _visAgeAccumTimeSec  = 0.5f;

    // --- Compile-time specialization snapshot ---------------------------------
    // diBiasCorrectionEnabled + hasTransparentShadowCasters + hasGlass are
    // baked into the candidate/temporal/spatial/gbuffer kernels at compileImpl
    // time (former UInt args). Bit-identical by construction: the baked
    // constant is the exact value the removed runtime arg would have carried.
    // Pipeline calls specializationChanged() at the render-thread safe point
    // and recompileCallables() on a flip (same path as a callable DLL
    // hot-reload). hasGlass additionally C++-gates the PSR glass machinery
    // (unjit classification trace, medium-list init, both glass branches) —
    // glassless scenes skip it entirely (perf review R2 item 8); the
    // invariant is Geometry::has_visible_glass()==false ⇒ no visible
    // type-3/11 material on meshes or procedural instances.
    uint _bakedDiBiasCorrectionEnabled = 1u;
    uint _bakedTransparentShadowCasters = 0u;
    uint _bakedHasGlass = 0u;
    uint _bakedCgnsEnabled = 0u;

public:
    [[nodiscard]] bool specializationChanged() const noexcept {
        return _bakedDiBiasCorrectionEnabled != (_diBiasCorrectionEnabled ? 1u : 0u) ||
               (_geom != nullptr &&
                _bakedTransparentShadowCasters !=
                    (_geom->has_transparent_shadow_casters() ? 1u : 0u)) ||
               (_geom != nullptr &&
                _bakedHasGlass != (_geom->has_visible_glass() ? 1u : 0u)) ||
               _bakedCgnsEnabled != (_cgnsEnabled ? 1u : 0u);
    }
};

} // namespace core
} // namespace newtype

// Register DIParams as a LuisaCompute DSL struct - must be outside namespace
LUISA_STRUCT(newtype::core::DIParams,
	temporalMaxM,
	spatialMaxM,
	visMaxAge,
	spatialRadius,
	spatialNormalThresh,
	spatialDepthThresh,
	wSumCap,
	targetPdfFloor,
	brdfCutoff,
	brdfCandidateRoughnessCutoff,
	mFactorExponent,
	mFactorThreshold,
	spatialNeighborCount,
	disocclusionBoostSamples,
	localLightCandidateCount,
	cgnsCandidateCount,
	cgnsRadius,
	cgnsNormalExponent,
	cgnsPosOmega,
	cgnsGoodScore,
	cgnsMaterialGate,
	cgnsMatSimRoughness,
	cgnsMatSimF0,
	cgnsMatSimAlbedo
) {};
