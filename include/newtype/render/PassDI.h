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
// are NOT in this struct. diBiasCorrectionEnabled stays a kernel arg for
// now (fold in later).
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
	float temporalMotionThresh;
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
};

//==========================================================================
// DI Shader Type Aliases
//==========================================================================

// G-Buffer shader: trace primary rays -> write visibility buffer
// With PSR trace-through for dielectric surfaces and stochastic alpha cutout
using GBufShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,      // 0: gbuf_depth (linear ray t)
    luisa::compute::Image<uint>,       // 1: gbuf_vis   (inst_id, prim_id)
    luisa::compute::Image<float>,      // 2: gbuf_bary_motion (RGBA16F: RG=bary, BA=motion)
    luisa::compute::Image<float>,      // 3: glass_throughput (RGBA16F: RGB=attenuation, A=Fresnel)
    newtype::util::CameraData,         // 4: camera
    luisa::compute::Accel,             // 5: TLAS (for PSR trace-through)
    SceneGeometryResources,            // 6: instance/transform/material buffers
    luisa::compute::BindlessArray,     // 7: vertex_bindless (normal reconstruction)
    luisa::compute::BindlessArray,     // 8: tex_bindless (material textures)
    luisa::compute::Image<uint>,       // 9: seed (for stochastic alpha)
    uint                              // 10: has_glass (skip unjit-first trace when 0)
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // 11: procedural bindless
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
    luisa::uint,                                     // 16: presample_tile_count_x
    luisa::uint,                                     // 17: cbField (checkerboard, 0=off)
    luisa::compute::Image<float>,                    // 18: glass_throughput (.rgb=attenuation, .a=fresnel)
    luisa::uint                                      // 18b: hasTransparentShadowCasters (env filter fast path)
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray                    // 19: procedural bindless
#endif
>;

// Presample local light candidates: fill tiles with alias table samples
using PresampleLocalShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<PresampledCandidate>,    // 0: tile output buffer
    luisa::uint,                                     // 1: frame_count
    LightSamplingResources,                          // 2: triangle lights + alias table + emissive counts
    luisa::uint                                      // 3: presample_tile_count_x
>;

// Presample env light candidates: fill tiles with envmap CDF samples
using PresampleEnvShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<PresampledCandidate>,    // 0: tile output buffer
    luisa::uint,                                     // 1: frame_count
    EnvLightResources,                              // 2: envmap + CDFs + rotation
    luisa::uint                                      // 3: presample_tile_count_x
>;

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
    luisa::uint,                       // 18: diBiasCorrectionEnabled
    luisa::uint,                       // 18b: diTemporalBiasRayTraced (RAY_TRACED temporal correction)
    luisa::compute::Accel,             // 18c: TLAS for the temporal visibility re-trace
    luisa::uint                        // 18d: hasTransparentShadowCasters
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray      // 19: procedural bindless
#endif
>;

// Boiling filter: discard outlier reservoirs within 16x16 blocks
using BoilingFilterDIType = luisa::compute::Shader<2,
    luisa::compute::Buffer<Reservoir>,             // 0: reservoir_buffer (in/out)
    float,                                          // 1: strength (0..1, lower = more aggressive)
    luisa::uint                                     // 2: cbField (checkerboard, 0=off)
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
    luisa::uint                        // 15: diBiasCorrectionEnabled
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

    // --- Render passes (append commands to CommandList for batched submission) ---
    void renderGBuffer(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderPresampleLocal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderPresampleEnv(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderCandidate(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength);
    void renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    void flipReservoir() noexcept { _resIdx = 1u - _resIdx; }

    // --- Accessors ---
    luisa::compute::Buffer<Reservoir>& reservoirBuffer() noexcept { return _resBuf[_resIdx]; }
    luisa::compute::Buffer<Reservoir>& reservoirPrevBuffer() noexcept { return _resBuf[1u - _resIdx]; }
    // Presampled env tiles (shared with PassGI for the multi-candidate x2 NEE).
    const luisa::compute::Buffer<PresampledCandidate>& presample_env_tiles() const noexcept { return _presampleEnvTiles; }
    uint visMaxAge() const noexcept { return _visMaxAge; }
    float visMaxDistance() const noexcept { return _visMaxDistance; }
    float envVisMaxDistance() const noexcept { return _envVisMaxDistance; }

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
    SpatialReuseShaderType      _spatialReuseShader;

    // --- Runtime-tunable params buffer (uploaded once per frame) ---
    luisa::compute::Buffer<DIParams> _diParamsBuf;
    DIParams _diParamsCpu{};  // member so the pointer outlives cmdlist submission
    void _populateDiParams(luisa::compute::CommandList& cmdlist) noexcept;

    // --- ReSTIR DI reservoirs (ping-pong) ---
    std::array<luisa::compute::Buffer<Reservoir>, 2u> _resBuf;
    uint _resIdx = 0u;

    // --- Light presampling tiles ---
    static constexpr uint kPresampleTileSize    = 256u;
    static constexpr uint kPresampleEnvTileSize = 64u;
    static constexpr uint kPresampleBlockSize   = 16u;
    static constexpr uint kPresampleEnvBlockSize = 8u;

    luisa::compute::Buffer<PresampledCandidate> _presampleLocalTiles;
    luisa::compute::Buffer<PresampledCandidate> _presampleEnvTiles;
    uint _presampleTileCountX = 0u;
    uint _presampleTileCountY = 0u;
    uint _presampleTileCount  = 0u;

    // --- Light sampling mode (must match other passes) ---
    static constexpr bool kUniformLightSampling = false;

    // --- DI tunable parameters (captured at shader compile time) ---
    static constexpr uint kCandidateCount     = 16u;
    static constexpr uint kBrdfCandidateCount = 1u;
    static constexpr uint kEnvCandidateCount  = 8u;
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
    float _envVisMaxDistance     = 0.0f;    // env lights at infinity - any pixel motion can flip occlusion, so default disabled
    float _temporalMotionThresh = 32.0f;   // max pixel displacement for temporal reuse (px)
    float _brdfCandidateRoughnessCutoff = 0.1f; // skip BRDF candidates below this roughness (narrow-lobe variance source)
    float _mFactorExponent       = 6.0f;   // RTXDI MFactor pow exponent (ref uses 8 w/ pairwise MIS; 6 compensates for single-direction)
    float _mFactorThreshold      = 0.001f; // skip neighbor reservoirs whose MFactor falls below this (0.1% effective M)
    bool  _diBiasCorrectionEnabled = true; // DI BASIC piSum MIS (parallel to _giTier3BiasCorrectionEnabled)
    bool  _diTemporalBiasRayTraced = true; // RAY_TRACED temporal correction: re-trace prev's reused sample (RTXDI Medium preset)

    // FPS-aware visMaxAge derivation state. Synced from PassGI via
    // setVisAgeDerivation(). When _visAgeAutoDerived is true, _populateDiParams
    // overwrites _visMaxAge each frame from _visAgeAccumTimeSec * liveFps / 15.
    bool  _visAgeAutoDerived   = false;
    float _visAgeAccumTimeSec  = 0.5f;
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
	temporalMotionThresh,
	brdfCandidateRoughnessCutoff,
	mFactorExponent,
	mFactorThreshold,
	spatialNeighborCount,
	disocclusionBoostSamples
) {};
