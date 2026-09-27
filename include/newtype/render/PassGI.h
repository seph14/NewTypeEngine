#pragma once
#include <luisa/luisa-compute.h>
#include "cinder/Json.h"
#include "newtype/core/Config.h"
#include "newtype/render/GIReservoir.h"
#include "newtype/render/GIBounce.h"
#include "newtype/render/GIShading.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/ReSTIR.h"
#include "newtype/render/SurfaceResolver.h"
#include "newtype/render/LightSampler.h"
#include "newtype/core/BindingGroups.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/Camera.h"
#include "newtype/core/FrameContext.h"
#if NT_ENABLE_SHARC
// Phase 2 query hook: the initial shader binds the SHARC cache read-only.
// PassSharc.h (not Sharc.h) for the SharcParams staging type + engine key
// alias; no cycle — PassSharc does not include PassGI.
#include "newtype/render/PassSharc.h"
#endif
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif

namespace newtype {
namespace core {

using namespace newtype::render;

//==========================================================================
// GIParams - runtime-tunable GI parameters uploaded per frame
//==========================================================================
// Mirrors PassDenoiser/RelaxConstants pattern. Fields populated from the
// CPU-side PassGI members each frame via _populateGiParams(). Loop-bound
// counts (kGiInitialCandidateCount, kGiInitialEnvCandidateCount,
// kGiGlassInitialSamples) stay compile-time and are NOT in this struct.
// kGiSpatialNeighborCount is the compile-time MAX (sizes the spiral offsets
// at trace time); the runtime value lives in giSpatialNeighborCount below
// so DXC can't statically bound the spatial-reuse loops and unroll them
// (same fix as DI commit 29b6cda — that dropped DI spatial compile from
// >55s to 800ms; GI was 3s for the same reason).
struct GIParams {
	uint  giSpatialRadius;
	uint  giSpatialNeighborCount;
	float giWSumCap;
	uint  giMaxAge;
	uint  giTemporalMaxM;
	uint  giSpatialMaxM;
	float giMaxRadiance;
	float giRoughnessInvalidationThresh;
	float giMatSimRoughness;
	float giMatSimF0;
	float giMatSimAlbedo;
	float giMISRoughness;
	float giBiasCorrectionDivergenceThresh;
	float spatialNormalThresh;
	float spatialDepthThresh;
	uint  giTier3BiasCorrectionEnabled;
	uint  giSuppressAwayFromLight;
	float giSuppressAwayFromLightThresh;
	// Temporal freshness gate knee (log2 radiance-ratio units); <= 0 disables.
	// Temporal freshness knee (log2 units): full merge weight up to 2^knee
	// prev/current radiance divergence, zero at 2^(knee+2). <= 0 disables the
	// gate. See docs/gi_temporal_dynamic_receiver_plan.md §3.2.
	float giFreshnessKnee;
	// Stagnancy-decorrelation EMA factor for the smoothing pass
	// (RTXDI decorrelationEmaFactor): 1 = use current raw age only,
	// 0 = keep smoothed history indefinitely.
	float giDecorrelationEmaFactor;
};

//==========================================================================
// GI Shader Type Aliases
//==============================================================================

// GIReservoir zero-fill: same rationale as PassDI's ZeroReservoirShaderType —
// recreated GI reservoir/history buffers land in recycled pooled-heap memory
// with no zero-fill; zeroed entries read M()==0 == invalid so GI temporal /
// spatial / shade treat them as "no history" instead of tracing degenerate
// rays from garbage sample positions (resize-TDR root cause 2026-09-16).
using ZeroGIReservoirShaderType = luisa::compute::Shader<1,
	luisa::compute::Buffer<GIReservoir>,              // buffer to zero
	luisa::uint>;                                     // element count

// GI Initial Sampling: G-Buffer -> BRDF ray -> secondary hit -> GI reservoir
using GIInitialShaderType = luisa::compute::Shader<2,
	luisa::compute::Buffer<GIParams>,                  // 0: gi_params (runtime-tunable)
	luisa::compute::Buffer<GIReservoir>,              // 1: gi_reservoir_buffer (output)
	luisa::compute::Image<float>,                     // 1: gbuf_depth
	luisa::compute::Image<uint>,                      // 2: gbuf_vis
	luisa::compute::Image<float>,                     // 3: gbuf_bary_motion (reads .xy() for bary)
	luisa::compute::Image<uint>,                      // 4: seed
	luisa::compute::Accel,                            // 5: TLAS (secondary + shadow rays)
	newtype::util::CameraData,                        // 6: camera
	SceneGeometryResources,                           // 7: instance/transform/material buffers
	luisa::compute::BindlessArray,                    // 8: vertex_bindless
	luisa::compute::BindlessArray,                    // 9: tex_bindless (material textures)
	LightSamplingResources,                           // 10: triangle lights + alias table + emissive counts
	EnvLightResources,                                // 11: envmap + CDFs + rotation
	float,                                            // 12: env_exposure
	luisa::compute::Buffer<render::PresampledCandidate>, // 13: presample_env_tiles
	luisa::uint,                                      // 14: presample_env_total_entries (tile_count_x * tile_count_y * 64)
	luisa::uint,                                      // 15: cbField (checkerboard, 0=off)
	luisa::compute::Image<float>,                     // 16: glass_throughput (RGB=attenuation, A=Fresnel)
	// Formerly args 17/18/21 + sharcQueryOn: oneBounce, giScale,
	// hasTransparentShadowCasters and sharcQueryOn are compile-time
	// specialized now (baked in compileImpl; flips recompile via
	// refreshSpecialization — see the specialization block in PassGI.cpp).
	luisa::uint,                                      // 17: giSuppressAwayFromLight (1=suppress GI on surfaces facing away from primary light)
	float                                           // 18: giSuppressAwayFromLightThresh (skip when dot(ns, light_dir) < thresh)
#if NT_ENABLE_PROCEDURAL
	,
	luisa::compute::BindlessArray                     // 19: procedural bindless (instances, positions, indices, AABBs, normals)
#endif
#if NT_ENABLE_SHARC
	// Phase 2 query hook: read-only cache + grid params (same staging buffer
	// as Update/Resolve, so query keys match) + hit/miss counter buffer
	// (PassSharc-owned diagnostics).
	,
	luisa::compute::Buffer<SharcParams>               //    sharc params (sceneScale/levelBias/capacity)
	,
	luisa::compute::Buffer<SharcKeyHost>              //    hash entries (read)
	,
	luisa::compute::Buffer<render::SharcPackedData>   //    resolved radiance (read)
	,
	luisa::compute::Buffer<luisa::uint>               //    query stats (atomic: [0]=hits, [1]=misses)
#endif
>;

// GI Upsample: 2x2 nearest replication (integer div) half-res reservoirs -> full-res
// (not bilinear — see the upsample kernel's coord mapping in PassGI.cpp)
using GIUpsampleShaderType = luisa::compute::Shader<2,
	luisa::compute::Buffer<GIReservoir>,              // 0: output full-res
	luisa::compute::Buffer<GIReservoir>,              // 1: input half-res
	luisa::uint,                                      // 2: half_width
	luisa::uint                                       // 3: half_height
>;


// GI Temporal Reuse: merge current GI reservoir with previous frame (Jacobian reweighting)
using GITemporalReuseShaderType = luisa::compute::Shader<2,
	luisa::compute::Buffer<GIParams>,                  // 0: gi_params (runtime-tunable)
	luisa::compute::Buffer<GIReservoir>,              // 1: gi_reservoir_buffer (in/out)
	luisa::compute::Buffer<GIReservoir>,              // 1: gi_reservoir_prev (previous frame, read-only)
	luisa::compute::Image<float>,                     // 2: gbuf_depth
	luisa::compute::Image<uint>,                      // 3: gbuf_vis
	luisa::compute::Image<float>,                     // 4: gbuf_bary_motion (RGBA16F: RG=bary, BA=motion)
	luisa::uint,                                      // 5: frame_count
	newtype::util::CameraData,                        // 6: camera
	SceneGeometryResources,                           // 7: instance/transform/material buffers
	luisa::compute::BindlessArray,                    // 8: vertex_bindless
	luisa::compute::BindlessArray,                    // 9: tex_bindless (material textures)
	luisa::uint,                                      // 10: cbField (checkerboard, 0=off)
	luisa::compute::Image<float>,                     // 11: glass_throughput
	luisa::compute::Image<float>,                     // 12: gbuf_depth_prev (prev frame)
	luisa::compute::Image<uint>,                      // 13: gbuf_vis_prev (prev frame)
	luisa::compute::Image<float>,                     // 14: denoise_normal_prev (prev frame)
	luisa::uint                                      // 15: tier3BiasCorrectionEnabled (1=Tier 3 piSum MIS, 0=legacy 1/M)
	// Formerly arg 16: giScale is compile-time specialized (baked in
	// compileImpl; flips recompile via refreshSpecialization).
#if NT_ENABLE_PROCEDURAL
	,
	luisa::compute::BindlessArray                     // 16: procedural bindless (instances, positions, indices, AABBs, normals)
#endif
>;


// GI Spatial Reuse: share GI reservoirs with neighbors (ping-pong)
using GISpatialReuseShaderType = luisa::compute::Shader<2,
	luisa::compute::Buffer<GIParams>,                  // 0: gi_params (runtime-tunable)
	luisa::compute::Buffer<GIReservoir>,              // 1: gi_reservoir_output (write)
	luisa::compute::Buffer<GIReservoir>,              // 1: gi_reservoir_input (read snapshot)
	luisa::compute::Image<float>,                     // 2: gbuf_depth
	luisa::compute::Image<uint>,                      // 3: gbuf_vis
	luisa::compute::Image<float>,                     // 4: gbuf_bary_motion (reads .xy() for bary)
	luisa::compute::Accel,                            // 5: TLAS (post-merge current→x2 visibility check)
	luisa::uint,                                      // 6: frame_count
	SceneGeometryResources,                           // 7: instance/transform/material buffers
	luisa::compute::BindlessArray,                    // 8: vertex_bindless
	luisa::compute::BindlessArray,                    // 9: tex_bindless (material textures)
	newtype::util::CameraData,                        // 10: camera
	luisa::uint,                                      // 11: cbField (checkerboard, 0=off)
	luisa::compute::Image<float>,                     // 12: glass_throughput
	luisa::compute::Image<float>,                     // 13: denoise_normal (cached world-space normals)
	luisa::uint                                       // 14: tier3BiasCorrectionEnabled (1=Tier 3 piSum MIS, 0=legacy 1/M)
#if NT_ENABLE_PROCEDURAL
	, luisa::compute::BindlessArray                   // 15: procedural bindless
#endif
>;


// Boiling filter for GI reservoirs
using BoilingFilterGIType = luisa::compute::Shader<2,
    luisa::compute::Buffer<GIReservoir>,              // 0: gi_reservoir_buffer (in/out)
    float,                                             // 1: strength (0..1, lower = more aggressive)
    luisa::uint,                                      // 2: cbField (checkerboard, 0=off)
    luisa::compute::Image<float>,                     // 3: firefly flag output (R=1 outlier;
                                                      //    written when arg 4 != 0, compacted dims)
    luisa::uint                                       // 4: fireflyReplace (1 = flag outliers for
                                                      //    shade-time decorrelation swap instead
                                                      //    of culling them here; RTXDI 3.1
                                                      //    fireflyReplacementFilter)
>;

// Stagnancy smoothing (RTXDI 3.1 ComputeSmoothedDuplicationMap adaptation):
// EMA-blends the current raw stagnancy (final reservoir age / giMaxAge) with
// a reprojected, 5x5 a-trous (step 2) average of the previous smoothed map.
// Ping-pong R16F pair at the shade (checkerboard-compacted) resolution.
using GISmoothStagnancyShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<GIParams>,                  // 0: gi_params (giMaxAge, EMA factor)
    luisa::compute::Buffer<GIReservoir>,              // 1: final GI reservoirs (age source;
                                                      //    the same buffer shade reads)
    luisa::compute::Image<float>,                     // 2: gbuf_bary_motion (.zw NDC motion)
    luisa::compute::Image<float>,                     // 3: gbuf_depth (current, full-res)
    luisa::compute::Image<uint>,                      // 4: gbuf_vis (current, full-res)
    luisa::compute::Image<float>,                     // 5: gbuf_depth_prev (full-res)
    luisa::compute::Image<uint>,                      // 6: gbuf_vis_prev (full-res)
    luisa::compute::Image<float>,                     // 7: stagnancy_prev (read)
    luisa::compute::Image<float>,                     // 8: stagnancy_out (write)
    luisa::uint,                                      // 9: cbField
    luisa::uint                                       // 10: resetHistory (1 = bootstrap from raw)
>;

// Zero-fill for the stagnancy/firefly images (recycled-heap garbage reads
// otherwise; same rationale as ZeroGIReservoirShaderType).
using ZeroFloatImageShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,                     // image to zero
    luisa::uint                                       // pixel count (w*h)
>;

//==========================================================================
// PassGI Class
//==========================================================================

class PassGI {
public:
	explicit PassGI() noexcept = default;
	PassGI(PassGI&&) noexcept = default;
	PassGI(const PassGI&) = delete;
	PassGI& operator=(const PassGI&) = delete;
	PassGI& operator=(PassGI&&) = delete;
	~PassGI() noexcept = default;

	// --- Lifetime ---
	void compile(luisa::compute::Device& device,
	             const render::SurfaceResolverPoly& resolver, bool checkerboard = false,
	             bool transparentShadowCasters = false);
	// Recompile only the resolver-dependent sub-shaders (initial, temporal, spatial).
	// Resolver-free sub-shaders (upsample, boiling) are unaffected by custom-callable
	// DLL changes. Also re-bakes the specialization flags (oneBounce/giScale/
	// hasTransparentShadowCasters/sharcQueryOn) — Pipeline calls this at the
	// render-thread safe point when specializationChanged() reports a flip.
	void recompileCallables(luisa::compute::Device& device,
	                        const render::SurfaceResolverPoly& resolver,
	                        bool transparentShadowCasters = false);
	void createImages(luisa::compute::Device& device, uint width, uint height);
	// Zero-fill every GI reservoir/history buffer (both full-res slots, both
	// half-res slots, the half->full upsample output, and the initial
	// snapshot). Call after createImages; requires compile() to have run.
	void zeroInitBuffers();
	void release();

	// --- Render passes (append commands to CommandList for batched submission) ---
	void renderInitial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	void renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	void renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength);
	void renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	// Stagnancy smoothing (RTXDI 3.1 decorrelation support; see
	// docs/rtxdi31_cgns_decorrelation_plan.md). Call AFTER the pre-shade
	// reservoir flip so it reads the same final GI reservoir buffer shade
	// reads. Rotates the ping-pong pair; stagnancyTexture() returns the
	// freshly written slot afterwards.
	void renderStagnancySmooth(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	void copyReservoirs(luisa::compute::CommandList& cmdlist);
	void snapshotInitial(luisa::compute::CommandList& cmdlist) noexcept;
	void renderUpsample(luisa::compute::CommandList& cmdlist, uint width, uint height);
	// Flips the active-mode reservoir pair: the full-res _giResBuf pair in
	// full-res mode, the half-res pair in half-res mode. Full-res mode flips
	// twice per frame (pre-shade + end-of-frame); half-res flips exactly once
	// (end-of-frame) so temporal history carries the temporal+boiling output.
	void flipReservoir() noexcept {
	    if (_giHalfRes) { _giResIdxHalf = 1u - _giResIdxHalf; }
	    else            { _giResIdx     = 1u - _giResIdx; }
	}

	// --- Accessors ---
	bool enabled() const noexcept { return _enabled; }
	void setEnabled(bool v) noexcept { _enabled = v; }

	bool giOneBounce() const noexcept { return _giOneBounce; }
	void setGIOneBounce(bool v) noexcept { _giOneBounce = v; }

	bool giHalfRes() const noexcept { return _giHalfRes; }
	void setGIHalfRes(bool v) noexcept { _giHalfRes = v; }

	luisa::compute::Buffer<GIReservoir>& reservoirBuffer() noexcept {
		return _giHalfRes ? _giReservoirBufferFull : _giResBuf[_giResIdx];
	}
	luisa::compute::Buffer<GIReservoir>& reservoirPrevBuffer() noexcept { return _giResBuf[1u - _giResIdx]; }
	// Half-res working pair (half-res mode): current = initial/temporal/boiling
	// target, prev = temporal history. Upsample reads current after boiling.
	luisa::compute::Buffer<GIReservoir>& halfReservoirBuffer() noexcept { return _giResBufHalf[_giResIdxHalf]; }
	luisa::compute::Buffer<GIReservoir>& halfReservoirPrevBuffer() noexcept { return _giResBufHalf[1u - _giResIdxHalf]; }
	luisa::compute::Buffer<GIReservoir>& initialSnapshotBuffer() noexcept { return _giInitialSnapshot; }

	// --- Stagnancy decorrelation (RTXDI 3.1) ---
	// Smoothed-stagnancy texture freshly written this frame (shaded-space /
	// checkerboard-compacted dims — read with the shade kernel's dispatch_id).
	luisa::compute::Image<float>& stagnancyTexture() noexcept { return _giStagnancy[_giStagnancyIdx]; }
	// Firefly flag written by the GI boiling filter when replacement is on.
	luisa::compute::Image<float>& fireflyFlagTexture() noexcept { return _giFireflyFlag; }
	uint  giDecorrelationMode()    const noexcept { return _giDecorrelationMode; }
	float giDecorrelationFactor()  const noexcept { return _giDecorrelationFactor; }
	float giDecorrelationStagnancyExponent() const noexcept { return _giDecorrelationStagnancyExponent; }
	// Firefly replacement is a no-op without temporal reuse: mode != None and
	// factor > 0 gate it exactly like RTXDI_PTDetectDecorrelationFireflies.
	bool  giFireflyReplaceActive() const noexcept {
		return _giDecorrelationFirefly && _giDecorrelationMode != 0u && _giDecorrelationFactor > 0.0f;
	}
	float giDecorrelationMultiplyBound() const noexcept { return _giDecorrelationMultiplyBound; }

	// --- Stagnancy decorrelation bulk access (DLSS-RR engage/disengage) ---
	struct GiDecorrelationSettings {
		uint  mode;
		float factor;
		float stagnancyExponent;
		float emaFactor;
		bool  firefly;
		float multiplyBound;
	};
	[[nodiscard]] GiDecorrelationSettings giDecorrelationSettings() const noexcept {
		return { _giDecorrelationMode, _giDecorrelationFactor,
			     _giDecorrelationStagnancyExponent, _giDecorrelationEmaFactor,
			     _giDecorrelationFirefly, _giDecorrelationMultiplyBound };
	}
	void applyGiDecorrelationSettings(const GiDecorrelationSettings& s) noexcept {
		_giDecorrelationMode = s.mode;
		_giDecorrelationFactor = s.factor;
		_giDecorrelationStagnancyExponent = s.stagnancyExponent;
		_giDecorrelationEmaFactor = s.emaFactor;
		_giDecorrelationFirefly = s.firefly;
		_giDecorrelationMultiplyBound = s.multiplyBound;
	}
	// RTXDI 3.1 DLSS-RR compatibility preset (the "Apply RR-style preset"
	// button values): decorrelates temporally-correlated input noise so the
	// RR network sees fresh samples instead of ReLAX-shaped history.
	void applyRrCompatPreset() noexcept {
		_giDecorrelationMode = 2u;             // Stagnancy
		_giDecorrelationFactor = 0.1f;
		_giDecorrelationStagnancyExponent = 1.0f;
		_giDecorrelationEmaFactor = 0.1f;
		_giDecorrelationFirefly = true;
	}

	float giMISRoughness()    const noexcept { return _giMISRoughness; }
	void  setGIMISRoughness(float v) noexcept { _giMISRoughness = v; }

	bool  deltaBranchNEEEnabled() const noexcept { return _deltaBranchNEEEnabled; }
	void  setDeltaBranchNEEEnabled(bool v) noexcept { _deltaBranchNEEEnabled = v; }

	// Shade-side mirror-NEE candidate count, INDEPENDENT of GI's kGiInitialCandidateCount
	// so _shadeShader DXC compile time stays tunable. M=1 = single-sample RIS (denoiser cleans up).
	// Public so the shade kernel in PipelineInit.cpp can reference it as a $for loop bound.
	static constexpr uint kShadeMirrorNEECandidateCount = 1u;

	// --- UI ---
	void drawUi();

	// --- Config serialization ---
	void toJson(ci::Json& j) const;
	void fromJson(const ci::Json& j);

private:
	// --- Compile helper ---
	// compileImpl: full compile body shared by compile() and recompileCallables().
	// When resolverOnly=true, skips the resolver-free sub-shaders (upsample, boiling)
	// since their IR is unchanged by a custom-callable DLL reload.
	void compileImpl(luisa::compute::Device& device,
	                 const render::SurfaceResolverPoly& resolver,
	                 bool resolverOnly);

	// --- Compiled shaders ---
	GIInitialShaderType         _giInitialShader;
	GIUpsampleShaderType        _giUpsampleShader;
	GITemporalReuseShaderType   _giTemporalReuseShader;
	GISpatialReuseShaderType    _giSpatialReuseShader;
	BoilingFilterGIType         _boilingFilterGI;
	ZeroGIReservoirShaderType   _zeroGIReservoirShader;
	GISmoothStagnancyShaderType _stagnancySmoothShader;
	ZeroFloatImageShaderType    _zeroFloatImageShader;

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

	// --- PassDI-shared presampled env tiles (non-owning pointer, wired by Pipeline) ---
	const luisa::compute::Buffer<render::PresampledCandidate>* _presampleEnvTilesPtr = nullptr;
	uint _presampleEnvTotalEntries = 0u;
public:
	void set_presample_env_tiles(
		const luisa::compute::Buffer<render::PresampledCandidate>& tiles) noexcept {
		_presampleEnvTilesPtr = &tiles;
		// Pre-compile wiring (Pipeline ctor) passes PassDI's not-yet-allocated
		// pool — Buffer::size() asserts on an invalid buffer (Debug abort,
		// crash 2026-09-19). buildScene re-wires with the real pool after
		// PassDI::compile; the stored pointer self-heals too (PassDI move-
		// assigns the member in place, so its address is stable).
		_presampleEnvTotalEntries = tiles ? static_cast<uint>(tiles.size()) : 0u;
	}

#if NT_ENABLE_SHARC
	// --- SHARC query hook (plan §7 Phase 2) ---
	// Non-owning cache bindings (wired by Pipeline right after PassSharc's
	// createResources) + the per-frame effective enable (Pipeline passes
	// PassSharc's enabled() so a disabled cache never serves stale queries).
private:
	const luisa::compute::Buffer<SharcKeyHost>* _sharcEntriesPtr = nullptr;
	const luisa::compute::Buffer<render::SharcPackedData>* _sharcResolvedPtr = nullptr;
	const luisa::compute::Buffer<SharcParams>* _sharcParamsPtr = nullptr;
	const luisa::compute::Buffer<luisa::uint>* _sharcQueryStatsPtr = nullptr;
	bool _sharcQueryEnabled = false;
public:
	void set_sharc_cache(
		const luisa::compute::Buffer<SharcKeyHost>& entries,
		const luisa::compute::Buffer<render::SharcPackedData>& resolved,
		const luisa::compute::Buffer<SharcParams>& params,
		const luisa::compute::Buffer<luisa::uint>& query_stats) noexcept {
		_sharcEntriesPtr = &entries;
		_sharcResolvedPtr = &resolved;
		_sharcParamsPtr = &params;
		_sharcQueryStatsPtr = &query_stats;
	}
	void setSharcQueryEnabled(bool v) noexcept { _sharcQueryEnabled = v; }
	[[nodiscard]] bool sharcQueryEnabled() const noexcept { return _sharcQueryEnabled; }
	// (leave visibility public: the FPS-aware accessors below were public
	// before this block and must stay that way)
#endif

	// FPS-aware accumulation: exposes state for PassDI visMaxAge coordination
	// (PassDI's visMaxAge is derived from _giAccumulationTime when enabled).
	bool giAccumulationTimeEnabled() const noexcept { return _giAccumulationTimeEnabled; }
	float giAccumulationTime() const noexcept { return _giAccumulationTime; }
private:

	// --- Runtime-tunable params buffer (uploaded once per frame) ---
	luisa::compute::Buffer<GIParams> _giParamsBuf;
	GIParams _giParamsCpu{};  // member so the pointer outlives cmdlist submission
	void _populateGiParams(luisa::compute::CommandList& cmdlist) noexcept;

	// --- GI reservoirs (full-res, ping-pong) ---
	std::array<luisa::compute::Buffer<GIReservoir>, 2u> _giResBuf;
	uint _giResIdx = 0u;

	// --- GI reservoirs (half-res pair, ping-ponged when _giHalfRes=true) ---
	std::array<luisa::compute::Buffer<GIReservoir>, 2u> _giResBufHalf;
	uint _giResIdxHalf = 0u;
	luisa::compute::Buffer<GIReservoir> _giReservoirBufferFull;

	// --- Frozen snapshot of initial GI reservoirs (post-renderInitial, pre-renderTemporal) ---
	// Read by shade shader for RTXDI-style initial-vs-final MIS at final shade.
	luisa::compute::Buffer<GIReservoir> _giInitialSnapshot;

	// --- Stagnancy decorrelation state (RTXDI 3.1) ---
	// Smoothed-stagnancy ping-pong pair + boiling-filter firefly flag, both at
	// the shade (checkerboard-compacted) resolution. _giStagnancyIdx points at
	// the most recently WRITTEN slot after each renderStagnancySmooth call.
	std::array<luisa::compute::Image<float>, 2u> _giStagnancy;
	luisa::compute::Image<float> _giFireflyFlag;
	uint _giStagnancyIdx = 0u;
	uint _stagnancyWidth = 0u;   // compacted dims (for the zero-fill dispatch)
	uint _stagnancyHeight = 0u;

	// --- Half-res dimensions ---
	uint _halfWidth = 0;
	uint _halfHeight = 0;

	// --- Light sampling mode (must match Pipeline) ---
	static constexpr bool kUniformLightSampling = false;

	// --- GI tunable parameters (captured at shader compile time) ---
	static constexpr uint kGiSpatialNeighborCount = 4u;
	uint  _giSpatialRadius        = 30u;
	float _giWSumCap              = 1e4f;
	uint  _giMaxAge               = 30u;
	uint  _giTemporalMaxM         = 8u;
	uint  _giSpatialMaxM          = 1024u;
	float _giMaxRadiance          = 10.0f;
	float _giRoughnessInvalidationThresh = 0.02f;  // Gap 1: reject temporal/spatial merges when |stored_rough - receiver_rough| exceeds thresh (0.5 = gate disabled)
	float _giMatSimRoughness  = 0.25f;  // RTXDI default (relative) for material similarity
	float _giMatSimF0         = 0.5f;  // RTXDI default (absolute luminance delta)
	float _giMatSimAlbedo     = 0.5f;  // RTXDI default (absolute luminance delta)
	static constexpr uint kGiGlassInitialSamples = 2u;
	static constexpr uint kGiInitialCandidateCount    = 1u;  // NEE candidates at x2 (triangle lights, alias-table sampled)
	static constexpr uint kGiInitialEnvCandidateCount = 0u;  // env candidates at x2 (disabled by default — open scenes produce unwanted env GI on surfaces expected dark; compile-time only, not JSON-tunable)
	float _giMISRoughness          = 0.3f;   // RTXDI kMISRoughness: floor for roughened-BRDF MIS at shade (0=off)
	bool  _deltaBranchNEEEnabled   = false;  // Shade-side delta branch: run NEE at mirror hit x2 for r<kMinRoughness metals (default OFF = current behavior)
	bool  _giSuppressAwayFromLight = false;  // DEBUG: skip GI on surfaces facing away from the first light (leak hypothesis test)
	bool  _giTier3BiasCorrectionEnabled = false;  // Tier 3 BASIC piSum MIS normalization in temporal+spatial reuse (default OFF until verified)
	float _giBiasCorrectionDivergenceThresh = 2.0f;  // Skip Tier 3 when pdfs similar (0=always apply, try 2.0 to skip wall-to-wall diffuse)
	float _giSuppressAwayFromLightThresh = 0.0f; // DEBUG: dot(ns, light_dir) threshold for the above (lower = stricter, catches grazing)
	// Temporal freshness gate knee in log2 radiance-ratio units (docs/gi_temporal_dynamic_receiver_plan.md §3.2):
	// full merge weight up to 2^knee radiance divergence, zero at 2^(knee+2).
	// 5.0 default (32x knee / 128x reject), 1.0 = old curve, <= 0 disables the gate.
	float _giFreshnessKnee = 5.0f;
	// --- Stagnancy decorrelation (RTXDI 3.1 Decorrelation.hlsli adaptation) ---
	// Mode: 0=None, 1=Uniform, 2=Stagnancy. Defaults mirror RTXDI's RR-off
	// preset (None); the RR-style preset (Stagnancy, 0.1/1.0/0.1, firefly on)
	// is a UI button for the future DLSS-RR hookup + experimentation.
	uint  _giDecorrelationMode = 0u;
	float _giDecorrelationFactor = 0.05f;
	float _giDecorrelationStagnancyExponent = 1.5f;
	float _giDecorrelationEmaFactor = 0.2f;
	bool  _giDecorrelationFirefly = false;
	float _giDecorrelationMultiplyBound = 10.0f;
	// --- Shared edge-stopping thresholds (duplicated from Pipeline for compile-time capture) ---
	float _spatialNormalThresh    = 0.5f;
	float _spatialDepthThresh     = 0.1f;

	// --- FPS-aware accumulation (RTXDI FullSample pattern) ---
	// When enabled, _giMaxAge above is overwritten each frame from
	// giAccumulationTime * smoothedFps. Defaults OFF for bit-identical
	// fallback. visMaxAge in PassDI is also derived from this knob (with
	// a /15 ratio) via Pipeline coordination.
	bool  _giAccumulationTimeEnabled = false;
	float _giAccumulationTime        = 0.5f;  // seconds; 30 frames @ 60 fps

	// --- State ---
	bool _enabled     = true;
	bool _giOneBounce = false;
	bool _giHalfRes   = false;
	bool _checkerboard = true;

	// --- Compile-time specialization snapshot -------------------------------
	// oneBounce / giScale / hasTransparentShadowCasters / sharcQueryOn are
	// baked into the kernels at compileImpl time (former UInt args). compile()
	// snapshots the values it baked here; Pipeline::render() compares them
	// against the current desired values at the DLL-reload safe point and
	// calls recompileCallables() on a flip (same path as a callable DLL
	// hot-reload). Bit-identical by construction: the baked constant is the
	// exact value the removed runtime arg would have carried.
	bool _transparentShadowCasters = false;
	uint _bakedOneBounce = 0u;
	uint _bakedGiScale = 1u;
	uint _bakedTransparentShadowCasters = 0u;
	uint _bakedSharcQueryOn = 0u;

public:
	// Desired (current) specialization values vs the baked snapshot.
	[[nodiscard]] uint desiredSharcQueryOn() const noexcept {
	#if NT_ENABLE_SHARC
		return (_sharcQueryEnabled && _sharcEntriesPtr != nullptr &&
		        _sharcQueryStatsPtr != nullptr) ? 1u : 0u;
	#else
		return 0u;
	#endif
	}
	[[nodiscard]] bool specializationChanged(bool transparentShadowCasters) const noexcept {
		return _bakedOneBounce != (_giOneBounce ? 1u : 0u) ||
		       _bakedGiScale != (_giHalfRes ? 2u : 1u) ||
		       _bakedTransparentShadowCasters != (transparentShadowCasters ? 1u : 0u) ||
		       _bakedSharcQueryOn != desiredSharcQueryOn();
	}
};

} // namespace core
} // namespace newtype

// Register GIParams as a LuisaCompute DSL struct - must be outside namespace
LUISA_STRUCT(newtype::core::GIParams,
	giSpatialRadius,
	giSpatialNeighborCount,
	giWSumCap,
	giMaxAge,
	giTemporalMaxM,
	giSpatialMaxM,
	giMaxRadiance,
	giRoughnessInvalidationThresh,
	giMatSimRoughness,
	giMatSimF0,
	giMatSimAlbedo,
	giMISRoughness,
	giBiasCorrectionDivergenceThresh,
	spatialNormalThresh,
	spatialDepthThresh,
	giTier3BiasCorrectionEnabled,
	giSuppressAwayFromLight,
	giSuppressAwayFromLightThresh,
	giFreshnessKnee,
	giDecorrelationEmaFactor
) {};
