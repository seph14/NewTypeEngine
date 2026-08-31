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
};

//==========================================================================
// GI Shader Type Aliases
//==========================================================================

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
	luisa::uint,                                      // 17: oneBounce (1=skip 2nd bounce)
	luisa::uint,                                      // 18: giScale (1=full, 2=half-res)
	luisa::uint,                                      // 19: giSuppressAwayFromLight (1=suppress GI on surfaces facing away from primary light)
	float,                                           // 20: giSuppressAwayFromLightThresh (skip when dot(ns, light_dir) < thresh)
	luisa::uint                                      // 21: hasTransparentShadowCasters (1=transparent-aware closest-hit shadows; 0=any-hit fast path)
#if NT_ENABLE_PROCEDURAL
	,
	luisa::compute::BindlessArray                     // 22: procedural bindless (instances, positions, indices, AABBs, normals)
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
	luisa::uint,                                      // 15: tier3BiasCorrectionEnabled (1=Tier 3 piSum MIS, 0=legacy 1/M)
	luisa::uint                                       // 16: giScale (1=full, 2=half-res)
#if NT_ENABLE_PROCEDURAL
	,
	luisa::compute::BindlessArray                     // 17: procedural bindless (instances, positions, indices, AABBs, normals)
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
	luisa::uint                                       // 2: cbField (checkerboard, 0=off)
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
	             const render::SurfaceResolverPoly& resolver, bool checkerboard = false);
	// Recompile only the resolver-dependent sub-shaders (initial, temporal, spatial).
	// Resolver-free sub-shaders (upsample, boiling) are unaffected by custom-callable
	// DLL changes.
	void recompileCallables(luisa::compute::Device& device,
	                        const render::SurfaceResolverPoly& resolver);
	void createImages(luisa::compute::Device& device, uint width, uint height);
	void release();

	// --- Render passes (append commands to CommandList for batched submission) ---
	void renderInitial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	void renderTemporal(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
	void renderBoiling(luisa::compute::CommandList& cmdlist, const FrameContext& ctx, float strength);
	void renderSpatial(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
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
		_presampleEnvTotalEntries = static_cast<uint>(tiles.size());
	}

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
	static constexpr uint kGiInitialEnvCandidateCount = 0u;  // env candidates at x2 (disabled by default — open scenes produce unwanted env GI on surfaces expected dark; enable per-scene via JSON config)
	float _giMISRoughness          = 0.3f;   // RTXDI kMISRoughness: floor for roughened-BRDF MIS at shade (0=off)
	bool  _deltaBranchNEEEnabled   = false;  // Shade-side delta branch: run NEE at mirror hit x2 for r<kMinRoughness metals (default OFF = current behavior)
	bool  _giSuppressAwayFromLight = false;  // DEBUG: skip GI on surfaces facing away from the first light (leak hypothesis test)
	bool  _giTier3BiasCorrectionEnabled = false;  // Tier 3 BASIC piSum MIS normalization in temporal+spatial reuse (default OFF until verified)
	float _giBiasCorrectionDivergenceThresh = 2.0f;  // Skip Tier 3 when pdfs similar (0=always apply, try 2.0 to skip wall-to-wall diffuse)
	float _giSuppressAwayFromLightThresh = 0.0f; // DEBUG: dot(ns, light_dir) threshold for the above (lower = stricter, catches grazing)
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
	giSuppressAwayFromLightThresh
) {};
