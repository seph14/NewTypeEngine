#pragma once

#include <algorithm>

#include <luisa/luisa-compute.h>
#include "newtype/core/Config.h"
#include "newtype/render/Sharc.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/ReSTIR.h"
#include "newtype/render/SurfaceResolver.h"
#include "newtype/render/LightSampler.h"
#include "newtype/core/BindingGroups.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/Camera.h"
#include "newtype/core/FrameContext.h"

namespace newtype {
namespace core {

using namespace newtype::render;

//==============================================================================
// SharcParams — runtime-tunable SHARC parameters uploaded per frame
//==============================================================================
// PRIMITIVES ONLY — LUISA_STRUCT rejects DSL wrapper types. Mirrors the
// SSSParams staging-buffer pattern (member struct, deferred copy_from).
struct SharcParams {
    float sceneScale;            // voxel = base^level / sceneScale (≈ scene extent)
    float levelBias;             // LOD bias (0 = off)
    float radianceScale;         // atomic accumulation quantization (1e3)
    uint  accumulationFrameNum;  // EWMA window (default 32)
    uint  staleFrameNumMax;      // eviction threshold (default 64)
    float roughnessMin;          // clamp at update-path hits (0.4; SHARC gate)
    float maxRadiance;           // firefly clamp on inserted direct lighting
    uint  updateBounces;         // max path length in the update pass (3)
    uint  downscaleFactor;       // one active pixel per NxN tile (5)
    uint  capacity;              // entriesNum power-of-two (set by createResources)
};

//==============================================================================
// Shader type aliases
//==============================================================================
// The lock buffer is ALWAYS allocated and bound (4 MiB at 2^20 entries) so a
// single signature serves both insert routes; the kernel builds the matching
// cache bundle via if-constexpr on Sharc.h's config aliases.
using SharcKeyHost = SharcEngineLayout::KeyHost;

// Phase 3 gather constants (plan §7): roughness epsilon classifying a pixel
// as rough glass ("~0.05"; the validation scene's smoothest sphere sits at
// 0.05 so the epsilon stays just below it), and the per-tap trace-leg budget
// for the simplified smooth-glass crossing replay inside a gather tap.
inline constexpr float kRoughGlassEpsilon = 0.04f;
inline constexpr uint  kGatherGlassLegs   = 4u;

// Per-tap outlier ceiling (speckle fix, Phase B): every gather tap value —
// cache hit or env — is clamped to this × the luminance of the pixel's stable
// reference (transmission: the denoised PSR background; reflection: the
// mirror-env fallback), chroma preserved. Emitter-adjacent voxels and
// escaped-ray sun texels can't re-introduce fireflies. Constant in v1 (no
// slider); 25× is loose enough that legitimate bright reflections (bright
// ground under area lights, ~5–10 radiance vs ~1–5 mirror sky) rarely bind.
inline constexpr float kGatherTapClamp = 25.0f;

// Update: sparse (1 pixel per NxN tile) multi-bounce paths feed the cache.
// 64-bit keys + CAS route need SM6.6 — the kernel calls set_warp_size(32).
using SharcUpdateShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<SharcParams>,       // 0: params
    luisa::compute::Buffer<SharcKeyHost>,      // 1: hash entries
    luisa::compute::Buffer<luisa::uint>,       // 2: locks (unused by CAS route)
    luisa::compute::Buffer<luisa::uint4>,      // 3: accumulation
    luisa::compute::Buffer<SharcPackedData>,   // 4: resolved
    luisa::compute::Image<uint>,               // 5: seed image (INT1)
    luisa::compute::uint,                      // 6: frame_count
    luisa::compute::Accel,                     // 7: TLAS
    newtype::util::CameraData,                 // 8: camera
    SceneGeometryResources,                    // 9: instance/transform/material buffers
    luisa::compute::BindlessArray,             // 10: vertex_bindless
    luisa::compute::BindlessArray,             // 11: tex_bindless
    LightSamplingResources,                    // 12: triangle lights + alias table
    EnvLightResources,                         // 13: envmap + CDFs + rotation
    float,                                     // 14: env_exposure
    luisa::compute::Buffer<render::PresampledCandidate>, // 15: presampled env tiles
    luisa::compute::uint,                      // 16: presample env total entries
    luisa::compute::uint,                      // 17: hasTransparentShadowCasters
    luisa::compute::uint,                      // 18: resolution x (tile-sparse dispatch)
    luisa::compute::uint                       // 19: resolution y
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray              // 20: procedural bindless
#endif
>;

// Resolve: one thread per entry (EWMA + eviction + adjacent-level blend).
using SharcResolveShaderType = luisa::compute::Shader<1,
    luisa::compute::Buffer<SharcParams>,       // 0: params
    luisa::compute::Buffer<SharcKeyHost>,      // 1: hash entries
    luisa::compute::Buffer<luisa::uint>,       // 2: locks (unused by CAS route)
    luisa::compute::Buffer<luisa::uint4>,      // 3: accumulation
    luisa::compute::Buffer<SharcPackedData>,   // 4: resolved
    newtype::util::CameraData                  // 5: camera (prev position)
>;

// Clear: reset all three (four with locks) buffers to empty-entry state.
using SharcClearShaderType = luisa::compute::Shader<1,
    luisa::compute::Buffer<SharcKeyHost>,      // 0: hash entries
    luisa::compute::Buffer<luisa::uint>,       // 1: locks
    luisa::compute::Buffer<luisa::uint4>,      // 2: accumulation
    luisa::compute::Buffer<SharcPackedData>,   // 3: resolved
    luisa::uint                                // 4: capacity
>;

// Occupancy counter: one thread per entry, atomic count of valid keys.
using SharcCountShaderType = luisa::compute::Shader<1,
    luisa::compute::Buffer<SharcKeyHost>,      // 0: hash entries
    luisa::compute::Buffer<luisa::uint>,       // 1: stats (1 uint)
    luisa::uint                                // 2: capacity
>;

// Debug view: colored hash / bucket occupancy of the primary surface.
#if NT_DEBUG_VIZ
using SharcDebugShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<SharcParams>,       // 0: params
    luisa::compute::Buffer<SharcKeyHost>,      // 1: hash entries
    luisa::compute::Image<float>,              // 2: output target
    luisa::compute::Image<float>,              // 3: gbuf_depth
    luisa::compute::Image<uint>,               // 4: gbuf_vis
    luisa::compute::Image<float>,              // 5: gbuf_bary_motion
    newtype::util::CameraData,                 // 6: camera
    SceneGeometryResources,                    // 7: scene buffers
    luisa::compute::BindlessArray,             // 8: vertex_bindless
    luisa::compute::BindlessArray,             // 9: tex_bindless
    luisa::uint                                // 10: mode (1=colored hash, 2=occupancy)
>;
#endif

//==============================================================================
// PassSharc — SHARC radiance cache pass (Phase 1: Update + Resolve; Phase 2
// adds the read-only query bindings consumed by PassGI initial)
//==============================================================================
// Docs/sharc_rough_glass_plan.md §7: builds and maintains the world-space
// radiance cache (Update + Resolve, dispatched after G-buffer + presample,
// before GI, on the same stream — auto UAV barriers order Update -> Resolve);
// exposes the cache + grid params so the GI initial pass can query it at its
// x2 hits (Phase 2).
class PassSharc {
public:
    // Debug view modes (drawUi-exposed; Off renders normally)
    enum class DebugView : int { Off = 0, ColoredHash = 1, Occupancy = 2 };

    explicit PassSharc() noexcept = default;
    PassSharc(PassSharc&&) noexcept = default;
    PassSharc(const PassSharc&) = delete;
    PassSharc& operator=(const PassSharc&) = delete;
    PassSharc& operator=(PassSharc&&) = delete;
    ~PassSharc() noexcept = default;

    void compile(luisa::compute::Device& device,
                 newtype::scene::Geometry& geom,
                 const SurfaceResolverPoly& resolver);
    void createResources(luisa::compute::Device& device,
                         uint entriesNum = render::kSharcDefaultEntriesNum);
    void release();

    // Update: sparse path-traced inserts. Resets the cache (clear dispatch)
    // when ctx.accumReset or on the first call after createResources.
    void renderUpdate(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
    // Resolve: EWMA + eviction, one thread per entry.
    void renderResolve(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);
#if NT_DEBUG_VIZ
    // Debug blit (ColoredHash / Occupancy) to `target`.
    void renderDebug(luisa::compute::CommandList& cmdlist, const FrameContext& ctx,
                     luisa::compute::Image<float>& target);
#endif

    void requestReset() noexcept { _needsReset = true; }

    // Occupancy + query-stat readback, two phases (no stream synchronize):
    //  - submitPolls enqueues the counter dispatch + readbacks at the END of
    //    the frame (after the tail work) and signals a timeline event.
    //  - readPolls consumes the results at the next cadence tick (>= 1 frame
    //    later) via a host-side event wait that is already satisfied in
    //    practice. Replaces the former mid-frame drawUi polls that ended in
    //    `stream << synchronize()`. Query stats always ride along (the query
    //    consumers are live whenever the cache is enabled).
    void submitPolls(luisa::compute::Stream& stream);
    void readPolls();
    [[nodiscard]] uint entryCount() const noexcept { return _entryCountHost; }
    [[nodiscard]] uint capacity() const noexcept { return _entriesNum; }
    [[nodiscard]] float occupancy() const noexcept {
        return _entriesNum ? static_cast<float>(_entryCountHost) / static_cast<float>(_entriesNum) : 0.0f;
    }

    [[nodiscard]] bool enabled() const noexcept { return _enabled; }
    void setEnabled(bool v) noexcept { _enabled = v; }
    [[nodiscard]] DebugView debugView() const noexcept { return _debugView; }
    void setDebugView(DebugView v) noexcept { _debugView = v; }

    // Rough-glass gather + smooth dispersive replay follow enabled() (the
    // Phase-2/Phase-3 A/B toggles were removed 2026-09-09 after the soak
    // validated the shipping path — same discipline as the GI temporal
    // RTXDI-gates flag, docs/gi_temporal_dynamic_receiver_plan.md §3.4).

    // Gather tuning (plan §8: tap budget K decided in Phase 3 by measuring;
    // temporal accumulation IN for v1). Counters and blend are read by the
    // glass-tint dispatch each frame.
    [[nodiscard]] uint gatherTransTaps() const noexcept { return _gatherTransTaps; }
    void setGatherTransTaps(uint v) noexcept { _gatherTransTaps = std::clamp(v, 1u, 8u); }
    [[nodiscard]] uint gatherReflTaps() const noexcept { return _gatherReflTaps; }
    void setGatherReflTaps(uint v) noexcept { _gatherReflTaps = std::clamp(v, 0u, 8u); }
    [[nodiscard]] float gatherTemporalAlpha() const noexcept { return _gatherTemporalAlpha; }
    void setGatherTemporalAlpha(float v) noexcept { _gatherTemporalAlpha = std::clamp(v, 0.02f, 1.0f); }

    // Read-only cache bindings for query consumers (non-owning; valid after
    // createResources until release). The params buffer carries the SAME
    // grid parameters as Update/Resolve — query keys must match theirs.
    [[nodiscard]] const luisa::compute::Buffer<SharcKeyHost>& entriesBuffer() const noexcept { return _entriesBuf; }
    [[nodiscard]] const luisa::compute::Buffer<SharcPackedData>& resolvedBuffer() const noexcept { return _resolvedBuf; }
    [[nodiscard]] const luisa::compute::Buffer<SharcParams>& paramsBuffer() const noexcept { return _paramsBuf; }
    // Query diagnostics written by GI initial ([0]=hits, [1]=attempted misses)
    // and the Phase-3 rough-glass gather ([2]=hits, [3]=attempted misses).
    [[nodiscard]] const luisa::compute::Buffer<luisa::uint>& queryStatsBuffer() const noexcept { return _queryStatsBuf; }

    // Query hit-rate readout: dispatch-free readback of the counters GI
    // initial incremented since the last poll. Window = the poll interval.
    // Rides submitPolls/readPolls (includeQueryStats gate).
    [[nodiscard]] uint queryHits() const noexcept { return _queryHitsHost; }
    [[nodiscard]] uint queryMisses() const noexcept { return _queryMissesHost; }
    [[nodiscard]] uint queryAttempts() const noexcept { return _queryHitsHost + _queryMissesHost; }
    [[nodiscard]] float queryHitRate() const noexcept {
        return queryAttempts() ?
            static_cast<float>(_queryHitsHost) / static_cast<float>(queryAttempts()) : 0.0f;
    }

    // Phase-3 gather query counters (same poll window as the GI counters).
    [[nodiscard]] uint gatherHits() const noexcept { return _gatherHitsHost; }
    [[nodiscard]] uint gatherMisses() const noexcept { return _gatherMissesHost; }
    [[nodiscard]] uint gatherAttempts() const noexcept { return _gatherHitsHost + _gatherMissesHost; }
    [[nodiscard]] float gatherHitRate() const noexcept {
        return gatherAttempts() ?
            static_cast<float>(_gatherHitsHost) / static_cast<float>(gatherAttempts()) : 0.0f;
    }

    // Reflection taps that resolved to the fallback without attempting a
    // lookup (escape / glass / gate-reject — the speckle-fix blind spot the
    // hit-rate readout missed). Denominator for the fallback share = attempted
    // lookups + these; transmission pre-lookup fallbacks (miss / TIR / leg
    // budget) are uncounted, so the % is a reflection-leaning diagnostic.
    [[nodiscard]] uint gatherFallbackTaps() const noexcept { return _gatherFallbackHost; }
    [[nodiscard]] float gatherFallbackShare() const noexcept {
        uint counted = gatherAttempts() + _gatherFallbackHost;
        return counted ? static_cast<float>(_gatherFallbackHost) / static_cast<float>(counted) : 0.0f;
    }

    void drawUi();

private:
    scene::Geometry*            _geom = nullptr;
    uint                        _entriesNum = 0u;

    SharcUpdateShaderType       _updateShader;
    SharcResolveShaderType      _resolveShader;
    SharcClearShaderType        _clearShader;
    SharcCountShaderType        _countShader;
#if NT_DEBUG_VIZ
    SharcDebugShaderType        _debugShader;
#endif

    luisa::compute::Buffer<SharcKeyHost>    _entriesBuf;
    luisa::compute::Buffer<luisa::uint>     _locksBuf;
    luisa::compute::Buffer<luisa::uint4>    _accumBuf;
    luisa::compute::Buffer<SharcPackedData> _resolvedBuf;
    luisa::compute::Buffer<luisa::uint>     _statsBuf;
    uint                          _entryCountHost = 0u;
    luisa::compute::Buffer<luisa::uint>     _queryStatsBuf;  // [0]=GI hits, [1]=GI misses, [2]=gather hits, [3]=gather misses, [4]=gather refl fallback taps
    uint                          _queryStatsHost[5] = {0u, 0u, 0u, 0u, 0u}; // readback staging (member: outlives GPU-side copy_to)
    uint                          _queryHitsHost = 0u;
    uint                          _queryMissesHost = 0u;
    uint                          _gatherHitsHost = 0u;
    uint                          _gatherMissesHost = 0u;
    uint                          _gatherFallbackHost = 0u;

    // Poll completion event: signaled by submitPolls on the render stream,
    // host-waited by readPolls at the next cadence tick.
    luisa::compute::TimelineEvent _pollEvent;
    uint64_t                      _pollFence = 0u;
    bool                          _pollQueued = false;

    luisa::compute::Buffer<SharcParams> _paramsBuf;
    SharcParams                 _paramsCpu{};       // CLASS MEMBER (cmdlist deferred)
    bool                        _paramDirty = true;
    bool                        _needsReset = true;

    // PassDI-shared presampled env tiles (non-owning, wired by Pipeline)
    const luisa::compute::Buffer<render::PresampledCandidate>* _presampleEnvTilesPtr = nullptr;
    uint _presampleEnvTotalEntries = 0u;

public:
    void set_presample_env_tiles(
        const luisa::compute::Buffer<render::PresampledCandidate>& tiles) noexcept {
        _presampleEnvTilesPtr = &tiles;
        // Pre-compile wiring (Pipeline ctor) passes PassDI's not-yet-allocated
        // pool — Buffer::size() asserts on an invalid buffer (Debug abort,
        // crash 2026-09-19). buildScene re-wires with the real pool after
        // PassDI::compile (see PipelineInit.cpp presample-pool comment).
        _presampleEnvTotalEntries = tiles ? static_cast<uint>(tiles.size()) : 0u;
    }

private:

    bool      _enabled = true;
    // K: GGX transmission taps. The allocation is the deterministic
    // (k+frame)%3 channel pick in the tint shader — no i.i.d. channel strobe
    // (the old picks left ~58% of frames with an empty channel, a
    // 3×-amplitude chroma strobe on high-contrast content). Default 4
    // (2026-09-06, measured: K=6 = exactly 2 taps/channel showed no
    // meaningful visual gain over 4's balanced-±1 2/1/1 rotation, and the
    // +2 taps cost rough-glass-pixel time); K=6 remains a UI option — the
    // shader code is K-agnostic (docs/dispersion_speckle_fix_plan.md §2).
    uint      _gatherTransTaps = 4u;
    uint      _gatherReflTaps = 2u;      // K': GGX reflection taps
    // Gather history MINIMUM blend alpha: the shader blends at
    // max(alphaMin, 1/history_age) — a running mean that converges, floored
    // here to bound ghosting/lag (2026-09-03; the original fixed 0.2 EWMA
    // left a permanently boiling ~1/3-of-per-frame-variance residue).
    float     _gatherTemporalAlpha = 0.05f;
    DebugView _debugView = DebugView::Off;

    void _populateParams(luisa::compute::CommandList& cmdlist) noexcept;
};

} // namespace core
} // namespace newtype

// Register SharcParams as a LuisaCompute DSL struct - must be at global scope.
LUISA_STRUCT(newtype::core::SharcParams,
    sceneScale,
    levelBias,
    radianceScale,
    accumulationFrameNum,
    staleFrameNumMax,
    roughnessMin,
    maxRadiance,
    updateBounces,
    downscaleFactor,
    capacity
) {};
