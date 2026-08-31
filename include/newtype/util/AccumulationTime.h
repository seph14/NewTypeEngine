#pragma once

// ============================================================================
// AccumulationTime — FPS-aware temporal-accumulation math helpers.
// ============================================================================
// Mirrors the NVIDIA RTXDI FullSample pattern
// (RTXDI/Samples/FullSample/Source/UserInterface.cpp:1469-1481):
//
//   accumulationTime (seconds) is the user-facing knob.
//   Each frame, maxAccumulatedFrameNum is derived from live FPS so the
//   wall-clock temporal-blur tau stays constant across framerates.
//
// Used by:
//   - newtype::render::RelaxDenoiser  (diffuse/specular slow+fast EMAs)
//   - newtype::render::PassGI         (giMaxAge)
//   - newtype::render::PassDI         (visMaxAge, derived from GI knob)
//
// Call sites fetch FPS via ci::app::getWindow()->getApp()->getAverageFps()
// (Cinder's built-in EMA, already used at util/Profiler.cpp:91), then pass
// it through `clampSmoothedFps` to bound pathological values during
// loading/hitches.

#include <algorithm>
#include <cstdint>

namespace newtype::util {

// Clamp raw FPS to a safe window. Lower bound 15 fps guards against
// hitches producing absurd accumulation lengths; upper bound 240 fps
// guards against empty-frame bursts after loading.
inline float clampSmoothedFps(float rawFps) noexcept {
    return std::clamp(rawFps, 15.0f, 240.0f);
}

// RTXDI-canonical: maxAccumulatedFrameNum = min( accumulationTime * fps + 0.5, 63 )
inline uint32_t computeAccumulatedFrames(float accumulationTimeSec, float fps) noexcept {
    float f = accumulationTimeSec * fps + 0.5f;
    f = std::clamp(f, 0.0f, 63.0f);
    return static_cast<uint32_t>(f);
}

// RTXDI-canonical: maxFastAccumulatedFrameNum = max( slow / 6, 2 )
inline uint32_t computeFastAccumulatedFrames(uint32_t slow) noexcept {
    return std::max(slow / 6u, 2u);
}

} // namespace newtype::util
