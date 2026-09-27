#pragma once

#include "newtype/core/Config.h"

#include <span>
#include <luisa/luisa-compute.h>

#if NT_ENABLE_EDITOR

#include "newtype/core/IFeature.h"

namespace newtype::feature {

// Editor gizmo overlay — world-space debug drawing composited onto the final
// (tone-mapped) image. Singleton handle (ShaderManager/Profiler idiom):
// components push world-space primitives via Gizmo::get(); the companion
// GizmoFeature drains the stack at FeaturePoint::AfterToneMap, projecting
// with the frame's own unjittered camera and compositing in-place onto the
// display target. Pushes are CPU-side vector appends; the GPU dispatch is
// skipped entirely when the stack is empty, so the overlay costs nothing
// when unused.
//
// The unified GPU primitive is a capsule (segment + half-width in pixels):
// points are zero-length capsules, world spheres become discs with projected
// radius, wire cubes are 12 capsules — one buffer, one kernel loop.
class Gizmo {
public:
    // CPU-side draw stack. Deliberately trivial (vector + flags) so the
    // singleton's static destruction cannot touch the Luisa device; the
    // device buffer lives in GizmoFeature and dies with the pipeline.
    struct State {
        enum class Kind : uint8_t { CapsulePx, DiscWorldRadius };
        struct Prim {
            luisa::float3 a;       // world-space endpoints
            luisa::float3 b;       // (unused for DiscWorldRadius)
            luisa::float4 rgba;
            float size;            // CapsulePx: half-width px; DiscWorldRadius: world radius
            Kind kind;
        };
        std::vector<Prim> prims;
        bool enabled = true;
        uint framePrimCount = 0u; // drained prim count, for drawUi stats
    };

    static Gizmo& get() noexcept;

    void drawLine(luisa::float3 a, luisa::float3 b, luisa::float3 rgb,
                  float alpha = 1.f, float widthPx = 1.5f);
    void drawPolyline(std::span<const luisa::float3> pts, luisa::float3 rgb,
                      float alpha = 1.f, float widthPx = 1.5f, bool closed = false);
    void drawWireCube(luisa::float3 center, luisa::float3 halfSize, luisa::float3 rgb,
                      float alpha = 1.f, float widthPx = 1.5f);
    void drawPoint(luisa::float3 pos, luisa::float3 rgb,
                   float alpha = 1.f, float radiusPx = 4.f);
    void drawSphere(luisa::float3 center, float worldRadius, luisa::float3 rgb,
                    float alpha = 1.f);

    void setEnabled(bool enabled) noexcept { _state.enabled = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return _state.enabled; }

    // GizmoFeature access to the draw stack.
    [[nodiscard]] State& state() noexcept { return _state; }
    [[nodiscard]] const State& state() const noexcept { return _state; }

private:
    Gizmo() = default;
    State _state;
};

// AfterToneMap consumer: drains Gizmo::get()'s draw stack into the display
// target. Register AFTER FxaaFeature so gizmos composite on top of the AA'd
// image (features at one injection point run in registration order).
class GizmoFeature : public core::IFeature {
public:
    [[nodiscard]] core::FeaturePoint point() const noexcept override {
        return core::FeaturePoint::AfterToneMap;
    }
    void onInit(luisa::compute::Device& device) override;
    void onExecute(luisa::compute::Stream& stream, const core::FeatureContext& ctx) override;
    void drawUi() override;

    // The overlay must composite AFTER tonemap writes the display target —
    // never claim the temp route (that is FXAA's copy-skip, and this pass
    // does not implement the temp-read/write-display contract it implies).
    [[nodiscard]] bool wantsTonemapTempRoute(const core::Pipeline&) const noexcept override {
        return false;
    }

    /// The overlay projects via view_proj + tan(fov/2) — pinhole-only.
    [[nodiscard]] bool requiresPerspectiveProjection() const noexcept override {
        return true;
    }

    void load(const nlohmann::json& file) override;
    [[nodiscard]] nlohmann::json toJson() const override;

    using ShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,          // 0: display target (in-place read-modify-write)
        luisa::compute::Buffer<luisa::float4>, // 1: capsules, stride 3 float4s
        luisa::compute::uint                   // 2: capsule count
    >;

private:
    ShaderType _shader;
    luisa::compute::Buffer<luisa::float4> _primBuffer;
    uint _capacityFloat4 = 0u; // capacity of _primBuffer in float4s
    bool _compiled = false;
};

} // namespace newtype::feature

#else // !NT_ENABLE_EDITOR

// No-op production fallback: same push API, everything compiles away. Call
// sites (PathHelper & co.) stay untouched when NT_ENABLE_EDITOR is 0.
namespace newtype::feature {

class Gizmo {
public:
    static Gizmo& get() noexcept {
        static Gizmo g;
        return g;
    }
    void drawLine(luisa::float3, luisa::float3, luisa::float3,
                  float = 1.f, float = 1.5f) noexcept {}
    void drawPolyline(std::span<const luisa::float3>, luisa::float3,
                      float = 1.f, float = 1.5f, bool = false) noexcept {}
    void drawWireCube(luisa::float3, luisa::float3, luisa::float3,
                      float = 1.f, float = 1.5f) noexcept {}
    void drawPoint(luisa::float3, luisa::float3,
                   float = 1.f, float = 4.f) noexcept {}
    void drawSphere(luisa::float3, float, luisa::float3,
                    float = 1.f) noexcept {}

    void setEnabled(bool) noexcept {}
    [[nodiscard]] bool enabled() const noexcept { return false; }
};

} // namespace newtype::feature

#endif // NT_ENABLE_EDITOR
