#include "newtype/feature/Gizmo.h"

#if NT_ENABLE_EDITOR

#include <algorithm>
#include <cmath>
#include <vector>

#include <luisa/dsl/sugar.h>

#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Camera.h"
#include "cinder/CinderImGui.h"

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

namespace {

// Soft cap so a runaway pusher (feature disabled mid-session, editor flag
// mismatch) can never grow the stack unbounded between drains.
constexpr size_t kMaxPrims = 1u << 16;

// 12 cube edges over the 8 corners bit-packed as x=bit0, y=bit1, z=bit2.
constexpr uint kCubeEdges[12][2] = {
    {0u, 1u}, {2u, 3u}, {4u, 5u}, {6u, 7u}, // along x
    {0u, 2u}, {1u, 3u}, {4u, 6u}, {5u, 7u}, // along y
    {0u, 4u}, {1u, 5u}, {2u, 6u}, {3u, 7u}, // along z
};

} // namespace

//==========================================================================
// Gizmo (singleton handle)
//==========================================================================

Gizmo& Gizmo::get() noexcept {
    static Gizmo g;
    return g;
}

void Gizmo::drawLine(float3 a, float3 b, float3 rgb, float alpha, float widthPx) {
    if (!_state.enabled || _state.prims.size() >= kMaxPrims) return;
    _state.prims.push_back(
        {a, b, float4{rgb.x, rgb.y, rgb.z, alpha}, widthPx * 0.5f, State::Kind::CapsulePx});
}

void Gizmo::drawPolyline(std::span<const float3> pts, float3 rgb,
                         float alpha, float widthPx, bool closed) {
    if (!_state.enabled || pts.size() < 2u) return;
    for (size_t i = 0u; i + 1u < pts.size(); ++i)
        drawLine(pts[i], pts[i + 1u], rgb, alpha, widthPx);
    if (closed) drawLine(pts.back(), pts.front(), rgb, alpha, widthPx);
}

void Gizmo::drawWireCube(float3 center, float3 halfSize, float3 rgb,
                         float alpha, float widthPx) {
    if (!_state.enabled) return;
    float3 v[8];
    for (uint i = 0u; i < 8u; ++i) {
        auto sx = (i & 1u) ? 1.f : -1.f;
        auto sy = (i & 2u) ? 1.f : -1.f;
        auto sz = (i & 4u) ? 1.f : -1.f;
        v[i] = center + float3{sx, sy, sz} * halfSize;
    }
    for (const auto& e : kCubeEdges)
        drawLine(v[e[0]], v[e[1]], rgb, alpha, widthPx);
}

void Gizmo::drawPoint(float3 pos, float3 rgb, float alpha, float radiusPx) {
    // Zero-length capsule == screen-space disc of the given radius.
    drawLine(pos, pos, rgb, alpha, radiusPx * 2.f);
}

void Gizmo::drawSphere(float3 center, float worldRadius, float3 rgb, float alpha) {
    if (!_state.enabled || _state.prims.size() >= kMaxPrims) return;
    _state.prims.push_back(
        {center, center, float4{rgb.x, rgb.y, rgb.z, alpha}, worldRadius, State::Kind::DiscWorldRadius});
}

//==========================================================================
// GizmoFeature — AfterToneMap drain
//==========================================================================

void GizmoFeature::onInit(Device& device) {
    auto kernel = Kernel2D([&](
        ImageFloat target, BufferVar<float4> prims, UInt capsuleCount
        ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 res = dispatch_size().xy();
        $if(any(coord >= res)) { $return(); };

        // Capsules are in image-pixel space, row 0 = image bottom, y up —
        // the exact inverse of the G-buffer pass's pixel→NDC mapping, so no
        // flip is needed against the write target.
        Float2 px = make_float2(coord) + 0.5f;
        Float4 acc = target.read(coord);

        $for(c, 0u, capsuleCount) {
            UInt i = c * 3u;
            Float4 ab  = prims->read(i);
            Float4 col = prims->read(i + 1u);
            Float  hw  = prims->read(i + 2u).x;
            Float2 a = ab.xy();
            Float2 b = ab.zw();
            // Expanded-AABB early-out (half width + 1px AA skirt).
            Float2 mn = min(a, b) - make_float2(hw + 1.0f);
            Float2 mx = max(a, b) + make_float2(hw + 1.0f);
            $if(all(px >= mn) & all(px <= mx)) {
                // Point-to-segment distance; degenerate a==b is a disc.
                Float2 abv = b - a;
                Float t = clamp(dot(px - a, abv) / max(dot(abv, abv), 1e-8f), 0.0f, 1.0f);
                Float d = length(px - (a + t * abv)) - hw;
                Float cov = 1.0f - smoothstep(0.0f, 1.0f, d);
                acc = make_float4(lerp(acc.xyz(), col.xyz(), cov * col.w), acc.w);
            };
        };

        target.write(coord, acc);
    });

    _shader = device.compile(kernel);
    _compiled = true;
}

void GizmoFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    auto& state = Gizmo::get().state();

    // Drain discipline: the stack never survives a frame, dispatch or not —
    // pushers run every frame between beginUiFrame and this point.
    struct StackDrain {
        Gizmo::State& s;
        ~StackDrain() {
            s.framePrimCount = static_cast<uint>(s.prims.size());
            s.prims.clear();
        }
    } drain{state};

    if (!_compiled || !state.enabled || state.prims.empty()) return;
    // Post-tonemap contract (same guard as FXAA): HDR float display never
    // tonemaps, and the DX12 present path is RGBA8-only anyway.
    if (!ctx.pipeline.lastFrameTonemapped()) return;

    // Project world → image-pixel space with the frame camera. view_proj is
    // unjittered (jitter lives only in generate_ray), so gizmos stay stable.
    const auto& cam = ctx.frame.camera;
    const float W = static_cast<float>(ctx.width());
    const float H = static_cast<float>(ctx.height());
    const float halfTan = std::tan(cam.fov * 0.5f * 3.14159265358979f / 180.f);

    std::vector<float4> data;
    data.reserve(state.prims.size() * 3u);
    for (const auto& prim : state.prims) {
        float4 clipA = cam.view_proj * make_float4(prim.a, 1.f);
        if (clipA.w <= 1e-4f) continue; // behind the camera
        float2 pa = (clipA.xy() / clipA.w * 0.5f + 0.5f) * float2{W, H};
        float2 pb = pa;
        float hw = prim.size;
        if (prim.kind == Gizmo::State::Kind::DiscWorldRadius) {
            // World radius → pixel radius via the view-space depth.
            hw = clamp(prim.size / clipA.w * (H * 0.5f) / halfTan, 1.f, 8192.f);
        } else {
            float4 clipB = cam.view_proj * make_float4(prim.b, 1.f);
            if (clipB.w <= 1e-4f) continue;
            pb = (clipB.xy() / clipB.w * 0.5f + 0.5f) * float2{W, H};
        }
        data.push_back(make_float4(pa.x, pa.y, pb.x, pb.y));
        data.push_back(prim.rgba);
        data.push_back(make_float4(hw, 0.f, 0.f, 0.f));
    }
    if (data.empty()) return;

    const uint float4Count = static_cast<uint>(data.size());
    if (_capacityFloat4 < float4Count) {
        // Grow with headroom so steady-state editor sessions stop reallocating.
        _capacityFloat4 = std::max(float4Count, 1024u);
        _primBuffer = core::Renderer::device().create_buffer<float4>(_capacityFloat4);
    }
    stream << _primBuffer.view(0u, float4Count).copy_from(data.data());
    stream << _shader(ctx.renderTarget, _primBuffer, float4Count / 3u)
              .dispatch(ctx.width(), ctx.height());
}

void GizmoFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    // Keep the singleton's push gate and the pipeline's dispatch gate in
    // sync — a mismatch would grow the stack between (skipped) drains.
    bool en = j.value("enabled", Gizmo::get().enabled());
    Gizmo::get().setEnabled(en);
    setEnabled(en);
}

nlohmann::json GizmoFeature::toJson() const {
    nlohmann::json file;
    file["enabled"] = Gizmo::get().enabled();
    return file;
}

void GizmoFeature::drawUi() {
    if (ImGui::CollapsingHeader("Gizmo (editor overlay)")) {
        auto& gizmo = Gizmo::get();
        bool en = gizmo.enabled();
        if (ImGui::Checkbox("Enable", &en)) {
            gizmo.setEnabled(en);
            setEnabled(en);
        }
        ImGui::Text("Prims last frame: %u", gizmo.state().framePrimCount);
    }
}

} // namespace newtype::feature

#endif // NT_ENABLE_EDITOR
