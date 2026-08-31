#include "newtype/feature/ChromaticAberrationFeature.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

void ChromaticAberrationFeature::onInit(Device& device) {
    auto kernel = Kernel2D([&](
        ImageFloat output, ImageFloat input,
        Float strength, Float2 center,
        UInt sample_count, UInt frame_count
        ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float2 res_f = make_float2(cast<float>(resolution.x), cast<float>(resolution.y));
        Float2 uv = (make_float2(coord) + 0.5f) / res_f;
        Float2 dir = uv - center;
        UInt2 max_xy = make_uint2(cast<uint>(resolution.x) - 1, cast<uint>(resolution.y) - 1);
        res_f -= 1.f;

        auto barrelDistortion = [&](Float amt) {
            Float dist = dot(dir, dir);
            return saturate(uv + dir * dist * amt);
        };

        auto linterp = [&](Float t) {
            return saturate(1.f - abs(2.f * t - 1.f));
        };

        auto remap = [&](Float t, Float a, Float b) {
            return saturate((t - a) / (b - a));
        };

        auto spectrum_offset = [&](Float t) {
            Float4 ret;
            Float lo = step(t, .5f);
            Float hi = 1.f - lo;
            Float w = linterp(remap(t, 1.f / 6.f, 5.f / 6.f));
            ret = make_float4(lo, 1.f, hi, 1.f) * make_float4(1.f - w, w, 1.f - w, 1.f);

            return pow(ret, 1.f / 2.2f);
        };

        Float r_acc = def(0.0f);
        Float b_acc = def(0.0f);
        Float4 sumcol = def(make_float4(0.f));
        Float4 sumw = def(make_float4(0.f));

        $for(i, sample_count) {
            Float t = cast<float>(i) / cast<float>(sample_count);
            auto w = spectrum_offset(t);
            sumw += w;

            auto sf = barrelDistortion( t * strength );
            auto si = clamp(make_uint2(sf * res_f), def(make_uint2(0)), max_xy);
            sumcol += w * input.read(si);
        };

        output.write(coord, sumcol / sumw);
    });

    _shader = device.compile(kernel);
    _compiled = true;
}

void ChromaticAberrationFeature::onResize(Device& device, uint width, uint height) {
    _width = width;
    _height = height;
}

void ChromaticAberrationFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    if (!_compiled || !enabled()) return;
    uint w = ctx.width(), h = ctx.height();

    auto& temp = ctx.pipeline.requestTempImage(
        PixelStorage::FLOAT4, w, h, "postproc_temp");
    auto& profiler = util::Profiler::instance();

    stream << ctx.renderTarget.copy_to(temp);
    profiler.set_pass("PostEffects/ChromaticAberration");
    stream << _shader(ctx.renderTarget, temp, strength, center,
                      sampleCount, ctx.frameCount()).dispatch(w, h);
}

void ChromaticAberrationFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool en = j.value("enabled", false);
    setEnabled(en);
    strength    = j.value("strength", strength);
    sampleCount = j.value("samples",  sampleCount);
    if (j.contains("center") && j["center"].is_array() && j["center"].size() >= 2) {
        center = make_float2(j["center"][0].get<float>(), j["center"][1].get<float>());
    }
}

nlohmann::json ChromaticAberrationFeature::toJson() const {
    nlohmann::json file;
    file["enabled"]  = enabled();
    file["strength"] = strength;
    file["samples"]  = sampleCount;
    file["center"]   = { center.x, center.y };
    return file;
}

void ChromaticAberrationFeature::drawUi() {
    if (ImGui::CollapsingHeader("Chromatic Aberration")) {
        ImGui::ScopedId scpId("fCA");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::SliderFloat("Strength", &strength, 0.0f, 1.0f);
            ImGui::SliderFloat2("Center", reinterpret_cast<float*>(&center), -1.0f, 2.0f);
            int sc = static_cast<int>(sampleCount);
            if (ImGui::SliderInt("Samples", &sc, 1, 8)) sampleCount = static_cast<uint>(sc);
        }
    }
}

} // namespace newtype::feature
