#include "newtype/feature/DoFFeature.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

void DoFFeature::onInit(Device& device) {
    auto kernel = Kernel2D([&](ImageFloat output, ImageFloat input,
                               ImageFloat gbuf_depth,
                               Float focus_dist, Float aperture_val,
                               Float max_coc, UInt sample_count,
                               UInt frame_count) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float4 center = input.read(coord);
        Float depth = gbuf_depth.read(coord).x;

        // Skip sky pixels
        $if(depth > 1e10f) {
            output.write(coord, center);
            $return();
        };

        // Circle of Confusion — coc_px is the blur radius in pixels
        Float coc = aperture_val * abs(depth - focus_dist) / max(depth, 1e-3f);
        Float coc_px = min(coc * max_coc, max_coc);

        // Full passthrough for in-focus pixels (no blending needed)
        $if(coc_px < 0.5f) {
            output.write(coord, center);
            $return();
        };

        // Per-pixel random rotation to break spiral coherence
        UInt seed = coord.x * 1973u + coord.y * 9277u + frame_count * 26699u;
        Float rot_angle = fract(sin(cast<float>(seed)) * 43758.5453f) * 6.283185f;

        // Golden-angle spiral gather with per-pixel rotation
        Float3 accum = def(make_float3(0.0f));
        Float  accum_w = def(0.0f);
        constexpr float golden_angle = 2.39996322972f;

        $for(i, sample_count) {
            Float fi = cast<float>(i) + 0.5f;  // offset to avoid duplicate center
            Float angle = fi * golden_angle + rot_angle;
            Float r = sqrt(fi / cast<float>(sample_count)) * coc_px;
            Float2 offset = make_float2(cos(angle), sin(angle)) * r;
            Float2 sample_f = make_float2(coord) + 0.5f + offset;

            // Bilinear-safe integer coord
            Int2 sample_coord = make_int2(cast<int>(floor(sample_f.x)),
                                          cast<int>(floor(sample_f.y)));
            sample_coord = clamp(sample_coord, make_int2(0), make_int2(cast<int>(resolution.x) - 1,
                                                                        cast<int>(resolution.y) - 1));

            Float sample_depth = gbuf_depth.read(make_uint2(sample_coord)).x;

            // Depth-aware weight: soft rejection for samples at very different depth
            Float depth_ratio = sample_depth / max(depth, 1e-3f);
            Float weight = ite(depth_ratio > 0.5f & depth_ratio < 2.0f, 1.0f, 0.25f);

            Float4 sample_color = input.read(make_uint2(sample_coord));
            accum += sample_color.xyz() * weight;
            accum_w += weight;
        };

        // Center sample with full weight
        accum += center.xyz();
        accum_w += 1.0f;

        // Soft blend: crossfade between original and blurred based on coc_px
        Float3 blurred = accum / accum_w;
        Float blend = saturate((coc_px - 0.5f) / 1.5f);  // smooth over 1.5px transition
        Float3 result = lerp(center.xyz(), blurred, blend);

        output.write(coord, make_float4(result, center.w));
    });

    _shader = device.compile(kernel);
    _compiled = true;
}

void DoFFeature::onResize(Device& device, uint width, uint height) {
    _width = width;
    _height = height;
}

void DoFFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    if (!_compiled || !enabled()) return;
    uint w = ctx.width(), h = ctx.height();

    // Request temp image from pool (shared with other post features)
    auto& temp = ctx.pipeline.requestTempImage(
        PixelStorage::FLOAT4, w, h, "postproc_temp");
    auto& profiler = util::Profiler::instance();

    // Ping-pong: copy renderTarget → temp, then DoF(temp → renderTarget)
    stream << ctx.renderTarget.copy_to(temp);
    profiler.set_pass("PostEffects/DoF");
    stream << _shader(ctx.renderTarget, temp, ctx.frame.gbufDepth,
                      focusDistance, aperture, maxCoC, sampleCount,
                      ctx.frameCount()).dispatch(w, h);
}

void DoFFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool enabled = j.value("enabled", false);
    setEnabled(enabled);
    focusDistance = j.value("focusdist", focusDistance);
    aperture = j.value("aperture", aperture);
    maxCoC = j.value("coc", maxCoC);
    sampleCount = j.value("sample", sampleCount);
}

nlohmann::json DoFFeature::toJson() const {
    nlohmann::json file;
    file["enabled"] = enabled();
    file["focusdist"] = focusDistance;
    file["aperture"] = aperture;
    file["coc"] = maxCoC;
    file["sample"] = sampleCount;
    return file;
}

void DoFFeature::drawUi() {
    if (ImGui::CollapsingHeader("DoF")) {
        ImGui::ScopedId scpId("fDoF");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::SliderFloat("Focus Dist", &focusDistance, 0.1f, 50.0f);
            ImGui::SliderFloat("Aperture", &aperture, 0.0f, 0.5f);
            ImGui::SliderFloat("Max CoC", &maxCoC, 1.0f, 32.0f, "%.1f px");
            ImGui::SliderInt("Samples", reinterpret_cast<int*>(&sampleCount), 4, 32);
        }
    }
}

} // namespace newtype::plugin
