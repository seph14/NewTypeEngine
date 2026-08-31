#include "newtype/feature/MotionBlurFeature.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

void MotionBlurFeature::onInit(Device& device) {
    auto kernel = Kernel2D([&](ImageFloat output, ImageFloat input,
                               ImageFloat gbuf_bary_motion, ImageFloat gbuf_depth,
                               ImageUInt gbuf_vis,
                               Float shutter_scale, UInt sample_count,
                               UInt frame_count, UInt debug_mode) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord      = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        $if(coord.x >= resolution.x | coord.y >= resolution.y) { $return(); };

        Float4 center = input.read(coord);
        Float depth = gbuf_depth.read(coord).x;
        //UInt inst_id = gbuf_vis.read(coord).x;

        // Debug mode 2: pure pass-through (tests blit + pipeline, no blur logic)
        $if(debug_mode == 2u) {
            output.write(coord, center);
            $return();
        };

        // Skip sky pixels (inst_id == ~0u = no geometry hit)
        $if(depth > 1e10f) {
        //$if(inst_id == ~0u) {
            output.write(coord, center);
            $return();
        };

        // Motion vectors in NDC [-1,1] delta, convert to pixel velocity
        Float2 motion = gbuf_bary_motion.read(coord).zw();
        Float2 velocity = motion * make_float2(cast<float>(resolution.x) * 0.5f,
                                                cast<float>(resolution.y) * 0.5f);
        Float speed = length(velocity);

        // Debug mode 1: speed heatmap
        $if(debug_mode == 1u) {
            // Red channel = speed / 20, green = 0, show rings clearly
            Float s = saturate(speed / 20.0f);
            output.write(coord, make_float4(s, 0.0f, ite(speed < 2.0f, 1.0f, 0.0f), 1.0f));
            $return();
        };

        // Early-out: skip sub-pixel motion (TAA jitter, static geometry)
        $if(speed < 2.0f) {
            output.write(coord, center);
            $return();
        };

        // Integer hash (no sin() — avoids float32 precision rings on GPU)
        UInt seed = coord.x * 1973u + coord.y * 9277u + frame_count * 26699u;
        seed = (seed ^ 61u) ^ (seed >> 16u);
        seed = seed + (seed << 3u);
        seed = seed ^ (seed >> 4u);
        seed = seed * 0x27d4eb2du;
        seed = seed ^ (seed >> 15u);
        Float jitter = cast<float>(seed & 0xffffu) / 65536.0f;

        // Gather along motion vector
        Float2 direction = velocity / speed;
        Float3 accum = center.xyz();
        Float  accum_w = 1.0f;

        $for(i, sample_count) {
            Float t = (cast<float>(i) + jitter) / cast<float>(sample_count) - 0.5f;
            t = t * shutter_scale;
            Float2 sample_f = make_float2(coord) + direction * speed * t;
            Int2 sample_coord = make_int2(cast<int>(floor(sample_f.x)), cast<int>(floor(sample_f.y)));
            sample_coord = clamp(sample_coord, make_int2(0),
                                 make_int2(cast<int>(resolution.x) - 1, cast<int>(resolution.y) - 1));

            Float sample_depth = gbuf_depth.read(make_uint2(sample_coord)).x;

            // Reject sky samples (inst_id == ~0u)
            UInt sample_inst = gbuf_vis.read(make_uint2(sample_coord)).x;
            $if(sample_inst == ~0u) { $continue; };

            // Bilateral depth weight: tight gaussian rejection prevents
            // cross-surface color bleeding (e.g. floor color onto walls)
            Float depth_diff = abs(sample_depth - depth);
            Float depth_ratio = max(sample_depth, depth) / max(min(sample_depth, depth), 1e-6f);
            // Hard reject if depth ratio > 1.5% difference (prevents all cross-surface bleeding)
            Float hard_reject = ite(depth_ratio > 1.02f, 0.0f, 1.0f);
            Float sigma = max(0.001f * depth, 0.005f);
            Float weight = hard_reject * exp(-depth_diff * depth_diff / (sigma * sigma));

            Float4 sample_color = input.read(make_uint2(sample_coord));
            accum += sample_color.xyz() * weight;
            accum_w += weight;
        };

        output.write(coord, make_float4(accum / max(accum_w, 1e-3f), center.w));
    });

    _shader = device.compile(kernel);
    _compiled = true;
}

void MotionBlurFeature::onResize(Device& device, uint width, uint height) {
    _width = width;
    _height = height;
}

void MotionBlurFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    if (!_compiled || !enabled()) return;
    uint w = ctx.width(), h = ctx.height();

    // Request same temp image as DoF (shared pool, same id)
    auto& temp = ctx.pipeline.requestTempImage(
        PixelStorage::FLOAT4, w, h, "postproc_temp");
    auto& profiler = util::Profiler::instance();

    // Ping-pong: blit renderTarget → temp, then MB(temp → renderTarget)
    stream << ctx.renderTarget.copy_to(temp);
    profiler.set_pass("PostEffects/MotionBlur");
    stream << _shader(ctx.renderTarget, temp, ctx.frame.gbufBaryMotion,
                      ctx.frame.gbufDepth, ctx.frame.gbufVis,
                      shutterScale, sampleCount,
                      ctx.frameCount(), static_cast<uint>(debugMode)).dispatch(w, h);
}

void MotionBlurFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool enabled = j.value("enabled", false);
    setEnabled(enabled);
    shutterScale = j.value("shutter", shutterScale);
    sampleCount = j.value("sample", sampleCount);
}

nlohmann::json MotionBlurFeature::toJson() const {
    nlohmann::json file;
    file["enabled"] = enabled();
    file["shutter"] = shutterScale;
    file["sample"] = sampleCount;
    return file;
}

void MotionBlurFeature::drawUi() {
    if (ImGui::CollapsingHeader("Motion Blur")) {
        ImGui::ScopedId scpId("fmb");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::SliderFloat("Shutter", &shutterScale, 0.0f, 2.0f);
            ImGui::SliderInt("Samples", reinterpret_cast<int*>(&sampleCount), 2, 16);
            ImGui::SliderInt("Debug", &debugMode, 0, 2);
        }
    }
}

} // namespace newtype::plugin
