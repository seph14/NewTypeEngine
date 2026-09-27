#pragma once
#include <luisa/luisa-compute.h>
#include "newtype/core/IFeature.h"
#include "newtype/core/Pipeline.h" // willRun() reads pipeline state inline

namespace newtype::feature {

// FXAA (3.11 console variant) on the tone-mapped LDR display target.
// The AA stand-in while primary jitter/TAA stays off: with the primary
// jitter disabled (docs/primary_jitter_plan.md — the unjittered-reconstruction
// hybrid makes per-frame triangle flips speckle under jitter), geometry
// silhouettes hard-alias ~1px. FXAA runs after tonemap as a final screen-space
// directional blur with a luma clamp, so it cannot touch the HDR radiance
// chain the denoiser validates against.
class FxaaFeature : public core::IFeature {
public:
    using ShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (display target, in-place via temp)
        luisa::compute::Image<float>,   // 1: input  (temp copy of display target)
        float,                          // 2: spanMax (max edge blur length, pixels)
        float,                          // 3: reduceMul (edge-threshold scale vs corner luma avg)
        float                           // 4: reduceMin (edge-threshold floor)
    >;

    [[nodiscard]] core::FeaturePoint point() const override {
        return core::FeaturePoint::AfterToneMap;
    }
    void onInit(luisa::compute::Device& device) override;
    void onResize(luisa::compute::Device& device, uint width, uint height) override;
    void onExecute(luisa::compute::Stream& stream, const core::FeatureContext& ctx) override;
    void drawUi() override;

    // True when FXAA will dispatch this frame: compiled + enabled + not
    // skipped by the upscaler guard (perf R2 item 14 — FSR at renderScale<1
    // already reconstructs and RCAS-sharpens; a display-res FXAA pass on
    // top is double-AA at display cost). Single predicate for the tonemap
    // routing decision AND onExecute, so a skipped frame can never leave a
    // routed-but-unwritten display target.
    [[nodiscard]] bool willRun(const core::Pipeline& pipeline) const noexcept {
        if (!_compiled || !enabled()) return false;
        return !(pipeline.fxaaSkipWhenUpscaled() &&
                 pipeline.lastFrameUpscaled() &&
                 pipeline.renderScale() < 1.0f);
    }

    // perf R2 item 13: true only when the feature will run this frame — the
    // pipeline then routes tonemap into "fxaa_temp" and FXAA writes the
    // display target directly (input copy_to skipped).
    [[nodiscard]] bool wantsTonemapTempRoute(const core::Pipeline& pipeline) const noexcept override {
        return willRun(pipeline);
    }

    void load(const nlohmann::json& file) override;
    [[nodiscard]] nlohmann::json toJson() const override;

    // Parameters (FXAA console constants — see the fetched reference port)
    float spanMax   = 8.0f;          // max edge search span (pixels)
    float reduceMul = 1.0f / 8.0f;   // edge-threshold scale
    float reduceMin = 1.0f / 128.0f; // edge-threshold floor (dark-scene noise gate)

private:
    ShaderType _shader;
    uint _width = 0, _height = 0;
    bool _compiled = false;
};

} // namespace newtype::feature
