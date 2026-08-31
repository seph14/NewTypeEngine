#pragma once
#include <luisa/luisa-compute.h>
#include "newtype/core/IFeature.h"

namespace newtype::feature {

// Radial chromatic aberration on the post-bloom HDR render target.
// Per-channel offset along the direction from a user-tunable UV focus point.
// Samples are jittered across frames for temporal smoothing.
class ChromaticAberrationFeature : public core::IFeature {
public:
    using ShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (renderTarget, in-place via temp)
        luisa::compute::Image<float>,   // 1: input  (temp copy of renderTarget)
        float,                          // 2: strength (max pixel offset at unit dir length)
        luisa::float2,                  // 3: center (UV focus point)
        luisa::uint,                    // 4: sampleCount
        luisa::uint                     // 5: frameCount (jitter seed)
    >;

    [[nodiscard]] core::FeaturePoint point() const override {
        return core::FeaturePoint::AfterGlassTint;
    }
    void onInit(luisa::compute::Device& device) override;
    void onResize(luisa::compute::Device& device, uint width, uint height) override;
    void onExecute(luisa::compute::Stream& stream, const core::FeatureContext& ctx) override;
    void drawUi() override;

    void load(const nlohmann::json& file) override;
    [[nodiscard]] nlohmann::json toJson() const override;

    // Parameters
    float  strength    = 0.5f;                  // max pixel offset at unit distance from center
    uint   sampleCount = 3u;                    // radial gather count per channel (≥1)
    luisa::float2 center {0.5f, 0.5f};          // UV-space focus point; default = screen center

private:
    ShaderType _shader;
    uint _width = 0, _height = 0;
    bool _compiled = false;
};

} // namespace newtype::feature
