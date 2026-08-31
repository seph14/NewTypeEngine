#pragma once
#include <luisa/luisa-compute.h>
#include "newtype/core/IFeature.h"

namespace newtype::feature {

class MotionBlurFeature : public core::IFeature {
public:
    using MBShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,  // 0: output (renderTarget)
        luisa::compute::Image<float>,  // 1: input (temp copy)
        luisa::compute::Image<float>,  // 2: gbufBaryMotion (RGBA16F, .zw() = motion)
        luisa::compute::Image<float>,  // 3: depth (R32F)
        luisa::compute::Image<uint>,   // 4: gbufVis (INT2, .x = inst_id, ~0u = sky)
        float,                         // 5: shutterScale
        luisa::uint,                   // 6: sampleCount
        luisa::uint,                   // 7: frameCount (jitter seed)
        luisa::uint                    // 8: debugMode
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

private:
    // Parameters
    float shutterScale = 1.0f;
    uint  sampleCount = 8u;
    int   debugMode = 0;  // 0=off, 1=show speed heatmap, 2=pass-through

    MBShaderType   _shader;
    uint _width = 0, _height = 0;
    bool _compiled = false;
};

} // namespace newtype::plugin
