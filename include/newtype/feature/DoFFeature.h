#pragma once
#include <luisa/luisa-compute.h>
#include "newtype/core/IFeature.h"

namespace newtype::feature {

class DoFFeature : public core::IFeature {
public:
    using DoFShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,  // 0: output (renderTarget)
        luisa::compute::Image<float>,  // 1: input (temp copy)
        luisa::compute::Image<float>,  // 2: depth (R32F)
        float,                         // 3: focusDistance
        float,                         // 4: aperture
        float,                         // 5: maxCoC (pixels)
        luisa::uint,                   // 6: sampleCount
        luisa::uint                    // 7: frameCount (jitter seed)
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
    float focusDistance = 5.0f;
    float aperture     = 0.05f;
    float maxCoC       = 16.0f;
    uint  sampleCount  = 16u;

private:
    DoFShaderType _shader;
    uint _width = 0, _height = 0;
    bool _compiled = false;
};

} // namespace newtype::plugin
