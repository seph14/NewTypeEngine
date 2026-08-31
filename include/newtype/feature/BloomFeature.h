#pragma once
#include <luisa/luisa-compute.h>
#include <vector>
#include "newtype/core/IFeature.h"

namespace newtype::feature {

// HDR bloom (Karis-style "UnrealBloom" port).
// Pipeline: bright pass (Karis soft-knee) -> N Karis 13-tap downsamples ->
// N 9-tap tent upsamples with radius-controlled lerp -> additive composite.
// Operates on the denoised HDR render target at FeaturePoint::AfterGlassTint.
class BloomFeature : public core::IFeature {
public:
    using BrightShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (mips[0] at 1/2-res, HALF4)
        luisa::compute::Image<float>,   // 1: input  (full-res render target, FLOAT4)
        float,                          // 2: threshold
        float                           // 3: knee (soft width)
    >;
    using DownsampleShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (mips[i+1], HALF4)
        luisa::compute::Image<float>    // 1: input  (mips[i],   HALF4)
    >;
    using UpsampleShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (mips[i] updated, HALF4)
        luisa::compute::Image<float>,   // 1: lower  (mips[i],   HALF4)
        luisa::compute::Image<float>,   // 2: higher (mips[i+1], HALF4)
        float                           // 3: lerp factor (clamp(radius, 0, 1))
    >;
    using CompositeShaderType = luisa::compute::Shader<2,
        luisa::compute::Image<float>,   // 0: output (renderTarget, in-place)
        luisa::compute::Image<float>,   // 1: bloom (mips[0] upsampled, HALF4)
        float                           // 2: strength
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

    // Parameters (mirror three.js UnrealBloomPass defaults)
    float strength  = 1.0f;   // final additive intensity
    float radius    = 0.85f;  // 0–1, upsample blend factor (three.js "radius")
    float threshold = 0.0f;   // HDR luminance threshold; 0 = no threshold
    uint  mipLevels = 5u;     // pyramid depth (1/2 → 1/32 at 1080p); clamped on small targets

private:
    BrightShaderType      _bright;
    DownsampleShaderType  _downsample;
    DownsampleShaderType  _downsampleKaris;  // luminance-weighted, used for first mip pass only
    UpsampleShaderType    _upsample;
    CompositeShaderType   _composite;

    std::vector<luisa::compute::Image<float>> _mips;    // [0..N-1], HALF4, sizes w/2, w/4, ...
    // Upsample uses in-place read-then-write per thread (each thread reads its own coord of `lower`
    // before overwriting it). Same pattern as the glass tint in-place pass — safe in DX compute.
    uint _fullW = 0, _fullH = 0;
    uint _effectiveMips = 0u;
    bool _compiled = false;
};

} // namespace newtype::feature
