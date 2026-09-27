#pragma once

#include <luisa/luisa-compute.h>

namespace newtype {
namespace upscal {

// Available upscaler backends. "None" renders at display resolution with no
// upscale stage (bit-identical to the pre-upscaler pipeline).
enum class UpscalerMode {
    None = 0,
    Fsr3 = 1,   // FSR 3.1 via ffx_api (external/FidelityFX), dynamically loaded
    Dlss = 2,   // DLSS 4.5 Super Resolution via direct NGX (external/NVIDIA/DLSS)
};

// Per-frame dispatch parameters. See FfxApiDispatchDescUpscale field mapping
// in Fsr31Backend.cpp for the vendor-side interpretation.
struct UpscaleFrameParams {
    luisa::float2 jitterOffset{0.0f};  // sub-pixel camera jitter in RENDER pixels
                                       // (stays 0 while primaryJitterEnabled=false)
    float preExposure = 1.0f;          // engine applies exposure post-upscale
    float deltaTimeSeconds = 1.0f / 60.0f;
    bool  reset = false;               // history reset (resize / teleport / toggles)
    float cameraNear = 0.1f;
    float cameraFar = 100.0f;
    float cameraFovVertical = 0.7854f; // radians
    float sharpness = 0.0f;            // RCAS sharpening, 0 = off
    uint  renderWidth = 1u;
    uint  renderHeight = 1u;
    uint  displayWidth = 1u;
    uint  displayHeight = 1u;
};

// One upscaler backend = one dynamically-loaded vendor SDK. The pipeline owns
// exactly one active backend (UpscalerMode); contexts are recreated on
// resize / render-scale change. All entry points are called from the render
// thread outside the frame's stream submission except dispatch(), which runs
// between AfterGlassTint and the tonemap block.
class IUpscalerBackend {
public:
    virtual ~IUpscalerBackend() = default;

    // True when the vendor DLLs loaded AND the effect context exists.
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual const char* name() const noexcept = 0;

    // Creates the effect context. Returns false (and stays unavailable) when
    // DLLs are missing or context creation fails — the pipeline then falls
    // back to no-upscaler operation.
    virtual bool init(luisa::compute::Device& device,
                      uint renderWidth, uint renderHeight,
                      uint displayWidth, uint displayHeight) = 0;

    // Destroy + recreate the context at new sizes (resize / scale change).
    virtual void resize(uint renderWidth, uint renderHeight,
                        uint displayWidth, uint displayHeight) = 0;

    // Submits the upscale dispatch on the stream. color/depth/velocity are at
    // render resolution; output is FLOAT4 HDR at display resolution.
    virtual void dispatch(luisa::compute::Stream& stream,
                          luisa::compute::Image<float>& color,
                          luisa::compute::Image<float>& depth,
                          luisa::compute::Image<float>& velocity,
                          luisa::compute::Image<float>& output,
                          const UpscaleFrameParams& params) = 0;

    // Tears down the context (keeps DLLs loaded; init() may run again).
    virtual void shutdown() = 0;
};

} // namespace upscal
} // namespace newtype
