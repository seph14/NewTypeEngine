#pragma once

// DLSS 4.5 Ray Reconstruction (NGX feature SuperSamplingDenoising, Denoise
// mode DLUnified) as the engine's denoiser REPLACEMENT — it consumes the
// noisy pre-denoise signals at the Pass-9 slot where ReLAX would run and
// writes denoised HDR back into the render target, so OIT/glass/tonemap
// continue unchanged (docs/upscaling_feasibility_report.md §RTXDI addendum).
//
// Denoises 1:1 at RENDER resolution (in == out, DLAA-style) — upscaling stays
// the separate IUpscalerBackend stage after OIT/glass, matching the RTXDI 3.1
// SDK sample architecture. Feature creation needs an open command list, so it
// happens lazily inside the first dispatch's DXCustomCmd on the stream
// execution thread (same pattern as DlssSrBackend).

#include <luisa/luisa-compute.h>

#include <atomic>

struct NVSDK_NGX_Handle;

namespace newtype {
namespace ngx {

struct RrFrameParams {
    luisa::float2 jitterOffset{0.0f}; // render pixels (same value SR gets)
    bool reset = false;
    uint width = 1u;  // render resolution (in == out)
    uint height = 1u;
};

class DlssRrDenoiser final {
public:
    ~DlssRrDenoiser();

    [[nodiscard]] bool available() const noexcept {
        return _ngxHeld && !_featureFailed;
    }
    [[nodiscard]] const char* name() const noexcept {
        return "DLSS 4.5 Ray Reconstruction (NGX)";
    }

    // Brings up the shared NgxContext and checks RR availability. Does NOT
    // create the feature (that happens lazily on first dispatch).
    bool init(luisa::compute::Device& device, uint width, uint height);
    // Sizes are baked at creation — release now (callers drain streams) and
    // recreate lazily at the new size on the next dispatch.
    void resize(uint width, uint height);
    // color = noisy composited HDR (remod + emission + sky), output = the
    // frame render target (must differ from color). albedos/normalRoughness
    // come from the RR input format pass; depth/velocity are the upscaler
    // G-buffer pair. All at render resolution.
    void dispatch(luisa::compute::Stream& stream,
                  luisa::compute::Image<float>& color,
                  luisa::compute::Image<float>& depth,
                  luisa::compute::Image<float>& velocity,
                  luisa::compute::Image<float>& diffuseAlbedo,
                  luisa::compute::Image<float>& specularAlbedo,
                  luisa::compute::Image<float>& normalRoughness,
                  luisa::compute::Image<float>& output,
                  const RrFrameParams& params);
    void shutdown();

private:
    // Written from the stream execution thread (lazy feature creation);
    // read by the render thread only outside in-flight dispatches.
    NVSDK_NGX_Handle* _feature = nullptr;
    std::atomic<bool> _featureFailed = false;
    bool _ngxHeld = false;

    luisa::compute::Device* _device = nullptr; // retained for lazy creation
    uint _width = 0u, _height = 0u;

    // CPU-side parity check (DLSS rejects odd render dims) — sets
    // _featureFailed so available() flips before the first dispatch.
    void validateDimensions();
    void releaseFeature();
};

} // namespace ngx
} // namespace newtype
