#pragma once

// DLSS 4.5 Super Resolution upscaler backend via direct NGX (no Streamline).
// The NGX headers + nvsdk_ngx_s.lib stub are vendored under
// external/NVIDIA/DLSS; the feature DLLs (nvngx_dlss.dll) deploy next to the
// exe. Feature-creation needs an open D3D12 command list, so the NGX feature
// is created lazily inside the first dispatch's DXCustomCmd (on the Luisa
// stream's execution thread, in submission order) — init() only brings up
// the device-level ngx::NgxContext and checks SR availability.

#include "newtype/upscal/UpscalerBackend.h"

#include <atomic>

struct NVSDK_NGX_Handle;

namespace newtype {
namespace upscal {

class DlssSrBackend final : public IUpscalerBackend {
public:
    ~DlssSrBackend() override;

    [[nodiscard]] bool available() const noexcept override {
        return _ngxHeld && !_featureFailed;
    }
    [[nodiscard]] const char* name() const noexcept override {
        return "DLSS 4.5 SR (NGX)";
    }

    bool init(luisa::compute::Device& device,
              uint renderWidth, uint renderHeight,
              uint displayWidth, uint displayHeight) override;
    void resize(uint renderWidth, uint renderHeight,
                uint displayWidth, uint displayHeight) override;
    void dispatch(luisa::compute::Stream& stream,
                  luisa::compute::Image<float>& color,
                  luisa::compute::Image<float>& depth,
                  luisa::compute::Image<float>& velocity,
                  luisa::compute::Image<float>& output,
                  const UpscaleFrameParams& params) override;
    void shutdown() override;

private:
    // Written from the stream execution thread (lazy feature creation);
    // read by the render thread only outside in-flight dispatches.
    NVSDK_NGX_Handle* _feature = nullptr;
    std::atomic<bool> _featureFailed = false;
    bool _ngxHeld = false;

    luisa::compute::Device* _device = nullptr; // retained for lazy creation
    uint _renderWidth = 0u, _renderHeight = 0u;
    uint _displayWidth = 0u, _displayHeight = 0u;

    // CPU-side parity check (DLSS rejects odd render dims) — sets
    // _featureFailed so available() flips before the first dispatch.
    void validateDimensions();
    void releaseFeature();
};

} // namespace upscal
} // namespace newtype
