#pragma once

// FSR 3.1 upscaler backend via the ffx_api (FidelityFX SDK 2.x, vendored under
// external/FidelityFX). Compile-time dependency is header-only; the runtime
// DLLs (amd_fidelityfx_loader_dx12.dll + amd_fidelityfx_upscaler_dx12.dll)
// are loaded dynamically, so a missing SDK degrades to available() == false.

#include "newtype/upscal/UpscalerBackend.h"

namespace newtype {
namespace upscal {

class Fsr31Backend final : public IUpscalerBackend {
public:
    ~Fsr31Backend() override;

    [[nodiscard]] bool available() const noexcept override { return _context != nullptr; }
    [[nodiscard]] const char* name() const noexcept override { return "FSR 3.1 (ffx_api)"; }

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
    // Opaque loader module + ffx_api entry points (ffx_api_loader.h table).
    void* _loader = nullptr;                 // HMODULE
    void* _createContext = nullptr;          // PfnFfxCreateContext
    void* _destroyContext = nullptr;         // PfnFfxDestroyContext
    void* _dispatch = nullptr;               // PfnFfxDispatch
    void* _context = nullptr;                // ffxContext handle

    luisa::compute::Device* _device = nullptr; // retained for context recreation
    uint _renderWidth = 0u, _renderHeight = 0u;
    uint _displayWidth = 0u, _displayHeight = 0u;

    bool loadDlls();
    bool createContext(luisa::compute::Device& device);
};

} // namespace upscal
} // namespace newtype
