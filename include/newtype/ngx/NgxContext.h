#pragma once

// Direct-NGX (no Streamline) device context, shared by the DLSS-SR upscaler
// backend (upscal/DlssSrBackend) and the DLSS Ray Reconstruction denoiser.
//
// The DLSS SDK is vendored under external/NVIDIA/DLSS (tag v310.9.1): headers
// + the nvsdk_ngx_s.lib stub linked at build time; the feature DLLs
// (nvngx_dlss.dll / nvngx_dlssd.dll) are deployed next to the exe and found
// by the driver-side NGX core. On non-RTX systems init or the availability
// queries fail and callers degrade gracefully (no upscaler / ReLAX).
//
// One NVSDK_NGX_D3D12_Init per device per process: the context is a process
// wide singleton refcounted by acquire()/release() — the last release shuts
// NGX down. Callers must drain the GPU work that used NGX features before
// releasing (the pipeline's stream-synchronize envelope covers this).

#include <cstdint>

struct ID3D12Device;
struct NVSDK_NGX_Parameter;

namespace newtype {
namespace ngx {

class NgxContext {
public:
    // Initializes NGX against the device on first use and queries the
    // SR/RR availability. Returns false when NGX is unavailable on this
    // system (non-RTX GPU / driver without NGX); reason is logged once.
    static bool acquire(ID3D12Device* device);
    // Drops one acquire; NGX shuts down when the last user releases.
    static void release();

    // Process-wide instance (valid whenever acquire() succeeded).
    [[nodiscard]] static NgxContext& the() noexcept { return _instance; }

    [[nodiscard]] bool initialized() const noexcept { return _initialized; }
    [[nodiscard]] bool srAvailable() const noexcept { return _srAvailable; }
    [[nodiscard]] bool rrAvailable() const noexcept { return _rrAvailable; }

    // Shared capability parameters. NGX parameter maps are a mutable
    // key-value store shared by feature create/evaluate — only touch them
    // from one thread at a time, i.e. inside DXCustomCmd::execute() on the
    // stream execution thread (commands execute in submission order), or
    // while the streams are known idle.
    [[nodiscard]] NVSDK_NGX_Parameter* capabilityParams() const noexcept {
        return _params;
    }

private:
    static NgxContext _instance;

    bool init(ID3D12Device* device);
    void shutdown();

    int _refs = 0;
    bool _initialized = false;
    bool _srAvailable = false;
    bool _rrAvailable = false;
    ID3D12Device* _device = nullptr; // retained for Shutdown1
    NVSDK_NGX_Parameter* _params = nullptr;
};

} // namespace ngx
} // namespace newtype
