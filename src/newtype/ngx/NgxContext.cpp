#include "newtype/ngx/NgxContext.h"

#include "cinder/Log.h"

// DLSS SDK (vendored under external/NVIDIA/DLSS, tag v310.9.1)
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_defs_dlssd.h> // SuperSamplingDenoising availability keys
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>

#include <d3d12.h>

#include <filesystem>

namespace newtype {
namespace ngx {

namespace {

// Dev placeholder (donut's default). Register an application ID with NVIDIA
// and switch to the rel DLLs before shipping.
constexpr unsigned long long kNgxApplicationId = 231313132ull;

void NVSDK_CONV ngx_log(const char* message, NVSDK_NGX_Logging_Level level,
                        NVSDK_NGX_Feature /*source*/) {
    // NGX with MinimumLoggingLevel = OFF only forwards errors this way.
    if (level == NVSDK_NGX_LOGGING_LEVEL_OFF && message == nullptr)
        return;
    CI_LOG_W("NGX: " << (message ? message : "<null>"));
}

} // anonymous namespace

NgxContext NgxContext::_instance;

bool NgxContext::acquire(ID3D12Device* device) {
    if (_instance._refs > 0) {
        ++_instance._refs;
        return _instance._initialized;
    }
    if (!_instance.init(device)) {
        // init() already logged the reason; do not keep a half-open state.
        return false;
    }
    _instance._refs = 1;
    return true;
}

void NgxContext::release() {
    if (_instance._refs == 0)
        return;
    if (--_instance._refs == 0)
        _instance.shutdown();
}

bool NgxContext::init(ID3D12Device* device) {
    if (_initialized)
        return true;
    if (device == nullptr)
        return false;

    // NGX wants a writable directory for its logs/temp files. The exe dir is
    // not reliably writable (Program Files); the temp dir always is.
    std::error_code ec;
    auto dataPath = std::filesystem::temp_directory_path(ec);
    if (ec)
        dataPath = std::filesystem::current_path(ec);
    const std::wstring dataPathW = dataPath.wstring();

    NVSDK_NGX_FeatureCommonInfo commonInfo{};
    commonInfo.LoggingInfo.LoggingCallback = &ngx_log;
    commonInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    commonInfo.LoggingInfo.DisableOtherLoggingSinks = true;

    auto rc = NVSDK_NGX_D3D12_Init(kNgxApplicationId, dataPathW.c_str(),
                                   device, &commonInfo);
    if (NVSDK_NGX_FAILED(rc)) {
        CI_LOG_W("NgxContext: NVSDK_NGX_D3D12_Init failed (result 0x"
            << std::hex << static_cast<unsigned>(rc)
            << std::dec << ") - no RTX GPU / NGX-capable driver? "
               "DLSS stays unavailable (FSR/ReLAX unaffected).");
        return false;
    }

    rc = NVSDK_NGX_D3D12_GetCapabilityParameters(&_params);
    if (NVSDK_NGX_FAILED(rc) || _params == nullptr) {
        CI_LOG_W("NgxContext: GetCapabilityParameters failed (result 0x"
            << std::hex << static_cast<unsigned>(rc) << std::dec << ").");
        NVSDK_NGX_D3D12_Shutdown1(device);
        return false;
    }

    int srAvail = 0, rrAvail = 0;
    NVSDK_NGX_Parameter_GetI(_params, NVSDK_NGX_Parameter_SuperSampling_Available,
                             &srAvail);
    NVSDK_NGX_Parameter_GetI(_params,
                             NVSDK_NGX_Parameter_SuperSamplingDenoising_Available,
                             &rrAvail);
    if (!srAvail || !rrAvail) {
        // Surface the driver-side reason (usually a too-old driver for the
        // vendored 310.9.1 feature DLLs — DLSS 4.5 wants 580.00+).
        int needUpdate = 0;
        NVSDK_NGX_Parameter_GetI(_params,
                                 NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                                 &needUpdate);
        CI_LOG_W("NgxContext: NGX initialized but features unavailable "
                 "(SR=" << srAvail << " RR=" << rrAvail
                 << ", needsUpdatedDriver=" << needUpdate
                 << ") - update the GeForce driver (DLSS 4.5 needs 580.00+).");
    }

    _device = device;
    _srAvailable = srAvail != 0;
    _rrAvailable = rrAvail != 0;
    _initialized = true;
    CI_LOG_I("NgxContext: NGX initialized (DLSS-SR "
        << (_srAvailable ? "available" : "unavailable") << ", DLSS-RR "
        << (_rrAvailable ? "available" : "unavailable") << ")");
    return true;
}

void NgxContext::shutdown() {
    if (!_initialized)
        return;
    // Features must already be released by their owners (GPU idle). The
    // capability parameter map is NGX-managed (DestroyParameters must not be
    // called on it) — Shutdown1 reclaims it.
    _params = nullptr;
    NVSDK_NGX_D3D12_Shutdown1(_device);
    _device = nullptr;
    _srAvailable = false;
    _rrAvailable = false;
    _initialized = false;
}

} // namespace ngx
} // namespace newtype
