#include "newtype/upscal/Fsr31Backend.h"

#include "newtype/core/Config.h"
#include "cinder/Log.h"

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

// ffx_api (vendored FidelityFX SDK 2.3.0 subset — external/FidelityFX)
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/dx12/ffx_api_dx12.h>

#include <windows.h>
#include <d3d12.h>

namespace newtype {
namespace upscal {

namespace {

void fsr_message(uint32_t type, const wchar_t* message) {
    // Cinder's log streaming has no wchar_t* inserter — narrow it.
    char narrow[512];
    size_t n = 0;
    for (; message[n] != L'\0' && n + 1 < sizeof(narrow); ++n)
        narrow[n] = static_cast<char>(message[n]);
    narrow[n] = '\0';
    if (type == FFX_API_MESSAGE_TYPE_ERROR)
        CI_LOG_E("FSR3: " << narrow);
    else
        CI_LOG_W("FSR3: " << narrow);
}

// DXCustomCmd wrapping one ffxDispatch(FFX_API_DISPATCH_DESC_TYPE_UPSCALE).
// Usage declaration -> LuisaCompute inserts the D3D12 barriers; execute()
// only records FSR's own commands on the already-barriered list. Pattern
// from the LuisaCompute examples/extension/fsr3.cpp FSRCommand, ported to
// the ffx_api dispatch-desc API (FidelityFX SDK 2.x).
class FsrUpscaleCommand final : public luisa::compute::DXCustomCmd {
    luisa::vector<ResourceUsage> _usages;
    PfnFfxDispatch _dispatch;
    ffxContext _context;
    mutable ffxDispatchDescUpscale _desc{};

    [[nodiscard]] luisa::span<ResourceUsage> get_resource_usages() noexcept override {
        return _usages;
    }

    FfxApiResource wrap(luisa::compute::Image<float>& image,
                        uint32_t ffxState, D3D12_RESOURCE_STATES barrierState) {
        _usages.emplace_back(luisa::compute::Argument::Texture{image.handle(), 0u},
                             barrierState);
        auto* res = static_cast<ID3D12Resource*>(image.native_handle());
        return ffxApiGetResourceDX12(res, ffxState);
    }

public:
    [[nodiscard]] luisa::compute::StreamTag stream_tag() const noexcept override {
        return luisa::compute::StreamTag::COMPUTE;
    }

    FsrUpscaleCommand(PfnFfxDispatch dispatch, ffxContext context,
                      luisa::compute::Image<float>& color,
                      luisa::compute::Image<float>& depth,
                      luisa::compute::Image<float>& velocity,
                      luisa::compute::Image<float>& output,
                      const UpscaleFrameParams& p)
        : _dispatch{dispatch}, _context{context} {
        _desc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        // Inputs are unjittered camera+geometry motion in render-res pixels
        // (prev = curr + mv), so the scale factor is identity. Depth is NDC
        // z/w, non-inverted, near/far = camera clips.
        _desc.color        = wrap(color,     FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _desc.depth        = wrap(depth,     FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _desc.motionVectors = wrap(velocity, FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _desc.exposure                    = FfxApiResource{};
        _desc.reactive                    = FfxApiResource{};
        _desc.transparencyAndComposition  = FfxApiResource{};
        _desc.output = wrap(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        _desc.jitterOffset.x = p.jitterOffset.x;
        _desc.jitterOffset.y = p.jitterOffset.y;
        _desc.motionVectorScale.x = 1.0f;
        _desc.motionVectorScale.y = 1.0f;
        _desc.renderSize.width  = p.renderWidth;
        _desc.renderSize.height = p.renderHeight;
        _desc.upscaleSize.width  = p.displayWidth;
        _desc.upscaleSize.height = p.displayHeight;
        _desc.enableSharpening = p.sharpness > 1e-5f;
        _desc.sharpness        = p.sharpness;
        // ffx_api expects milliseconds (the Cauldron sample multiplies by 1000).
        _desc.frameTimeDelta = p.deltaTimeSeconds * 1000.0f;
        _desc.preExposure = p.preExposure;
        _desc.reset       = p.reset;
        _desc.cameraNear  = p.cameraNear;
        _desc.cameraFar   = p.cameraFar;
        _desc.cameraFovAngleVertical = p.cameraFovVertical;
        _desc.viewSpaceToMetersFactor = 1.0f;
        _desc.flags = 0u;
    }

    void execute(IDXGIAdapter1*, IDXGIFactory2*, ID3D12Device*,
                 ID3D12GraphicsCommandList4* command_list) const noexcept override {
        _desc.commandList = command_list;
        ffxContext context = _context; // ffxDispatch takes the handle by pointer
        _dispatch(&context, &_desc.header);
    }
};

} // anonymous namespace

Fsr31Backend::~Fsr31Backend() {
    shutdown();
}

bool Fsr31Backend::loadDlls() {
    if (_createContext != nullptr) return true;

    // DLLs are deployed next to the executable (post-build copy from
    // external/FidelityFX/bin). The loader discovers the upscaler provider
    // DLL by name in its own search path — keep them colocated.
    HMODULE loader = LoadLibraryA("amd_fidelityfx_loader_dx12.dll");
    if (loader == nullptr) {
        CI_LOG_W("Fsr31Backend: amd_fidelityfx_loader_dx12.dll not found (GetLastError="
            << GetLastError() << "). Deploy external/FidelityFX/bin next to the exe.");
        return false;
    }
    auto create  = reinterpret_cast<PfnFfxCreateContext>(
        GetProcAddress(loader, "ffxCreateContext"));
    auto destroy = reinterpret_cast<PfnFfxDestroyContext>(
        GetProcAddress(loader, "ffxDestroyContext"));
    auto dispatch = reinterpret_cast<PfnFfxDispatch>(
        GetProcAddress(loader, "ffxDispatch"));
    if (create == nullptr || destroy == nullptr || dispatch == nullptr) {
        CI_LOG_W("Fsr31Backend: amd_fidelityfx_loader_dx12.dll is missing the "
                 "ffxCreateContext/ffxDestroyContext/ffxDispatch exports.");
        FreeLibrary(loader);
        return false;
    }
    _loader = loader;
    _createContext = create;
    _destroyContext = destroy;
    _dispatch = dispatch;
    return true;
}

bool Fsr31Backend::createContext(luisa::compute::Device& device) {
    if (!loadDlls()) return false;

    auto dxDevice = static_cast<ID3D12Device*>(device.native_handle());

    // Backend desc chained into the effect desc (single ffxCreateContext call
    // — no separate backend context handle is retained).
    ffxCreateBackendDX12Desc backendDesc{};
    backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backendDesc.device = dxDevice;

    ffxCreateContextDescUpscale createDesc{};
    createDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    createDesc.header.pNext = &backendDesc.header;
    // Input color is linear pre-tonemap HDR; exposure is handled by the
    // engine after the upscale (preExposure = 1), so auto-exposure is left
    // ON for FSR's internal luminance tracking.
    createDesc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                       FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    createDesc.maxRenderSize  = {_renderWidth, _renderHeight};
    createDesc.maxUpscaleSize = {_displayWidth, _displayHeight};
    createDesc.fpMessage = &fsr_message;

    ffxContext context = nullptr;
    auto rc = reinterpret_cast<PfnFfxCreateContext>(_createContext)(
        &context, &createDesc.header, nullptr);
    if (rc != FFX_API_RETURN_OK || context == nullptr) {
        CI_LOG_E("Fsr31Backend: ffxCreateContext(UPSCALE) failed with " << rc);
        return false;
    }
    _context = context;
    CI_LOG_I("Fsr31Backend: FSR 3.1 context created (render "
        << _renderWidth << "x" << _renderHeight << " -> display "
        << _displayWidth << "x" << _displayHeight << ")");
    return true;
}

bool Fsr31Backend::init(luisa::compute::Device& device,
                        uint renderWidth, uint renderHeight,
                        uint displayWidth, uint displayHeight) {
    _device = &device;
    _renderWidth = renderWidth;
    _renderHeight = renderHeight;
    _displayWidth = displayWidth;
    _displayHeight = displayHeight;
    return createContext(device);
}

void Fsr31Backend::resize(uint renderWidth, uint renderHeight,
                          uint displayWidth, uint displayHeight) {
    if (_renderWidth == renderWidth && _renderHeight == renderHeight &&
        _displayWidth == displayWidth && _displayHeight == displayHeight)
        return;
    // Context sizes are baked in at creation — full destroy/recreate.
    shutdown();
    _renderWidth = renderWidth;
    _renderHeight = renderHeight;
    _displayWidth = displayWidth;
    _displayHeight = displayHeight;
    if (_device != nullptr)
        createContext(*_device);
}

void Fsr31Backend::dispatch(luisa::compute::Stream& stream,
                            luisa::compute::Image<float>& color,
                            luisa::compute::Image<float>& depth,
                            luisa::compute::Image<float>& velocity,
                            luisa::compute::Image<float>& output,
                            const UpscaleFrameParams& params) {
    if (_context == nullptr) return;
    auto cl = luisa::compute::CommandList::create();
    cl << luisa::make_unique<FsrUpscaleCommand>(
        reinterpret_cast<PfnFfxDispatch>(_dispatch), _context,
        color, depth, velocity, output, params);
    stream << cl.commit();
}

void Fsr31Backend::shutdown() {
    if (_context != nullptr) {
        // The GPU must be idle before destroying the context — callers
        // (Pipeline::setUpscalerMode / setRenderScale / resize) synchronize
        // the stream around recreation.
        ffxContext context = _context;
        reinterpret_cast<PfnFfxDestroyContext>(_destroyContext)(&context, nullptr);
        _context = nullptr;
    }
}

} // namespace upscal
} // namespace newtype
