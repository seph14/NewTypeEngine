#include "newtype/ngx/DlssRayReconstruction.h"

#include "newtype/ngx/NgxContext.h"

#include "cinder/Log.h"

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

// DLSS SDK (vendored under external/NVIDIA/DLSS, tag v310.9.1)
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>

#include <windows.h>
#include <d3d12.h>

namespace newtype {
namespace ngx {

namespace {

// DXCustomCmd wrapping one NVSDK_NGX_D3D12_EvaluateFeature_C for DLSS-RR
// (Denoise mode DLUnified), with lazy feature creation on first execute —
// creation needs the stream's open command list. Parameter writes happen
// here on the stream execution thread where they serialize in submission
// order (the NGX parameter map is the shared NgxContext capability map).
class DlssRrCommand final : public luisa::compute::DXCustomCmd {
    luisa::vector<ResourceUsage> _usages;
    NVSDK_NGX_Handle** _featureSlot;
    std::atomic<bool>* _featureFailed;
    mutable NVSDK_NGX_DLSSD_Create_Params _create{};
    mutable NVSDK_NGX_D3D12_DLSSD_Eval_Params _eval{};
    mutable bool _createLogged = false;

    [[nodiscard]] luisa::span<ResourceUsage> get_resource_usages() noexcept override {
        return _usages;
    }

    ID3D12Resource* wrap(luisa::compute::Image<float>& image,
                         D3D12_RESOURCE_STATES barrierState) {
        _usages.emplace_back(luisa::compute::Argument::Texture{image.handle(), 0u},
                             barrierState);
        return static_cast<ID3D12Resource*>(image.native_handle());
    }

public:
    [[nodiscard]] luisa::compute::StreamTag stream_tag() const noexcept override {
        return luisa::compute::StreamTag::COMPUTE;
    }

    DlssRrCommand(NVSDK_NGX_Handle** featureSlot, std::atomic<bool>* featureFailed,
                  luisa::compute::Image<float>& color,
                  luisa::compute::Image<float>& depth,
                  luisa::compute::Image<float>& velocity,
                  luisa::compute::Image<float>& diffuseAlbedo,
                  luisa::compute::Image<float>& specularAlbedo,
                  luisa::compute::Image<float>& normalRoughness,
                  luisa::compute::Image<float>& output,
                  const RrFrameParams& p)
        : _featureSlot{featureSlot}, _featureFailed{featureFailed} {
        // RR input contract (DLSS-RR Integration Guide 310.9.1): unjittered
        // render-res pixel motion vectors, non-inverted NDC z/w depth
        // (Use_HW_Depth = HW), world-space normals with roughness packed in
        // .w (Roughness_Mode_Packed -> GBuffer.Roughness binds the same
        // texture), float diffuse albedo / specular F0, noisy composited HDR
        // color. Exposure/auto-exposure inputs are NOT supported by RR (the
        // model handles HDR internally). in == out (denoise-only, DLAA mode)
        // — upscaling is the separate DLSS-SR stage.
        _create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        _create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        _create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
        _create.InWidth = p.width;
        _create.InHeight = p.height;
        _create.InTargetWidth = p.width;
        _create.InTargetHeight = p.height;
        _create.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
        _create.InFeatureCreateFlags =
            NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
            NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

        auto* normalRes = wrap(normalRoughness,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInColor = wrap(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInDepth = wrap(depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInMotionVectors = wrap(velocity,
                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInDiffuseAlbedo = wrap(diffuseAlbedo,
                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInSpecularAlbedo = wrap(specularAlbedo,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInNormals = normalRes;
        // Packed mode: the helper binds GBuffer.Roughness from pInRoughness —
        // point it at the same normalRoughness texture (roughness lives in .w).
        _eval.pInRoughness = normalRes;
        _eval.pInOutput = wrap(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Jitter in input/render pixel space (guide §3.6 — RR recommends a
        // long Halton sequence; the engine's unbounded Halton(2,3) qualifies).
        _eval.InJitterOffsetX = p.jitterOffset.x;
        _eval.InJitterOffsetY = p.jitterOffset.y;
        _eval.InRenderSubrectDimensions.Width = p.width;
        _eval.InRenderSubrectDimensions.Height = p.height;
        _eval.InReset = p.reset ? 1 : 0;
        _eval.InMVScaleX = 0.0f; // helper defaults identity (pixels)
        _eval.InMVScaleY = 0.0f;
    }

    void execute(IDXGIAdapter1*, IDXGIFactory2*, ID3D12Device*,
                 ID3D12GraphicsCommandList4* command_list) const noexcept override {
        auto& ngx = NgxContext::the();
        if (*_featureSlot == nullptr && !_featureFailed->load(std::memory_order_relaxed)) {
            auto rc = NGX_D3D12_CREATE_DLSSD_EXT(command_list, 1u, 1u,
                                                 _featureSlot,
                                                 ngx.capabilityParams(),
                                                 &_create);
            if (NVSDK_NGX_FAILED(rc) || *_featureSlot == nullptr) {
                *_featureFailed = true;
                if (!_createLogged) {
                    _createLogged = true;
                    CI_LOG_E("DlssRrDenoiser: DLSS-RR feature creation failed "
                             "(result 0x" << std::hex << static_cast<unsigned>(rc)
                             << std::dec << ") - falling back to the raw "
                             "no-denoise composite.");
                }
                return;
            }
            if (!_createLogged) {
                _createLogged = true;
                CI_LOG_I("DlssRrDenoiser: DLSS-RR feature created ("
                    << _create.InWidth << "x" << _create.InHeight
                    << ", denoise-only 1:1)");
            }
        }
        if (*_featureSlot == nullptr)
            return;
        NGX_D3D12_EVALUATE_DLSSD_EXT(command_list, *_featureSlot,
                                     ngx.capabilityParams(), &_eval);
    }
};

} // anonymous namespace

DlssRrDenoiser::~DlssRrDenoiser() {
    shutdown();
}

// Same parity constraint as DlssSrBackend (the feature shares DLSS's 2-px
// phase grid): catch fatal odd render dims CPU-side so setDenoiserMode can
// report RR inactive immediately and the pipeline stays on ReLAX.
void DlssRrDenoiser::validateDimensions() {
    if ((_width % 2u == 0u) && (_height % 2u == 0u))
        return;
    _featureFailed = true;
    CI_LOG_E("DlssRrDenoiser: render size " << _width << "x" << _height
        << " has odd dimensions - DLSS-RR requires even dims. Staying on "
           "ReLAX until the size changes.");
}

bool DlssRrDenoiser::init(luisa::compute::Device& device, uint width, uint height) {
    _device = &device;
    _width = width;
    _height = height;

    auto* dxDevice = static_cast<ID3D12Device*>(device.native_handle());
    if (!NgxContext::acquire(dxDevice)) {
        _ngxHeld = false;
        return false;
    }
    _ngxHeld = true;
    if (!NgxContext::the().rrAvailable()) {
        CI_LOG_E("DlssRrDenoiser: NGX reports DLSS-RR unavailable (driver too "
                 "old for the vendored DLLs? DLSS 4.5 needs 580.00+).");
        // Release the acquire() ref taken above so NGX can shut down when
        // the last user (e.g. a working SR backend) goes away.
        NgxContext::release();
        _ngxHeld = false;
        return false;
    }
    validateDimensions();
    return true;
}

void DlssRrDenoiser::resize(uint width, uint height) {
    if (_width == width && _height == height)
        return;
    releaseFeature();
    // Clear the failed latch: a size change can turn a previously fatal
    // creation (odd dims) into a working one — retry once per resize.
    _featureFailed = false;
    _width = width;
    _height = height;
    validateDimensions();
}

void DlssRrDenoiser::releaseFeature() {
    if (_feature != nullptr) {
        // GPU idle guaranteed by the caller's stream-drain envelope.
        NVSDK_NGX_D3D12_ReleaseFeature(_feature);
        _feature = nullptr;
    }
}

void DlssRrDenoiser::dispatch(luisa::compute::Stream& stream,
                              luisa::compute::Image<float>& color,
                              luisa::compute::Image<float>& depth,
                              luisa::compute::Image<float>& velocity,
                              luisa::compute::Image<float>& diffuseAlbedo,
                              luisa::compute::Image<float>& specularAlbedo,
                              luisa::compute::Image<float>& normalRoughness,
                              luisa::compute::Image<float>& output,
                              const RrFrameParams& params) {
    if (!_ngxHeld || _featureFailed.load(std::memory_order_relaxed))
        return;
    auto cl = luisa::compute::CommandList::create();
    cl << luisa::make_unique<DlssRrCommand>(
        &_feature, &_featureFailed, color, depth, velocity,
        diffuseAlbedo, specularAlbedo, normalRoughness, output, params);
    stream << cl.commit();
}

void DlssRrDenoiser::shutdown() {
    releaseFeature();
    _featureFailed = false;
    if (_ngxHeld) {
        NgxContext::release();
        _ngxHeld = false;
    }
}

} // namespace ngx
} // namespace newtype
