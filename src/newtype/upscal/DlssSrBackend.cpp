#include "newtype/upscal/DlssSrBackend.h"

#include "newtype/ngx/NgxContext.h"

#include "cinder/Log.h"

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

// DLSS SDK (vendored under external/NVIDIA/DLSS, tag v310.9.1)
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_d3d.h>

#include <windows.h>
#include <d3d12.h>

namespace newtype {
namespace upscal {

namespace {

// Map the free renderScale slider onto DLSS's PerfQuality ladder so the
// feature is created with the quality/perf expectations closest to the
// actual ratio (sizes are passed explicitly; this is the tradeoff hint).
// UltraQuality is deliberately never emitted: the mode is not part of the
// DLSS 3.5+/4.x mode set and the vendored 310.9.1 DLL fails feature
// creation with FAIL_MissingInput when it is requested (verified: every
// scale in the 0.72-0.95 band failed, MaxQuality and below all created).
// Bands are midpoint-separated around the canonical ratios (DLAA 1.0,
// Quality .67, Balanced .59, Performance .5, UltraPerformance .33).
NVSDK_NGX_PerfQuality_Value map_perf_quality(float scale) {
    if (scale >= 0.95f) return NVSDK_NGX_PerfQuality_Value_DLAA;
    if (scale >= 0.6255f) return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    if (scale >= 0.544f) return NVSDK_NGX_PerfQuality_Value_Balanced;
    if (scale >= 0.4165f) return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

// DXCustomCmd wrapping one NVSDK_NGX_D3D12_EvaluateFeature_C for DLSS-SR,
// with lazy feature creation on first execute (creation needs an open
// command list — this is the engine's only sanctioned way onto the Luisa
// stream's list). Usage declaration -> LuisaCompute inserts the D3D12
// barriers; execute() records NGX's own commands. The NGX parameter map is
// the shared NgxContext capability map: the eval helper mutates it, so all
// parameter writes happen here on the stream execution thread where
// executions are serialized in submission order (never on the render
// thread, which may run frames ahead of the GPU).
class DlssUpscaleCommand final : public luisa::compute::DXCustomCmd {
    luisa::vector<ResourceUsage> _usages;
    NVSDK_NGX_Handle** _featureSlot;
    std::atomic<bool>* _featureFailed;
    mutable NVSDK_NGX_DLSS_Create_Params _create{}; // helper takes non-const
    mutable NVSDK_NGX_D3D12_DLSS_Eval_Params _eval{};
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

    DlssUpscaleCommand(NVSDK_NGX_Handle** featureSlot,
                       std::atomic<bool>* featureFailed,
                       luisa::compute::Image<float>& color,
                       luisa::compute::Image<float>& depth,
                       luisa::compute::Image<float>& velocity,
                       luisa::compute::Image<float>& output,
                       const UpscaleFrameParams& p)
        : _featureSlot{featureSlot}, _featureFailed{featureFailed} {
        // Same input contract as FSR (Fsr31Backend): unjittered render-res
        // pixel motion vectors (identity MV scale), NDC z/w non-inverted
        // depth (no DepthInverted flag), linear pre-tonemap HDR color, no
        // exposure buffer (AutoExposure flag; the engine applies exposure
        // after the upscale so preExposure = 1).
        const float scale =
            static_cast<float>(p.renderWidth) / static_cast<float>(p.displayWidth);
        _create.Feature.InWidth = p.renderWidth;
        _create.Feature.InHeight = p.renderHeight;
        _create.Feature.InTargetWidth = p.displayWidth;
        _create.Feature.InTargetHeight = p.displayHeight;
        _create.Feature.InPerfQualityValue = map_perf_quality(scale);
        _create.InFeatureCreateFlags =
            NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
            NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
            NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

        _eval.Feature.pInColor = wrap(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInDepth = wrap(depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.pInMotionVectors = wrap(velocity, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _eval.Feature.pInOutput = wrap(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Jitter offset in input/render pixel space — same value FSR gets
        // (cam NDC jitter * 0.5 * render size), Halton-32 compatible.
        _eval.InJitterOffsetX = p.jitterOffset.x;
        _eval.InJitterOffsetY = p.jitterOffset.y;
        _eval.InRenderSubrectDimensions.Width = p.renderWidth;
        _eval.InRenderSubrectDimensions.Height = p.renderHeight;
        _eval.InReset = p.reset ? 1 : 0;
        _eval.InMVScaleX = 0.0f; // helper defaults identity (pixels)
        _eval.InMVScaleY = 0.0f;
        _eval.InPreExposure = p.preExposure;
    }

    void execute(IDXGIAdapter1*, IDXGIFactory2*, ID3D12Device*,
                 ID3D12GraphicsCommandList4* command_list) const noexcept override {
        auto& ngx = ngx::NgxContext::the();
        if (*_featureSlot == nullptr && !_featureFailed->load(std::memory_order_relaxed)) {
            auto rc = NGX_D3D12_CREATE_DLSS_EXT(command_list, 1u, 1u,
                                                _featureSlot,
                                                ngx.capabilityParams(),
                                                &_create);
            if (NVSDK_NGX_FAILED(rc) || *_featureSlot == nullptr) {
                *_featureFailed = true;
                if (!_createLogged) {
                    _createLogged = true;
                    CI_LOG_E("DlssSrBackend: DLSS-SR feature creation failed "
                             "(result 0x" << std::hex << static_cast<unsigned>(rc)
                             << std::dec << ") - falling back to native render.");
                }
                return;
            }
            if (!_createLogged) {
                _createLogged = true;
                CI_LOG_I("DlssSrBackend: DLSS-SR feature created ("
                    << _create.Feature.InWidth << "x" << _create.Feature.InHeight
                    << " -> " << _create.Feature.InTargetWidth << "x"
                    << _create.Feature.InTargetHeight << ")");
            }
        }
        if (*_featureSlot == nullptr)
            return;
        NGX_D3D12_EVALUATE_DLSS_EXT(command_list, *_featureSlot,
                                    ngx.capabilityParams(), &_eval);
    }
};

} // anonymous namespace

DlssSrBackend::~DlssSrBackend() {
    shutdown();
}

// DLSS feature creation is deterministic-fatal on odd RENDER dimensions (the
// network's 2-px phase grid; observed as NVSDK create failing and the feature
// never recovering). Catch it CPU-side so the pipeline can bilinear-fallback
// from the first frame instead of showing a broken view for one frame. Odd
// DISPLAY dims are left to the real creation attempt — only handled by the
// async failure path if the driver rejects them.
void DlssSrBackend::validateDimensions() {
    if ((_renderWidth % 2u == 0u) && (_renderHeight % 2u == 0u))
        return;
    _featureFailed = true;
    CI_LOG_E("DlssSrBackend: render size " << _renderWidth << "x" << _renderHeight
        << " has odd dimensions - DLSS requires even input dims. Falling back "
           "to bilinear upscaling until the size changes.");
}

bool DlssSrBackend::init(luisa::compute::Device& device,
                         uint renderWidth, uint renderHeight,
                         uint displayWidth, uint displayHeight) {
    _device = &device;
    _renderWidth = renderWidth;
    _renderHeight = renderHeight;
    _displayWidth = displayWidth;
    _displayHeight = displayHeight;

    auto* dxDevice = static_cast<ID3D12Device*>(device.native_handle());
    if (!ngx::NgxContext::acquire(dxDevice)) {
        _ngxHeld = false;
        return false;
    }
    _ngxHeld = true;
    if (!ngx::NgxContext::the().srAvailable()) {
        CI_LOG_E("DlssSrBackend: NGX reports DLSS-SR unavailable on this "
                 "system (driver too old for the vendored DLLs? DLSS 4.5 "
                 "needs 580.00+).");
        // Release the acquire() ref taken above (see DlssRrDenoiser::init).
        ngx::NgxContext::release();
        _ngxHeld = false;
        return false;
    }
    validateDimensions();
    return true;
}

void DlssSrBackend::resize(uint renderWidth, uint renderHeight,
                           uint displayWidth, uint displayHeight) {
    if (_renderWidth == renderWidth && _renderHeight == renderHeight &&
        _displayWidth == displayWidth && _displayHeight == displayHeight)
        return;
    // Feature sizes are baked in at creation — release now (callers drain the
    // streams around resize, GPU idle) and let the next dispatch lazily
    // recreate at the new sizes. The failed latch clears too: a size change is
    // the one thing that can turn a previously fatal creation (odd dims,
    // transient failure) into a working one, so retry once per resize.
    releaseFeature();
    _featureFailed = false;
    _renderWidth = renderWidth;
    _renderHeight = renderHeight;
    _displayWidth = displayWidth;
    _displayHeight = displayHeight;
    validateDimensions();
}

void DlssSrBackend::releaseFeature() {
    if (_feature != nullptr) {
        NVSDK_NGX_D3D12_ReleaseFeature(_feature);
        _feature = nullptr;
    }
}

void DlssSrBackend::dispatch(luisa::compute::Stream& stream,
                             luisa::compute::Image<float>& color,
                             luisa::compute::Image<float>& depth,
                             luisa::compute::Image<float>& velocity,
                             luisa::compute::Image<float>& output,
                             const UpscaleFrameParams& params) {
    if (!_ngxHeld || _featureFailed.load(std::memory_order_relaxed))
        return;
    auto cl = luisa::compute::CommandList::create();
    cl << luisa::make_unique<DlssUpscaleCommand>(
        &_feature, &_featureFailed, color, depth, velocity, output, params);
    stream << cl.commit();
}

void DlssSrBackend::shutdown() {
    releaseFeature();
    _featureFailed = false;
    if (_ngxHeld) {
        ngx::NgxContext::release();
        _ngxHeld = false;
    }
}

} // namespace upscal
} // namespace newtype
