#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/VideoDecoderD3D11.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <propvarutil.h>
#include <d3d11.h>
#include <d3d11_4.h>   // ID3D11Multithread
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace newtype::media {

int VideoDecoderD3D11::s_instanceCount = 0;

VideoDecoderD3D11::~VideoDecoderD3D11() {
    close();
}

bool VideoDecoderD3D11::open(std::filesystem::path const& path,
                             ID3D12Device* luisaDevice,
                             Mode mode) {
    if (_reader) {
        CI_LOG_W("VideoDecoderD3D11::open called while already open; ignoring");
        return false;
    }

    // MFStartup is refcounted internally; we still keep our own counter so we know
    // when to call MFShutdown on the last close (matches MF semantics either way).
    if (s_instanceCount == 0) {
        HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
        if (FAILED(hr)) {
            CI_LOG_E("MFStartup failed: hr=0x" << std::hex << hr);
            return false;
        }
    }
    ++s_instanceCount;

    _mode = mode;

    bool ok = (_mode == Mode::CpuRgb32)
        ? _open_cpu_rgb32(path)
        : _open_hardware_nv12(path, luisaDevice);

    if (!ok) close();
    return ok;
}

//------------------------------------------------------------------------------
// Phase 1 path: software decode, RGB32 CPU buffer.
//------------------------------------------------------------------------------
bool VideoDecoderD3D11::_open_cpu_rgb32(std::filesystem::path const& path) {
    ComPtr<IMFAttributes> attr;
    HRESULT hr = MFCreateAttributes(&attr, 4);
    if (FAILED(hr)) {
        CI_LOG_E("MFCreateAttributes failed: hr=0x" << std::hex << hr);
        return false;
    }
    // Software decode + software video processor. DXVA disabled because no D3D11
    // device is provided in this mode.
    attr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    attr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    attr->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, TRUE);

    std::wstring wpath = path.wstring();
    ComPtr<IMFSourceReader> reader;
    hr = MFCreateSourceReaderFromURL(wpath.c_str(), attr.Get(), &reader);
    if (FAILED(hr)) {
        CI_LOG_E("MFCreateSourceReaderFromURL failed for '"
                 << path.string() << "': hr=0x" << std::hex << hr);
        return false;
    }
    _reader = reader.Detach();

    if (!_configure_rgb32_cpu_output()) return false;
    if (!_query_frame_metrics())         return false;

    CI_LOG_I("VideoDecoderD3D11 opened (CpuRgb32): "
             << _width << "x" << _height
             << ", stride=" << _cpuStride
             << ", duration=" << _durationSec << "s");
    return true;
}

//------------------------------------------------------------------------------
// Phase 2+ path: D3D11 hardware decode, NV12 texture array output.
//------------------------------------------------------------------------------
bool VideoDecoderD3D11::_open_hardware_nv12(std::filesystem::path const& path,
                                             ID3D12Device* luisaDevice) {
    if (!luisaDevice) {
        CI_LOG_E("HardwareNV12 mode requires a non-null Luisa D3D12 device for LUID matching");
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter;
    if (!_create_d3d11_device_on_luid(luisaDevice, &adapter)) return false;

    // Wrap the D3D11 device in IMFDXGIDeviceManager; MF will use it for DXVA decode.
    UINT resetToken = 0;
    ComPtr<IMFDXGIDeviceManager> dxgiManager;
    HRESULT hr = MFCreateDXGIDeviceManager(&resetToken, &dxgiManager);
    if (FAILED(hr)) {
        CI_LOG_E("MFCreateDXGIDeviceManager failed: hr=0x" << std::hex << hr);
        return false;
    }
    hr = dxgiManager->ResetDevice(_d3d11Device, resetToken);
    if (FAILED(hr)) {
        CI_LOG_E("IMFDXGIDeviceManager::ResetDevice failed: hr=0x" << std::hex << hr);
        return false;
    }
    _dxgiManager = dxgiManager.Detach();

    ComPtr<IMFAttributes> attr;
    hr = MFCreateAttributes(&attr, 4);
    if (FAILED(hr)) {
        CI_LOG_E("MFCreateAttributes failed: hr=0x" << std::hex << hr);
        return false;
    }
    // HW decode path: DXVA enabled, no video processing (decoder's native NV12
    // output is what we want; Phase 3 handles conversion to RGBA8).
    attr->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, _dxgiManager);
    attr->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, FALSE);
    attr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, FALSE);

    std::wstring wpath = path.wstring();
    ComPtr<IMFSourceReader> reader;
    hr = MFCreateSourceReaderFromURL(wpath.c_str(), attr.Get(), &reader);
    if (FAILED(hr)) {
        CI_LOG_E("MFCreateSourceReaderFromURL failed for '"
                 << path.string() << "': hr=0x" << std::hex << hr);
        return false;
    }
    _reader = reader.Detach();

    if (!_configure_nv12_hw_output()) return false;
    if (!_query_frame_metrics())      return false;
    if (!_init_video_processor())    return false;

    CI_LOG_I("VideoDecoderD3D11 opened (HardwareNV12): "
             << _width << "x" << _height
             << ", duration=" << _durationSec << "s"
             << " (NV12 → RGBA8 via Video Processor, BT.709)");
    return true;
}

bool VideoDecoderD3D11::_create_d3d11_device_on_luid(ID3D12Device* luisaDevice,
                                                      IDXGIAdapter1** outAdapter) {
    // Find the DXGI adapter backing the Luisa D3D12 device. If D3D11 and D3D12
    // land on different physical adapters, OpenSharedHandle1 in Phase 4+ will
    // silently produce garbage — this is the highest-severity risk per the plan.
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        CI_LOG_E("CreateDXGIFactory2 failed: hr=0x" << std::hex << hr);
        return false;
    }

    LUID const luLuid = luisaDevice->GetAdapterLuid();

    ComPtr<IDXGIAdapter1> matched;
    for (UINT i = 0; factory->EnumAdapters1(i, &matched) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        matched->GetDesc1(&desc);
        if (desc.AdapterLuid.LowPart == luLuid.LowPart &&
            desc.AdapterLuid.HighPart == luLuid.HighPart) {
            char descBuf[128] = {};
            std::wcstombs(descBuf, desc.Description, sizeof(descBuf) - 1);
            CI_LOG_I("D3D11 device will use adapter '" << descBuf
                     << "' (LUID match with D3D12: low=0x" << std::hex
                     << desc.AdapterLuid.LowPart << " high=0x"
                     << desc.AdapterLuid.HighPart << std::dec << ")");
            break;
        }
        matched.Reset();
    }
    if (!matched) {
        CI_LOG_E("No DXGI adapter matches Luisa D3D12 LUID "
                 << "(low=0x" << std::hex << luLuid.LowPart
                 << " high=0x" << luLuid.HighPart << std::dec << ")");
        return false;
    }

    // D3D11_CREATE_DEVICE_VIDEO_SUPPORT is required for DXVA-backed decode.
    // BGRA support is required to interoperate with our eventual RGBA8 output.
    // (D3D11_CREATE_DEVICE_MULTITHREADED was removed from modern Windows SDKs;
    // MF's IMFDXGIDeviceManager handles cross-thread synchronization itself.)
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    ComPtr<ID3D11Device>        device;
    ComPtr<ID3D11DeviceContext> context;
    hr = D3D11CreateDevice(
        matched.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,         // MUST be UNKNOWN when passing adapter
        nullptr,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0,                       // default feature levels
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context);
    if (FAILED(hr)) {
        CI_LOG_E("D3D11CreateDevice failed: hr=0x" << std::hex << hr);
        return false;
    }

    _d3d11Device  = device.Detach();
    _d3d11Context = context.Detach();
    *outAdapter   = matched.Detach();

    // MS docs for MF_SA_D3D11_DEVICE require multithread protection: MF's DXVA
    // decode runs on worker threads; without this, our render-thread VP/Map
    // calls race with MF's internal device usage and the GPU hangs at the next
    // Flush/Map. (D3D11 devices default to non-thread-safe mode.)
    ComPtr<ID3D11Multithread> multithread;
    hr = _d3d11Device->QueryInterface(IID_PPV_ARGS(&multithread));
    if (SUCCEEDED(hr) && multithread) {
        multithread->SetMultithreadProtected(TRUE);
        CI_LOG_I("D3D11 device multithread protection enabled");
    } else {
        CI_LOG_W("Failed to enable D3D11 multithread protection (hr=0x"
                 << std::hex << hr << ") — VP/Map may hang under concurrent MF access");
    }
    return true;
}

bool VideoDecoderD3D11::_configure_nv12_hw_output() {
    ComPtr<IMFMediaType> outType;
    HRESULT hr = MFCreateMediaType(&outType);
    if (FAILED(hr)) return false;
    if (FAILED(outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video))) return false;
    if (FAILED(outType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12))) return false;

    hr = _reader->SetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, outType.Get());
    if (FAILED(hr)) {
        CI_LOG_E("SetCurrentMediaType(NV12) failed: hr=0x" << std::hex << hr
                 << " — asset decoder may not natively output NV12"
                 << " (try CpuRgb32 mode for graceful fallback)");
        return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Phase 3: stand up the D3D11 Video Processor for NV12 → RGBA8 conversion.
// Per-frame blt happens in _convert_nv12_to_rgba8().
//------------------------------------------------------------------------------
bool VideoDecoderD3D11::_init_video_processor() {
    ComPtr<ID3D11VideoDevice> videoDevice;
    HRESULT hr = _d3d11Device->QueryInterface(IID_PPV_ARGS(&videoDevice));
    if (FAILED(hr)) {
        CI_LOG_E("QueryInterface(ID3D11VideoDevice) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Video Processor state mutations live on ID3D11VideoContext, not on the
    // immediate context itself.
    ComPtr<ID3D11VideoContext> videoContext;
    hr = _d3d11Context->QueryInterface(IID_PPV_ARGS(&videoContext));
    if (FAILED(hr)) {
        CI_LOG_E("QueryInterface(ID3D11VideoContext) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Content desc — describes the I/O shape so the driver can pick a fast path.
    // Usage = PLAYBACK_NORMAL signals to the driver to optimize for video
    // rendering (which implies BT.709 for HD-sized inputs).
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC contentDesc{};
    contentDesc.InputFrameFormat  = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    contentDesc.InputFrameRate    = { 30, 1 };
    contentDesc.InputWidth        = _width;
    contentDesc.InputHeight       = _height;
    contentDesc.OutputFrameRate   = { 30, 1 };
    contentDesc.OutputWidth       = _width;
    contentDesc.OutputHeight      = _height;
    contentDesc.Usage             = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
    hr = videoDevice->CreateVideoProcessorEnumerator(&contentDesc, &enumerator);
    if (FAILED(hr)) {
        CI_LOG_E("CreateVideoProcessorEnumerator failed: hr=0x" << std::hex << hr);
        return false;
    }

    ComPtr<ID3D11VideoProcessor> vp;
    hr = videoDevice->CreateVideoProcessor(enumerator.Get(), 0, &vp);
    if (FAILED(hr)) {
        CI_LOG_E("CreateVideoProcessor failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Verify the VP enumerator accepts NV12 input. Some drivers/enumerators are
    // picky about which formats they support; better to fail fast here than at
    // every per-frame input-view creation.
    UINT formatFlags = 0;
    hr = enumerator->CheckVideoProcessorFormat(DXGI_FORMAT_NV12, &formatFlags);
    if (FAILED(hr) || formatFlags == 0) {
        CI_LOG_E("VP enumerator does not support NV12 input (hr=0x"
                 << std::hex << hr << ", flags=" << formatFlags << ")");
        return false;
    }

    // Allocate our RGBA8 output textures — two of them:
    //   _rgba8TextureInternal: VP output target. Plain texture, no shared flags.
    //     VP rejects KEYEDMUTEX-flagged textures as output on some drivers
    //     (E_INVALIDARG on VideoProcessorBlt).
    //   _rgba8Texture: shared with D3D12 via NT handle + keyed mutex. We
    //     CopyResource from internal→shared per frame under the mutex.
    D3D11_TEXTURE2D_DESC rgba8Desc{};
    rgba8Desc.Width            = _width;
    rgba8Desc.Height           = _height;
    rgba8Desc.MipLevels        = 1;
    rgba8Desc.ArraySize        = 1;
    rgba8Desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    rgba8Desc.SampleDesc.Count = 1;
    rgba8Desc.Usage            = D3D11_USAGE_DEFAULT;
    rgba8Desc.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    ComPtr<ID3D11Texture2D> rgba8Internal;
    hr = _d3d11Device->CreateTexture2D(&rgba8Desc, nullptr, &rgba8Internal);
    if (FAILED(hr)) {
        CI_LOG_E("CreateTexture2D(RGBA8 internal) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Same desc + shared flags for the cross-API texture.
    rgba8Desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                          D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    ComPtr<ID3D11Texture2D> rgba8Tex;
    hr = _d3d11Device->CreateTexture2D(&rgba8Desc, nullptr, &rgba8Tex);
    if (FAILED(hr)) {
        CI_LOG_E("CreateTexture2D(RGBA8, shared) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Pull the keyed mutex from the SHARED texture only — coordinates cross-API
    // writes vs reads.
    ComPtr<IDXGIKeyedMutex> keyedMutex;
    hr = rgba8Tex.As(&keyedMutex);
    if (FAILED(hr)) {
        CI_LOG_E("QueryInterface(IDXGIKeyedMutex) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Mint the NT handle that D3D12 will OpenSharedHandle.
    ComPtr<IDXGIResource1> dxgiResource;
    hr = rgba8Tex.As(&dxgiResource);
    if (FAILED(hr)) {
        CI_LOG_E("QueryInterface(IDXGIResource1) failed: hr=0x" << std::hex << hr);
        return false;
    }
    HANDLE sharedHandle = nullptr;
    hr = dxgiResource->CreateSharedHandle(
        nullptr,
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr,
        &sharedHandle);
    if (FAILED(hr)) {
        CI_LOG_E("CreateSharedHandle failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Stable NV12 copy texture — single array slice. NV12 used as VP input
    // typically needs BOTH bind flags (per MS sample); SRV-only triggers
    // E_INVALIDARG on CreateVideoProcessorInputView on some drivers.
    // IMPORTANT: zero MiscFlags explicitly — rgba8Desc was mutated above to
    // carry SHARED flags for the cross-API texture, and we don't want nv12Copy
    // to inherit those (VP blt fails with E_INVALIDARG on shared-flagged input).
    D3D11_TEXTURE2D_DESC nv12CopyDesc = rgba8Desc;
    nv12CopyDesc.Format    = DXGI_FORMAT_NV12;
    nv12CopyDesc.BindFlags = D3D11_BIND_DECODER | D3D11_BIND_SHADER_RESOURCE;
    nv12CopyDesc.MiscFlags = 0;
    ComPtr<ID3D11Texture2D> nv12Copy;
    hr = _d3d11Device->CreateTexture2D(&nv12CopyDesc, nullptr, &nv12Copy);
    if (FAILED(hr)) {
        CI_LOG_E("CreateTexture2D(NV12 copy) failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Create the VP input view ONCE on _nv12Copy. The view references the
    // texture (not its contents), so it stays valid across frames — the
    // CopySubresourceRegion updates the underlying bytes, the view sees them.
    // Per-frame view creation churns the runtime and triggers spurious
    // E_INVALIDARG on some drivers.
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC ivDesc{};
    ivDesc.ViewDimension          = D3D11_VPIV_DIMENSION_TEXTURE2D;
    ivDesc.Texture2D.MipSlice     = 0;
    ivDesc.Texture2D.ArraySlice   = 0;

    ComPtr<ID3D11VideoProcessorInputView> nv12InputView;
    hr = videoDevice->CreateVideoProcessorInputView(
        nv12Copy.Get(), enumerator.Get(), &ivDesc, &nv12InputView);
    if (FAILED(hr)) {
        CI_LOG_E("CreateVideoProcessorInputView (cached) failed: hr=0x"
                 << std::hex << hr);
        return false;
    }

    // Output view — on the internal (non-shared) texture. VP writes here;
    // we then CopyResource to the shared texture under keyed mutex.
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ovDesc{};
    ovDesc.ViewDimension     = D3D11_VPOV_DIMENSION_TEXTURE2D;
    ovDesc.Texture2D.MipSlice = 0;

    ComPtr<ID3D11VideoProcessorOutputView> outputView;
    hr = videoDevice->CreateVideoProcessorOutputView(
        rgba8Internal.Get(), enumerator.Get(), &ovDesc, &outputView);
    if (FAILED(hr)) {
        CI_LOG_E("CreateVideoProcessorOutputView failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Configure color spaces:
    //   Input  (NV12): BT.709, limited range 16-235
    //   Output (RGBA8): full range 0-255
    // Nominal_Range field is 2 bits; values are 0=unspec, 1=limited, 2=full.
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE inCS{};
    inCS.Usage         = 0;  // 0 = playback
    inCS.YCbCr_Matrix  = 1;  // 1 = BT.709
    inCS.Nominal_Range = 1;  // 1 = limited (16-235)
    videoContext->VideoProcessorSetStreamColorSpace(vp.Get(), 0, &inCS);

    D3D11_VIDEO_PROCESSOR_COLOR_SPACE outCS{};
    outCS.Usage         = 0;  // playback
    outCS.YCbCr_Matrix  = 1;  // BT.709
    outCS.Nominal_Range = 2;  // 2 = full (0-255)
    videoContext->VideoProcessorSetOutputColorSpace(vp.Get(), &outCS);

    // Frame format: progressive (default assumption but explicit is safer).
    // Some drivers hang VP blt without this set.
    videoContext->VideoProcessorSetStreamFrameFormat(
        vp.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);

    // Set source/target rects once at init — full frame, no scaling.
    RECT const fullRect{ 0, 0, static_cast<LONG>(_width), static_cast<LONG>(_height) };
    videoContext->VideoProcessorSetStreamSourceRect (vp.Get(), 0, TRUE, &fullRect);
    videoContext->VideoProcessorSetStreamDestRect    (vp.Get(), 0, TRUE, &fullRect);
    videoContext->VideoProcessorSetOutputTargetRect  (vp.Get(), TRUE, &fullRect);

    // Default output rate (no frame rate conversion).
    videoContext->VideoProcessorSetStreamOutputRate(
        vp.Get(), 0, D3D11_VIDEO_PROCESSOR_OUTPUT_RATE_NORMAL, FALSE, nullptr);

    _videoDevice         = videoDevice.Detach();
    _videoContext        = videoContext.Detach();
    _videoEnum           = enumerator.Detach();
    _videoProcessor      = vp.Detach();
    _rgba8Texture        = rgba8Tex.Detach();
    _rgba8TextureInternal = rgba8Internal.Detach();
    _rgba8OutputView     = outputView.Detach();
    _nv12Copy            = nv12Copy.Detach();
    _nv12InputView       = nv12InputView.Detach();
    _keyedMutex          = keyedMutex.Detach();
    _sharedHandle        = sharedHandle;

    // Cached event query for GPU completion sync (see _convert_nv12_to_rgba8).
    D3D11_QUERY_DESC qDesc{};
    qDesc.Query     = D3D11_QUERY_EVENT;
    qDesc.MiscFlags = 0;
    ComPtr<ID3D11Query> gpuDoneQuery;
    hr = _d3d11Device->CreateQuery(&qDesc, &gpuDoneQuery);
    if (FAILED(hr)) {
        CI_LOG_W("CreateQuery(EVENT) failed: hr=0x" << std::hex << hr
                 << " — cross-API GPU sync will be unreliable");
    }
    _gpuDoneQuery = gpuDoneQuery.Detach();

    CI_LOG_I("Phase 4: RGBA8 shared via NT handle (VP→internal→shared), keyed mutex ready");
    return true;
}

bool VideoDecoderD3D11::_convert_nv12_to_rgba8() {
    if (!_videoProcessor || !_hwTexture || !_rgba8OutputView || !_nv12Copy || !_nv12InputView) {
        return false;
    }

    // Bail early if the device is in a removed state — every subsequent D3D11
    // call would otherwise fail with misleading codes (E_INVALIDARG, E_OUTOFMEMORY).
    HRESULT removed = _d3d11Device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        CI_LOG_E("D3D11 device removed before VP blt: hr=0x" << std::hex << removed);
        return false;
    }

    // Copy the decoder's NV12 slice into our stable single-slice NV12 copy.
    // NV12 textures have 2 subresources per array slice (Y at 2N, UV at 2N+1).
    // _hwSubresourceIndex from IMFDXGIBuffer is the array slice (0-based).
    D3D11_TEXTURE2D_DESC srcDesc{};
    _hwTexture->GetDesc(&srcDesc);
    if (_hwSubresourceIndex >= srcDesc.ArraySize) {
        CI_LOG_E("Decoder subresource index out of range: idx="
                 << _hwSubresourceIndex << " arraySize=" << srcDesc.ArraySize);
        return false;
    }

    // NV12 multi-array subresource layout in D3D11 is NOT interleaved:
    //   subresource 0 .. N-1   = Y planes of array slices 0 .. N-1
    //   subresource N .. 2N-1  = UV planes of array slices 0 .. N-1
    // The interleaved (Y0, UV0, Y1, UV1, ...) assumption produces a solid-green
    // image because we end up reading the wrong subresources.
    UINT const srcYSub  = _hwSubresourceIndex;
    UINT const srcUVSub = _hwSubresourceIndex + srcDesc.ArraySize;

    _d3d11Context->CopySubresourceRegion(_nv12Copy, 0, 0, 0, 0,
                                          _hwTexture, srcYSub,  nullptr);
    _d3d11Context->CopySubresourceRegion(_nv12Copy, 1, 0, 0, 0,
                                          _hwTexture, srcUVSub, nullptr);

    // Cached input view — created once at init, valid for the lifetime of
    // _nv12Copy. The view sees the latest bytes written via CopySubresourceRegion.
    D3D11_VIDEO_PROCESSOR_STREAM stream{};
    stream.Enable            = TRUE;
    stream.OutputIndex       = 0;
    stream.InputFrameOrField = 0;
    stream.PastFrames        = 0;
    stream.FutureFrames      = 0;
    stream.pInputSurface     = _nv12InputView;

    // VP writes to _rgba8TextureInternal (plain texture, no KEYEDMUTEX).
    HRESULT hr = _videoContext->VideoProcessorBlt(
        _videoProcessor, _rgba8OutputView, 0, 1, &stream);
    if (FAILED(hr)) {
        CI_LOG_E("VideoProcessorBlt failed: hr=0x" << std::hex << hr);
        return false;
    }

    // Promote to the shared texture under keyed mutex. Phase 4 uses key 0 on
    // both sides (D3D11 producer and D3D12 consumer) — simpler than the
    // 0/1 producer/consumer split, which requires both sides to participate
    // every frame or deadlock. Phase 6 will revisit if we need stricter
    // producer/consumer phasing.
    if (_keyedMutex) _keyedMutex->AcquireSync(0, INFINITE);
    _d3d11Context->CopyResource(_rgba8Texture, _rgba8TextureInternal);
    _d3d11Context->Flush();

    // Force GPU completion before handing the mutex off. D3D11's Flush only
    // submits; without this wait, D3D12's separate command queue can read the
    // shared texture before the CopyResource actually executes (returns zeros).
    if (_gpuDoneQuery) {
        _d3d11Context->End(_gpuDoneQuery);
        BOOL done = FALSE;
        while (_d3d11Context->GetData(_gpuDoneQuery, &done, sizeof(done), 0) == S_FALSE) {
            // spin until GPU completes the queued CopyResource + VP blt
        }
    }

    if (_keyedMutex) _keyedMutex->ReleaseSync(0);
    return true;
}

//------------------------------------------------------------------------------
// Phase 3 verification: copy RGBA8 texture back to CPU.
//------------------------------------------------------------------------------
bool VideoDecoderD3D11::readback_rgba8_to_cpu(std::vector<uint8_t>& outPixels,
                                               int32_t& outStride) {
    if (!_rgba8Texture || !_d3d11Device || !_d3d11Context) return false;
    if (_width == 0 || _height == 0) return false;

    // Surface device-removed state early — every subsequent D3D11 call would
    // either fail or return garbage. Cheap to check, surfaces real failures.
    HRESULT removed = _d3d11Device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        CI_LOG_E("D3D11 device removed before readback: hr=0x"
                 << std::hex << removed);
        return false;
    }

    // Lazily create / re-create the staging texture on dimension changes.
    if (!_stagingTexture || _stagingWidth != _width || _stagingHeight != _height) {
        D3D11_TEXTURE2D_DESC stagingDesc{};
        stagingDesc.Width            = _width;
        stagingDesc.Height           = _height;
        stagingDesc.MipLevels        = 1;
        stagingDesc.ArraySize        = 1;
        stagingDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        stagingDesc.SampleDesc.Count = 1;
        stagingDesc.Usage            = D3D11_USAGE_STAGING;
        stagingDesc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;
        stagingDesc.BindFlags        = 0;

        ComPtr<ID3D11Texture2D> staging;
        HRESULT hr = _d3d11Device->CreateTexture2D(&stagingDesc, nullptr, &staging);
        if (FAILED(hr)) {
            CI_LOG_E("CreateTexture2D(staging) failed: hr=0x" << std::hex << hr);
            return false;
        }
        if (_stagingTexture) _stagingTexture->Release();
        _stagingTexture  = staging.Detach();
        _stagingWidth    = _width;
        _stagingHeight   = _height;
    }

    _d3d11Context->CopyResource(_stagingTexture, _rgba8Texture);
    // Flush forces the CopyResource to actually run before Map; without it,
    // some drivers report RowPitch=0 / pData=stale on the subsequent Map,
    // which crashes the per-row memcpy below.
    _d3d11Context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = _d3d11Context->Map(_stagingTexture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || mapped.pData == nullptr || mapped.RowPitch == 0) {
        CI_LOG_E("Map(staging) failed: hr=0x" << std::hex << hr
                 << " pData=" << (mapped.pData ? "non-null" : "NULL")
                 << " rowPitch=" << std::dec << mapped.RowPitch);
        if (SUCCEEDED(hr)) _d3d11Context->Unmap(_stagingTexture, 0);
        return false;
    }

    outStride = static_cast<int32_t>(_width) * 4;
    outPixels.resize(static_cast<size_t>(outStride) * _height);

    // Source row-pitch from GPU may be wider than width*4 (alignment padding).
    // Per-row copy is mandatory, not optional.
    for (uint32_t y = 0; y < _height; ++y) {
        std::memcpy(outPixels.data() + static_cast<size_t>(outStride) * y,
                    static_cast<uint8_t const*>(mapped.pData) +
                        static_cast<size_t>(mapped.RowPitch) * y,
                    static_cast<size_t>(outStride));
    }
    _d3d11Context->Unmap(_stagingTexture, 0);
    return true;
}

bool VideoDecoderD3D11::_configure_rgb32_cpu_output() {
    ComPtr<IMFMediaType> outType;
    HRESULT hr = MFCreateMediaType(&outType);
    if (FAILED(hr)) return false;
    if (FAILED(outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video))) return false;
    if (FAILED(outType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32))) return false;

    hr = _reader->SetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, outType.Get());
    if (FAILED(hr)) {
        CI_LOG_E("SetCurrentMediaType(RGB32) failed: hr=0x" << std::hex << hr
                 << " — asset may be a codec RGB32 can't convert from");
        return false;
    }
    return true;
}

bool VideoDecoderD3D11::_query_frame_metrics() {
    ComPtr<IMFMediaType> actual;
    HRESULT hr = _reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &actual);
    if (FAILED(hr)) return false;
    if (_outputType) _outputType->Release();
    _outputType = actual.Detach();

    // Per MS docs, MF_MT_FRAME_SIZE packs (width << 32) | height into one UINT64.
    UINT64 frameSize = 0;
    hr = _outputType->GetUINT64(MF_MT_FRAME_SIZE, &frameSize);
    if (FAILED(hr)) return false;
    _width  = static_cast<uint32_t>(frameSize >> 32);
    _height = static_cast<uint32_t>(frameSize & 0xFFFFFFFFu);

    // MF_MT_DEFAULT_STRIDE: positive = top-down, negative = bottom-up (per MF docs).
    // Fallback to default 4-byte-aligned packed stride if attribute absent.
    UINT32 stride = 0;
    hr = _outputType->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride);
    if (SUCCEEDED(hr)) {
        _cpuStride = static_cast<int32_t>(stride);
    } else {
        _cpuStride = static_cast<int32_t>(((_width * 32u + 31u) & ~31u) >> 3);
    }

    // MF_MT_FRAME_RATE packs (numerator << 32) | denominator — the AVERAGE rate,
    // not per-frame timestamps. Good enough for presentation gating; VFR content
    // may present a frame one interval early/late.
    UINT64 frameRate = 0;
    hr = _outputType->GetUINT64(MF_MT_FRAME_RATE, &frameRate);
    if (SUCCEEDED(hr)) {
        UINT32 const num = static_cast<UINT32>(frameRate >> 32);
        UINT32 const den = static_cast<UINT32>(frameRate & 0xFFFFFFFFu);
        if (num > 0 && den > 0) {
            _frameDurationSec = static_cast<double>(den) / static_cast<double>(num);
        }
    }
    if (!(_frameDurationSec > 0.0)) _frameDurationSec = 1.0 / 30.0;

    // Duration from the presentation descriptor.
    if (_outputType) {
        PROPVARIANT dur; PropVariantInit(&dur);
        hr = _reader->GetPresentationAttribute(
            MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &dur);
        if (SUCCEEDED(hr)) {
            if (dur.vt == VT_UI8) {
                _durationSec = static_cast<double>(dur.uhVal.QuadPart) * 1e-7;
            }
            PropVariantClear(&dur);
        }
    }

    _cpuPixels.resize(static_cast<size_t>(std::abs(_cpuStride)) * _height);
    return true;
}

bool VideoDecoderD3D11::read_next_frame() {
    if (!_reader) return false;

    DWORD     flags = 0;
    LONGLONG  pts   = 0;
    ComPtr<IMFSample> sample;

    HRESULT hr = _reader->ReadSample(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        0,           // synchronous (no callback configured)
        nullptr,     // actualStreamIndex out
        &flags,      // streamFlags out
        &pts,        // timestamp out
        &sample);    // sample out (may be null on format-change events)

    if (FAILED(hr)) {
        CI_LOG_E("ReadSample failed: hr=0x" << std::hex << hr);
        return false;
    }

    if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
        _eof = true;
    }

    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
        // Re-query metrics and resize our staging buffer; the new type is already
        // current on the reader, so GetCurrentMediaType returns it.
        _query_frame_metrics();
        if (_mode == Mode::CpuRgb32) {
            _cpuPixels.resize(static_cast<size_t>(std::abs(_cpuStride)) * _height);
        }
    }

    if (!sample) {
        // Reader signaled an event without producing a sample (format change or
        // EOS). Caller can keep calling; we'll return false until EOS sets in.
        return false;
    }

    bool ok = false;
    if (_mode == Mode::HardwareNV12) {
        ok = _extract_hw_texture(sample.Get())
          && _convert_nv12_to_rgba8();
    } else {
        ok = _copy_cpu_buffer(sample.Get());
    }

    if (ok) _currentTimeSec = static_cast<double>(pts) * 1e-7;
    return ok;
}

bool VideoDecoderD3D11::seek(double sec) {
    if (!_reader) return false;
    if (sec < 0.0) sec = 0.0;
    if (_durationSec > 0.0 && sec > _durationSec) {
        sec = std::max(0.0, _durationSec - _frameDurationSec);
    }

    // Manual VT_I8 variant (InitPropVariantAsInt64 isn't available in this SDK
    // config), in 100 ns units. This SDK's mfreadwrite.h takes the PROPVARIANT
    // by const reference, not pointer.
    PROPVARIANT pos;
    PropVariantInit(&pos);
    pos.vt             = VT_I8;
    pos.hVal.QuadPart  = static_cast<LONGLONG>(sec * 1e7);
    HRESULT hr = _reader->SetCurrentPosition(GUID_NULL, pos);
    PropVariantClear(&pos);
    if (FAILED(hr)) {
        CI_LOG_E("SetCurrentPosition(" << sec << "s) failed: hr=0x" << std::hex << hr);
        return false;
    }

    _eof = false;
    _currentTimeSec = sec;   // provisional — next read_next_frame()'s PTS is authoritative
    return true;
}

bool VideoDecoderD3D11::_extract_hw_texture(IMFSample* sample) {
    // The sample carries an IMFDXGIBuffer wrapping an ID3D11Texture2D. The texture
    // is one slice of an array owned by the decoder; the slice index advances per
    // frame as reference frames shift through the array.
    ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = sample->ConvertToContiguousBuffer(&buffer);
    if (FAILED(hr)) {
        CI_LOG_E("ConvertToContiguousBuffer failed: hr=0x" << std::hex << hr);
        return false;
    }

    ComPtr<IMFDXGIBuffer> dxgiBuffer;
    hr = buffer->QueryInterface(IID_PPV_ARGS(&dxgiBuffer));
    if (FAILED(hr)) {
        CI_LOG_E("Sample buffer is not IMFDXGIBuffer (hr=0x"
                 << std::hex << hr
                 << ") — D3D11 device not wired to MF?");
        return false;
    }

    // Release previous frame's reference (decoder keeps its own; we just AddRef'd).
    if (_hwTexture) { _hwTexture->Release(); _hwTexture = nullptr; }
    _hwSubresourceIndex = 0;

    hr = dxgiBuffer->GetResource(IID_PPV_ARGS(&_hwTexture));
    if (FAILED(hr) || !_hwTexture) {
        CI_LOG_E("GetResource(IID_ID3D11Texture2D) failed: hr=0x" << std::hex << hr);
        return false;
    }

    UINT sub = 0;
    if (SUCCEEDED(dxgiBuffer->GetSubresourceIndex(&sub))) {
        _hwSubresourceIndex = sub;
    }

    // First-frame verification: log the decoder's NV12 texture descriptor so we
    // can confirm DXGI_FORMAT_NV12, ArraySize≥3, B3D11_BIND_DECODER_RESOURCE.
    static bool s_firstDescLogged = false;
    if (!s_firstDescLogged) {
        D3D11_TEXTURE2D_DESC desc{};
        _hwTexture->GetDesc(&desc);
        CI_LOG_I("[Phase2] NV12 decoder texture desc:"
                 << " format=0x" << std::hex << desc.Format
                 << " arraySize=" << std::dec << desc.ArraySize
                 << " w=" << desc.Width << " h=" << desc.Height
                 << " bindFlags=0x" << std::hex << desc.BindFlags
                 << " miscFlags=0x" << desc.MiscFlags << std::dec
                 << " (expect format=NV12, arraySize>=3,"
                 << " bindFlags & 0x400 (=D3D11_BIND_DECODER_RESOURCE))");
        s_firstDescLogged = true;
    }

    return true;
}

bool VideoDecoderD3D11::_copy_cpu_buffer(IMFSample* sample) {
    ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = sample->ConvertToContiguousBuffer(&buffer);
    if (FAILED(hr)) {
        CI_LOG_E("ConvertToContiguousBuffer failed: hr=0x" << std::hex << hr);
        return false;
    }

    BYTE* data = nullptr;
    DWORD maxLen = 0, curLen = 0;
    hr = buffer->Lock(&data, &maxLen, &curLen);
    if (FAILED(hr) || !data) return false;

    auto const copyBytes = std::min<size_t>(curLen, _cpuPixels.size());
    if (copyBytes > 0) {
        std::memcpy(_cpuPixels.data(), data, copyBytes);
    }
    buffer->Unlock();
    return true;
}

void VideoDecoderD3D11::close() {
    // Release our reference to the most recent decoder texture (MF still owns it).
    if (_hwTexture) { _hwTexture->Release(); _hwTexture = nullptr; }
    _hwSubresourceIndex = 0;

    if (_stagingTexture) { _stagingTexture->Release(); _stagingTexture = nullptr; }
    _stagingWidth = _stagingHeight = 0;

    // Phase 3 + 4: Video Processor resources + shared handle
    if (_rgba8OutputView)     { _rgba8OutputView->Release();     _rgba8OutputView     = nullptr; }
    if (_rgba8Texture)        { _rgba8Texture->Release();        _rgba8Texture        = nullptr; }
    if (_rgba8TextureInternal){ _rgba8TextureInternal->Release();_rgba8TextureInternal= nullptr; }
    if (_nv12InputView)       { _nv12InputView->Release();       _nv12InputView       = nullptr; }
    if (_nv12Copy)            { _nv12Copy->Release();            _nv12Copy            = nullptr; }
    if (_keyedMutex)          { _keyedMutex->Release();          _keyedMutex          = nullptr; }
    if (_sharedHandle)        { CloseHandle(_sharedHandle);      _sharedHandle        = nullptr; }
    if (_gpuDoneQuery)        { _gpuDoneQuery->Release();        _gpuDoneQuery        = nullptr; }
    if (_videoProcessor)  { _videoProcessor->Release();  _videoProcessor  = nullptr; }
    if (_videoEnum)       { _videoEnum->Release();       _videoEnum       = nullptr; }
    if (_videoContext)    { _videoContext->Release();    _videoContext    = nullptr; }
    if (_videoDevice)     { _videoDevice->Release();     _videoDevice     = nullptr; }

    if (_outputType) { _outputType->Release(); _outputType = nullptr; }
    if (_reader) {
        _reader->Flush(MF_SOURCE_READER_ALL_STREAMS);
        _reader->Release();
        _reader = nullptr;
    }
    if (_dxgiManager) { _dxgiManager->Release(); _dxgiManager = nullptr; }
    if (_d3d11Context) { _d3d11Context->Release(); _d3d11Context = nullptr; }
    if (_d3d11Device)  { _d3d11Device->Release();  _d3d11Device  = nullptr; }

    _cpuPixels.clear();
    _cpuPixels.shrink_to_fit();
    _width = _height = 0;
    _cpuStride = 0;
    _durationSec = _currentTimeSec = 0.0;
    _frameDurationSec = 1.0 / 30.0;
    _eof = false;

    if (s_instanceCount > 0) {
        --s_instanceCount;
        if (s_instanceCount == 0) {
            MFShutdown();
        }
    }
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
