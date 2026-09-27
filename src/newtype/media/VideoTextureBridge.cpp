#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/VideoTextureBridge.h"
#include "newtype/media/NativeTextureDesc.h"

#include <d3d12.h>
#include <dxgi1_4.h>

namespace newtype::media {

VideoTextureBridge::~VideoTextureBridge() {
    close();
}

bool VideoTextureBridge::init(luisa::compute::Device& luDevice,
                               ID3D12Device* d3d12Device,
                               void* sharedNtHandle,
                               uint32_t width, uint32_t height,
                               IDXGIKeyedMutex* keyedMutex) noexcept {
    if (!d3d12Device || !sharedNtHandle || width == 0 || height == 0) {
        CI_LOG_E("VideoTextureBridge::init: invalid arguments");
        return false;
    }

    // Open the NT handle on Luisa's D3D12 device. Both devices must be on the
    // same physical GPU — the decoder verifies this via LUID match at open().
    ID3D12Resource* raw = nullptr;
    HRESULT hr = d3d12Device->OpenSharedHandle(sharedNtHandle, IID_PPV_ARGS(&raw));
    if (FAILED(hr)) {
        CI_LOG_E("VideoTextureBridge: OpenSharedHandle failed: hr=0x" << std::hex << hr);
        return false;
    }
    _d3d12Resource = raw;
    _keyedMutex    = keyedMutex;
    _width         = width;
    _height        = height;

    // Wrap as Luisa Image<float> via NativeResourceExt + vendored NativeTextureDesc.
    // The texture starts (and stays) in COMMON state — see Phase 4 invariant in
    // the implementation plan; Luisa's barrier planner treats initState as the
    // canonical state for the wrap.
    auto* ext = luDevice.extension<luisa::compute::NativeResourceExt>();
    if (!ext) {
        CI_LOG_E("VideoTextureBridge: device has no NativeResourceExt");
        return false;
    }

    NativeTextureDesc nativeDesc{
        /*initState=*/D3D12_RESOURCE_STATE_COMMON,
        /*custom_format=*/DXGI_FORMAT_R8G8B8A8_UNORM,
        /*allowUav=*/false
    };
    _image = ext->create_native_image<float>(
        _d3d12Resource, width, height,
        luisa::compute::PixelStorage::BYTE4, 1u, &nativeDesc);

    _valid = true;
    CI_LOG_D("VideoTextureBridge: " << width << "x" << height
             << " Luisa Image<float> wrapped from D3D12 shared resource");
    return true;
}

void VideoTextureBridge::close() noexcept {
    _valid = false;
    // Image<float> destructor releases the Luisa side of the wrap; it does
    // not free the underlying ID3D12Resource. We release that explicitly
    // after the wrap is gone to avoid Luisa touching a freed resource.
    _image = luisa::compute::Image<float>{};
    if (_d3d12Resource) {
        _d3d12Resource->Release();
        _d3d12Resource = nullptr;
    }
    _keyedMutex = nullptr;
    _width = _height = 0;
}

void VideoTextureBridge::acquire_for_read() noexcept {
    if (_keyedMutex) _keyedMutex->AcquireSync(0, INFINITE);
}

void VideoTextureBridge::release_to_producer() noexcept {
    if (_keyedMutex) _keyedMutex->ReleaseSync(0);
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
