#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

// Cinder BEFORE Windows/D3D headers (see CLAUDE.md, DxGLInterop.h for the rule).
#include "cinder/Log.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef NTDDI_VERSION
#undef NTDDI_VERSION
#endif
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0A00
#define NTDDI_VERSION 0x0A00000A

#include <cstdint>
#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/native_resource_ext.hpp>

struct ID3D11Texture2D;
struct ID3D12Device;
struct ID3D12Resource;
struct IDXGIKeyedMutex;

namespace newtype::media {

//
// Wraps a D3D12-shared video texture (opened from the decoder's NT handle) as
// a Luisa Image<float> that DX kernels can sample. Coordinates cross-API sync
// via the keyed mutex acquired from the decoder.
//
class VideoTextureBridge {
public:
    VideoTextureBridge() = default;
    ~VideoTextureBridge();

    VideoTextureBridge(VideoTextureBridge const&) = delete;
    VideoTextureBridge& operator=(VideoTextureBridge const&) = delete;
    VideoTextureBridge(VideoTextureBridge&&) = delete;
    VideoTextureBridge& operator=(VideoTextureBridge&&) = delete;

    // Opens the shared NT handle on the Luisa D3D12 device and wraps the
    // resulting ID3D12Resource as a Luisa Image<float>. The keyed mutex is
    // borrowed (non-owning) from the decoder — both sides use key 0 in v1
    // (see [[Phase 4]] commentary in the implementation plan).
    bool init(luisa::compute::Device& luDevice,
              struct ID3D12Device* d3d12Device,
              void* sharedNtHandle,
              uint32_t width, uint32_t height,
              struct IDXGIKeyedMutex* keyedMutex) noexcept;

    void close() noexcept;

    // CPU-side keyed mutex coordination. Decoder releases the mutex after its
    // GPU work completes (it polls a D3D11 query); acquire here blocks until
    // the decoder is done. Release after the consuming Luisa stream has
    // synchronized.
    void acquire_for_read() noexcept;
    void release_to_producer() noexcept;

    [[nodiscard]] luisa::compute::Image<float>& image() noexcept { return _image; }
    [[nodiscard]] uint32_t width()  const noexcept { return _width; }
    [[nodiscard]] uint32_t height() const noexcept { return _height; }
    [[nodiscard]] bool     valid()  const noexcept { return _valid; }
    [[nodiscard]] explicit operator bool() const noexcept { return _valid; }

private:
    struct ID3D12Resource*  _d3d12Resource = nullptr;
    struct IDXGIKeyedMutex* _keyedMutex    = nullptr;   // non-owning
    luisa::compute::Image<float> _image;
    uint32_t                _width         = 0;
    uint32_t                _height        = 0;
    bool                    _valid         = false;
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
