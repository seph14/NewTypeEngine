#pragma once

// Cinder BEFORE Windows/D3D headers
#include "cinder/gl/Texture.h"

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

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <luisa/luisa-compute.h>
#include <stdexcept>

namespace newtype::gl_interop {

using Microsoft::WRL::ComPtr;

/**
 * @brief DX12-OpenGL interop via GL_EXT_memory_object_win32
 *
 * Creates a shared D3D12 texture, imports it into GL, and provides
 * a per-frame copy path: LuisaCompute render target -> shared texture -> GL display.
 *
 * Uses GL_EXT_memory_object_win32 for texture sharing and
 * GL_EXT_semaphore_win32 for D3D12 fence -> GL semaphore synchronization.
 */
class DxGLInterop {
public:
    DxGLInterop(ID3D12Device* device, uint32_t width, uint32_t height,
                DXGI_FORMAT dxFormat = DXGI_FORMAT_R32G32B32A32_FLOAT,
                GLint glFormat = GL_RGBA32F);
    ~DxGLInterop();

    // Non-copyable, non-movable
    DxGLInterop(const DxGLInterop&) = delete;
    DxGLInterop& operator=(const DxGLInterop&) = delete;
    DxGLInterop(DxGLInterop&&) = delete;
    DxGLInterop& operator=(DxGLInterop&&) = delete;

    /**
     * @brief Copy from a DX12 resource to the shared texture and signal fence
     * @param source Source DX12 resource (from LuisaCompute image.native_handle())
     *
     * Uses a dedicated command queue (separate from LuisaCompute's) to avoid
     * corrupting its internal lastFrame/executedFrame tracking.
     * Must be called AFTER stream << synchronize() to ensure source data is ready.
     */
    void copy_to_gl(ID3D12Resource* source, uint32_t width, uint32_t height);

    /**
     * @brief GL side: wait for the DX12 copy to complete via semaphore
     * Call before rendering with the GL texture in Cinder
     */
    void wait_for_copy();

    /** @brief The GL texture for Cinder display (imported from shared DX memory) */
    [[nodiscard]] ci::gl::Texture2dRef gl_texture() const { return _glTexture; }

    [[nodiscard]] explicit operator bool() const { return _valid; }

    /** @brief Resize: destroys and recreates all shared resources */
    void resize(uint32_t width, uint32_t height);

private:
    void _create_resources();
    void _destroy_resources();

    // DX12 resources
    ComPtr<ID3D12Device>              _device;
    ComPtr<ID3D12CommandQueue>        _copyQueue;    // separate queue — never touches LuisaCompute's queue
    ComPtr<ID3D12Resource>            _sharedTexture;
    ComPtr<ID3D12CommandAllocator>    _cmdAlloc;
    ComPtr<ID3D12GraphicsCommandList> _cmdList;
    ComPtr<ID3D12Fence>               _fence;
    HANDLE                            _sharedHandle = nullptr;
    HANDLE                            _fenceHandle  = nullptr;
    uint64_t                          _fenceValue   = 0;

    // GL resources
    GLuint                 _memoryObject = 0;
    GLuint                 _semaphore    = 0;
    GLuint                 _glTextureId  = 0;
    ci::gl::Texture2dRef   _glTexture;

    // Config
    uint32_t    _width  = 0;
    uint32_t    _height = 0;
    DXGI_FORMAT _dxFormat;
    GLint       _glFormat;
    bool        _valid = false;
};

} // namespace newtype::gl_interop
