#pragma once

// Version macros BEFORE any Windows/D3D header: wrl/client.h compiles AsAgile
// (needs AGILEREFERENCE_DEFAULT from combaseapi.h) only when NTDDI_VERSION is
// Win8.1+, and combaseapi.h itself gates that enum on NTDDI_VERSION at ITS
// include time — so the bump must precede the first Windows header pull
// (cinder/app/RendererD3d12.h's chain includes windows.h; the project-wide
// default is Win7 / 0x06010000).
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

#include "cinder/app/RendererD3d12.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

namespace newtype::core {

using Microsoft::WRL::ComPtr;

/**
 * @brief Presents LuisaCompute output through Cinder's RendererD3d12 swap chain.
 *
 * Replaces the DxGLInterop shared-heap + GL-semaphore path when the app runs
 * with Cinder's D3D12 renderer: the Luisa device adopts Cinder's ID3D12Device
 * (DirectXDeviceConfigExt), so the per-frame handoff is a plain
 * CopyTextureRegion into the current back buffer on Cinder's command queue —
 * no cross-API fence, no GL driver in the present path.
 *
 * ImGui renders on top of the image in the same command list (CinderImGui
 * force-disables auto-render under D3D12; this class drives the manual
 * ImGui_ImplDX12 flow documented in CinderImGui.h).
 */
class DxPresent {
public:
    DxPresent() = default;
    ~DxPresent();

    // Non-copyable, non-movable
    DxPresent(const DxPresent&) = delete;
    DxPresent& operator=(const DxPresent&) = delete;
    DxPresent(DxPresent&&) = delete;
    DxPresent& operator=(DxPresent&&) = delete;

    /**
     * @brief Create command objects on the renderer's device.
     * @param renderer Cinder's D3D12 renderer (not owned — outlives this class
     *                 via the engine's cleanup order)
     */
    bool init(ci::app::RendererD3d12* renderer);

    /** @brief Idle the GPU (renderer fence) and release command objects */
    void destroy();

    /**
     * @brief CPU side: start an ImGui frame (DX12 backend has no auto-render).
     * Call once at the top of the app's update(), before any UI code runs.
     * Kept alive even inside modal loops (mirrors the GL auto-render guard,
     * which skips rendering but never skips NewFrame).
     */
    void beginUiFrame(float deltaTime, uint32_t width, uint32_t height);

    /**
     * @brief GPU side: copy the ready display target into the current back
     * buffer, render ImGui on top, execute on Cinder's queue. Cinder's
     * finishDraw() presents immediately after.
     *
     * Same contract as DxGLInterop::copy_to_gl: the source must belong to a
     * fully synchronized Luisa stream (Renderer::endFrame only hands over
     * frames whose GPU work Pipeline::beginFrame already synced).
     * A null/mismatched source falls back to a clear + ImGui so flip-discard
     * never shows garbage.
     */
    void presentFrame(ID3D12Resource* source, uint32_t srcWidth, uint32_t srcHeight);

    [[nodiscard]] bool valid() const noexcept { return _valid; }
    [[nodiscard]] explicit operator bool() const noexcept { return _valid; }

private:
    void _closeUiFrameWithoutRendering();  // keep ImGui state consistent when no command list runs
    void _blitFrameBackground(ID3D12Resource* source, float width, float height,
                              uint32_t srcWidth, uint32_t srcHeight); // flipped-V image via ImGui background draw list; letterboxed (aspect-fit) when the source size differs from the back buffer

    ci::app::RendererD3d12*             _renderer = nullptr; // not owned
    // Own queue reference + fence for destroy(): the window-close path kills
    // Cinder's renderer impl BEFORE app cleanup runs (RendererImplD3d12::kill
    // resets the frame fence this class used to wait on via
    // RendererD3d12::waitForGpu). Holding the queue and fencing ourselves
    // keeps teardown safe regardless of renderer state.
    ComPtr<ID3D12CommandQueue>          _queue;
    ComPtr<ID3D12Fence>                 _destroyFence;
    HANDLE                              _fenceEvent = nullptr;
    UINT64                              _fenceValue = 0;
    ComPtr<ID3D12CommandAllocator>      _cmdAllocators[ci::app::RendererD3d12::MaxFrameCount];
    ComPtr<ID3D12GraphicsCommandList>   _cmdList;
    bool _valid = false;
    bool _uiFrameActive = false; // an ImGui frame was started by beginUiFrame() this app frame
};

} // namespace newtype::core
