#include "newtype/core/DxPresent.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "imgui/imgui_impl_dx12.h"
#include <algorithm>

namespace newtype::core {

namespace {
    // Matches the GL path's gl::clear() color (NewTypeEngine::draw).
    const float kBackground[4] = { 0.1f, 0.1f, 0.15f, 1.0f };
}

DxPresent::~DxPresent() {
    destroy();
}

bool DxPresent::init(ci::app::RendererD3d12* renderer) {
    if (!renderer)
        return false;

    ID3D12Device*        device = renderer->getDevice();
    ID3D12CommandQueue*  queue  = renderer->getCommandQueue();
    if (!device || !queue) {
        CI_LOG_E("DxPresent: RendererD3d12 has no device/queue yet");
        return false;
    }

    HRESULT hr;
    for (UINT i = 0; i < ci::app::RendererD3d12::MaxFrameCount; ++i) {
        hr = device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&_cmdAllocators[i]));
        if (FAILED(hr)) {
            CI_LOG_E("DxPresent: CreateCommandAllocator(" << i << ") failed (hr=0x" << std::hex << hr << ")");
            return false;
        }
    }

    hr = device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        _cmdAllocators[0].Get(), nullptr, IID_PPV_ARGS(&_cmdList));
    if (FAILED(hr)) {
        CI_LOG_E("DxPresent: CreateCommandList failed (hr=0x" << std::hex << hr << ")");
        return false;
    }
    _cmdList->Close(); // start closed; reset before each frame

    // Own fence for destroy(): see the member comment in DxPresent.h.
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_destroyFence));
    if (FAILED(hr)) {
        CI_LOG_E("DxPresent: CreateFence failed (hr=0x" << std::hex << hr << ")");
        return false;
    }
    _fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (_fenceEvent == nullptr) {
        CI_LOG_E("DxPresent: CreateEvent for the destroy fence failed");
        return false;
    }
    _queue = queue;

    _renderer = renderer;
    _valid = true;
    CI_LOG_I("DxPresent: initialized on Cinder RendererD3d12 ("
             << renderer->getBufferCount() << " back buffers, vsync="
             << (renderer->isVSyncEnabled() ? "on" : "off") << ")");
    return true;
}

void DxPresent::destroy() {
    if (!_valid && !_renderer)
        return;
    // Drain in-flight command lists before releasing their allocators. Uses
    // our own queue reference + fence: the window-close path (WM_CLOSE)
    // kills Cinder's renderer impl before app cleanup runs — its frame fence
    // is gone by then, so RendererD3d12::waitForGpu is not callable here.
    if (_queue && _destroyFence) {
        const UINT64 v = ++_fenceValue;
        _queue->Signal(_destroyFence.Get(), v);
        if (_destroyFence->GetCompletedValue() < v) {
            _destroyFence->SetEventOnCompletion(v, _fenceEvent);
            WaitForSingleObject(_fenceEvent, INFINITE);
        }
    }
    _cmdList.Reset();
    for (auto& alloc : _cmdAllocators)
        alloc.Reset();
    if (_fenceEvent != nullptr) {
        CloseHandle(_fenceEvent);
        _fenceEvent = nullptr;
    }
    _destroyFence.Reset();
    _queue.Reset();
    _renderer = nullptr;
    _valid = false;
}

void DxPresent::beginUiFrame(float deltaTime, uint32_t width, uint32_t height) {
    if (!_valid || !ImGui::IsUsingD3D12())
        return;

    ImGui_ImplDX12_NewFrame();

    // CinderImGui routes input through event signals (backend-independent),
    // but DisplaySize/DeltaTime are the app's job under D3D12 (auto-render off)
    ImGuiIO& io       = ImGui::GetIO();
    io.DisplaySize    = ImVec2(static_cast<float>(width), static_cast<float>(height));
    io.DeltaTime      = deltaTime > 0.0f ? deltaTime : (1.0f / 60.0f);

    ImGui::NewFrame();
    _uiFrameActive = true;
}

void DxPresent::_closeUiFrameWithoutRendering() {
    // A frame was started but no command list will render it this app frame
    // (mid-resize, list reset failure, modal-dialog guard). Abandon it so the
    // next beginUiFrame() doesn't hit "NewFrame twice without Render".
    if (_uiFrameActive) {
        ImGui::EndFrame();
        _uiFrameActive = false;
    }
}

void DxPresent::presentFrame(ID3D12Resource* source, uint32_t srcWidth, uint32_t srcHeight) {
    if (!_valid)
        return;

    ID3D12Resource* backBuffer = _renderer->getCurrentBackBuffer();
    if (!backBuffer) {
        // Mid-resize; Cinder's finishDraw() still presents. Keep ImGui sane.
        _closeUiFrameWithoutRendering();
        return;
    }

    const UINT frameIndex = _renderer->getCurrentBackBufferIndex();
    // startDraw() already waited on this back buffer's fence, so resetting the
    // per-buffer allocator here is safe (same contract as the Cinder samples).
    _cmdAllocators[frameIndex]->Reset();
    if (FAILED(_cmdList->Reset(_cmdAllocators[frameIndex].Get(), nullptr))) {
        _closeUiFrameWithoutRendering();
        return;
    }

    const D3D12_RESOURCE_DESC bbDesc = backBuffer->GetDesc();
    const UINT bbWidth  = static_cast<UINT>(bbDesc.Width);
    const UINT bbHeight = bbDesc.Height;

    // The back buffer enters the frame in PRESENT state (finishDraw presented
    // it last time around; RendererD3d12 never records anything itself).
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                     = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource     = backBuffer;
    barrier.Transition.Subresource   = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore   = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter    = D3D12_RESOURCE_STATE_RENDER_TARGET;
    _cmdList->ResourceBarrier(1, &barrier);

    // Draws (unlike ClearRenderTargetView) need the RT bound — the ImGui DX12
    // backend assumes the app called OMSetRenderTargets.
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = _renderer->getCurrentRtvHandle();
    _cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    _cmdList->ClearRenderTargetView(rtv, kBackground, 0, nullptr);

    // Source may differ from the back buffer while a render-size override is
    // active (non-perspective projections: equirect 2:1, room-rig atlases) —
    // those present letterboxed (aspect-fit, centered). Same-size sources
    // take the original full-screen path.
    const bool hasSource = source != nullptr;
    const bool imguiReady = ImGui::IsUsingD3D12() && _uiFrameActive;

    if (hasSource) {
        if (imguiReady) {
            // Present the frame as a fullscreen background image through ImGui's
            // own DX12 pipeline. Row-preserving CopyTextureRegion would show the
            // image flipped: Luisa's display target stores the image bottom-up
            // (row 0 = image bottom — the GL path never sees this because Cinder's
            // gl::draw UV convention cancels it), while a D3D12 swap chain scans
            // row 0 out at the TOP. Flipping V here is the single-place fix.
            _blitFrameBackground(source, static_cast<float>(bbWidth),
                                 static_cast<float>(bbHeight),
                                 srcWidth, srcHeight);
        } else {
            static bool sWarned = false;
            if (!sWarned) {
                sWarned = true;
                CI_LOG_E("DxPresent: no active ImGui frame - cannot present the "
                         "render image (showing clear + UI only)");
            }
        }
    }

    // ImGui (background image first, then UI) — same stacking as the GL path
    // (display quad, then UI). Input is already in the event queue;
    // CinderImGui's modal guard maps to ShouldSkipFrame here.
    if (imguiReady && !ImGui::ShouldSkipFrame()) {
        if (ID3D12DescriptorHeap* srvHeap = ImGui::GetD3D12SrvHeap()) {
            ID3D12DescriptorHeap* heaps[] = { srvHeap };
            _cmdList->SetDescriptorHeaps(1, heaps);
        }
        ImGui::Render();
        _uiFrameActive = false;
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), _cmdList.Get());
    } else {
        _closeUiFrameWithoutRendering();
    }

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    _cmdList->ResourceBarrier(1, &barrier);

    _cmdList->Close();

    // Cinder's queue: finishDraw() presents and signals the per-buffer fence
    // right after this, keeping the back-buffer cadence correct.
    ID3D12CommandList* cmdLists[] = { _cmdList.Get() };
    _renderer->getCommandQueue()->ExecuteCommandLists(1, cmdLists);
}

void DxPresent::_blitFrameBackground(ID3D12Resource* source, float width, float height,
                                     uint32_t srcWidth, uint32_t srcHeight) {
    // SRV in CinderImGui's shader-visible heap, fixed slot 1 (slot 0 is the
    // backend's legacy single font-atlas descriptor). Recreated per frame —
    // the display target ping-pongs between two Luisa images, and a CPU-side
    // descriptor write is negligible. Sampling without a source barrier is
    // legal for the same reason the old interop copy was: Luisa's stream is
    // fully synchronized by contract, so the texture is effectively COMMON and
    // implicitly promotes to SRV read on this direct queue.
    ID3D12DescriptorHeap* heap = ImGui::GetD3D12SrvHeap();
    if (!heap)
        return;

    const UINT inc = _renderer->getDevice()->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += inc; // slot 1
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += inc;

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels       = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    _renderer->getDevice()->CreateShaderResourceView(source, &srvDesc, cpu);

    // Flipped V (uv_min=(0,1) at screen top): row 0 of the source is the image
    // bottom, the swap chain scans row 0 out at the top.
    // Letterbox when the source aspect differs from the back buffer (render-
    // size override): fit inside the window, centered, bars take kBackground.
    ImVec2 pMin(0.0f, 0.0f), pMax(width, height);
    if (srcWidth > 0u && srcHeight > 0u &&
        (srcWidth != static_cast<uint32_t>(width) ||
         srcHeight != static_cast<uint32_t>(height))) {
        const float scale = std::min(width / static_cast<float>(srcWidth),
                                     height / static_cast<float>(srcHeight));
        const float dw = static_cast<float>(srcWidth) * scale;
        const float dh = static_cast<float>(srcHeight) * scale;
        pMin = ImVec2((width - dw) * 0.5f, (height - dh) * 0.5f);
        pMax = ImVec2((width + dw) * 0.5f, (height + dh) * 0.5f);
    }
    ImTextureRef tex((ImTextureID)(intptr_t)gpu.ptr);
    ImGui::GetBackgroundDrawList()->AddImage(
        tex, pMin, pMax,
        ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
}

} // namespace newtype::core
