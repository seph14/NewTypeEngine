#include "newtype/core/DxGLInterop.h"
#include "cinder/Log.h"
#include "glad/glad.h"

namespace newtype::gl_interop {

//==============================================================================
// Construction / Destruction
//==============================================================================

DxGLInterop::DxGLInterop(ID3D12Device* device, uint32_t width, uint32_t height,
                         DXGI_FORMAT dxFormat, GLint glFormat)
    : _width(width), _height(height), _dxFormat(dxFormat), _glFormat(glFormat) {
    // Ref-count the device to prevent LuisaCompute from destroying it before us
    _device = device;
    _create_resources();
}

DxGLInterop::~DxGLInterop() {
    _destroy_resources();
}

void DxGLInterop::_create_resources() {
    // Initialize our local copy of glad function pointers.
    // We compile glad.c directly into the project (cinder.dll has its own copy
    // that it doesn't export), so we must initialize ours before using any GL calls.
    if (!GLAD_GL_EXT_memory_object) {
        gladLoadGL();
    }

    HRESULT hr;

    //==========================================================================
    // Step 1: Create shared D3D12 texture
    //==========================================================================
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CreationNodeMask     = 1;
    heapProps.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Alignment          = 0;
    texDesc.Width              = _width;
    texDesc.Height             = _height;
    texDesc.DepthOrArraySize   = 1;
    texDesc.MipLevels          = 1;
    texDesc.Format             = _dxFormat;
    texDesc.SampleDesc.Count   = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags              = D3D12_RESOURCE_FLAG_NONE;

    hr = _device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,
        &texDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&_sharedTexture));
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateCommittedResource failed (hr=0x" << std::hex << hr << ")");
        return;
    }

    //==========================================================================
    // Step 2: Create shared handle for the texture
    //==========================================================================
    hr = _device->CreateSharedHandle(
        _sharedTexture.Get(),
        nullptr,       // default security
        GENERIC_ALL,
        nullptr,       // unnamed
        &_sharedHandle);
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateSharedHandle (texture) failed (hr=0x" << std::hex << hr << ")");
        return;
    }

    //==========================================================================
    // Step 3: Get allocation size for GL import
    //==========================================================================
    auto allocInfo = _device->GetResourceAllocationInfo(0, 1, &texDesc);

    //==========================================================================
    // Step 4: Create dedicated command queue (separate from LuisaCompute's)
    //
    // LuisaCompute's DX backend uses an execution thread with internal
    // lastFrame/executedFrame counters. Submitting external commands to its
    // queue corrupts this tracking, causing deadlocks on the next synchronize().
    // Our own queue avoids this entirely.
    //==========================================================================
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    hr = _device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&_copyQueue));
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateCommandQueue failed (hr=0x" << std::hex << hr << ")");
        return;
    }

    hr = _device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&_cmdAlloc));
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateCommandAllocator failed");
        return;
    }

    hr = _device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        _cmdAlloc.Get(),
        nullptr,
        IID_PPV_ARGS(&_cmdList));
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateCommandList failed");
        return;
    }
    _cmdList->Close(); // Start closed; reset before each copy

    //==========================================================================
    // Step 5: Create shared D3D12 fence (imported into GL as semaphore)
    //==========================================================================
    hr = _device->CreateFence(
        0,
        D3D12_FENCE_FLAG_SHARED,
        IID_PPV_ARGS(&_fence));
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateFence failed");
        return;
    }

    hr = _device->CreateSharedHandle(
        _fence.Get(),
        nullptr,
        GENERIC_ALL,
        nullptr,
        &_fenceHandle);
    if (FAILED(hr)) {
        CI_LOG_E("DxGLInterop: CreateSharedHandle (fence) failed");
        return;
    }

    //==========================================================================
    // Step 6: Import shared DX texture into GL via GL_EXT_memory_object_win32
    //==========================================================================
    if (!GLAD_GL_EXT_memory_object || !GLAD_GL_EXT_memory_object_win32) {
        CI_LOG_E("DxGLInterop: GL_EXT_memory_object / GL_EXT_memory_object_win32 not available");
        return;
    }

    glCreateMemoryObjectsEXT(1, &_memoryObject);
    glImportMemoryWin32HandleEXT(
        _memoryObject,
        allocInfo.SizeInBytes,
        GL_HANDLE_TYPE_D3D12_RESOURCE_EXT,
        _sharedHandle);

    //==========================================================================
    // Step 7: Create GL texture backed by imported DX memory
    //==========================================================================
    glCreateTextures(GL_TEXTURE_2D, 1, &_glTextureId);
    glTextureParameteri(_glTextureId, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(_glTextureId, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(_glTextureId, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_glTextureId, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTextureStorageMem2DEXT(
        _glTextureId,
        1,              // mip levels
        _glFormat,      // e.g. GL_RGBA32F
        _width,
        _height,
        _memoryObject,
        0);             // offset

    //==========================================================================
    // Step 8: Import D3D12 fence as GL semaphore
    //==========================================================================
    if (!GLAD_GL_EXT_semaphore || !GLAD_GL_EXT_semaphore_win32) {
        CI_LOG_E("DxGLInterop: GL_EXT_semaphore / GL_EXT_semaphore_win32 not available");
        return;
    }

    glGenSemaphoresEXT(1, &_semaphore);
    glImportSemaphoreWin32HandleEXT(
        _semaphore,
        GL_HANDLE_TYPE_D3D12_FENCE_EXT,
        _fenceHandle);

    // GL has its own reference to the fence handle now
    CloseHandle(_fenceHandle);
    _fenceHandle = nullptr;

    //==========================================================================
    // Step 9: Wrap GL texture in Cinder Texture2d
    //==========================================================================
    _glTexture = ci::gl::Texture2d::create(
        GL_TEXTURE_2D,
        _glTextureId,
        _width,
        _height,
        true);  // doNotDispose — we manage the GL texture ourselves

    _valid = true;
    CI_LOG_I("DxGLInterop: Created " << _width << "x" << _height
        << " shared texture (DXGI=0x" << std::hex << _dxFormat
        << ", GL=0x" << _glFormat << std::dec << ")");
}

void DxGLInterop::_destroy_resources() {
    _valid = false;

    // Release Cinder ref first (doNotDispose=true, so no glDeleteTextures)
    _glTexture.reset();

    // GL resources
    if (_glTextureId)  { glDeleteTextures(1, &_glTextureId);   _glTextureId  = 0; }
    if (_semaphore)    { glDeleteSemaphoresEXT(1, &_semaphore); _semaphore    = 0; }
    if (_memoryObject) { glDeleteMemoryObjectsEXT(1, &_memoryObject); _memoryObject = 0; }

    // DX12 resources (ComPtr auto-releases)
    _cmdList.Reset();
    _cmdAlloc.Reset();
    _copyQueue.Reset();
    _fence.Reset();
    _sharedTexture.Reset();

    // Handles
    if (_sharedHandle) { CloseHandle(_sharedHandle); _sharedHandle = nullptr; }
    if (_fenceHandle)  { CloseHandle(_fenceHandle);  _fenceHandle  = nullptr; }
}

//==============================================================================
// Per-Frame Copy
//==============================================================================

void DxGLInterop::copy_to_gl(ID3D12Resource* source,
                             uint32_t width, uint32_t height) {
    if (!_valid) return;

    // Reset command allocator and command list
    _cmdAlloc->Reset();
    _cmdList->Reset(_cmdAlloc.Get(), nullptr);

    // No explicit resource barriers needed.
    //
    // Our _copyQueue is separate from LuisaCompute's queue. Both the source
    // (LuisaCompute render target) and destination (shared texture) are in
    // D3D12_RESOURCE_STATE_COMMON on our queue (never used by it before).
    // COMMON is implicitly promoted to COPY_SOURCE / COPY_DEST during the
    // copy, then decays back to COMMON after ExecuteCommandLists completes.
    //
    // Caller MUST ensure stream << synchronize() has been called first,
    // so LuisaCompute's queue is idle and all GPU writes are committed.

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource        = source;
    srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource        = _sharedTexture.Get();
    dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    _cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    _cmdList->Close();

    // Execute on OUR dedicated queue (not LuisaCompute's)
    ID3D12CommandList* cmdLists[] = { _cmdList.Get() };
    _copyQueue->ExecuteCommandLists(1, cmdLists);

    // Signal our fence after the copy completes
    ++_fenceValue;
    _copyQueue->Signal(_fence.Get(), _fenceValue);
}

//==============================================================================
// GL-side Synchronization
//==============================================================================

void DxGLInterop::wait_for_copy() {
    if (!_valid || !_semaphore) return;

    // Wait for DX12 fence to reach _fenceValue before GL reads the texture
    GLuint texture = _glTextureId;
    GLenum srcLayout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    glWaitSemaphoreEXT(
        _semaphore,
        0, nullptr,       // no buffer barriers
        1, &texture,      // texture barrier
        &srcLayout);      // layout
}

//==============================================================================
// Resize
//==============================================================================

void DxGLInterop::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height) return;
    _width  = width;
    _height = height;
    _destroy_resources();
    _create_resources();
}

} // namespace newtype::gl_interop
