#include "newtype/core/Renderer.h"
#include "newtype/util/Recorder.h"
#include "newtype/core/ShaderManager.h"
#include "cinder/Log.h"
#include <algorithm>
#include "cinder/CinderAssert.h"
#include <luisa/luisa-compute.h>
#include <iostream>

// CUDA headers only needed for CUDA backend path
#if NT_ALLOW_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#endif

// DX12 headers only needed for DX backend path
#include "newtype/core/DxGLInterop.h"
#include "newtype/core/DxPresent.h"
#include <luisa/backends/ext/dx_config_ext.h>
#include "newtype/util/CompileProfiler.h"
#include <spdlog/sinks/basic_file_sink.h>

namespace newtype::core {
using namespace luisa;

#if NT_PROFILING
namespace {
    newtype::util::CompileProfiler g_compile_profiler;
}
#endif

namespace {
    // Holds a COM reference to an adopted Cinder ID3D12Device. Luisa's Device
    // is static and destructs at process exit — potentially AFTER the Cinder
    // renderer (the device's owner) is gone. Luisa adopts the device
    // non-owning (DirectXDeviceConfigExt), so this ref keeps the object alive
    // until Luisa's runtime has finished releasing its child objects. Must be
    // declared BEFORE Renderer's static members so it destructs after them.
    Microsoft::WRL::ComPtr<ID3D12Device> g_AdoptedDeviceRef;
}

// Define static members
luisa::compute::Device   Renderer::_device;
luisa::compute::Context* Renderer::_context = nullptr;
luisa::compute::Stream   Renderer::_stream;

Renderer::Renderer(uint32_t width, uint32_t height, Backend backend, bool enableValidation,
                   ci::app::RendererD3d12* dx12Renderer)
    : _width (width),
      _height(height),
      _type  (TextureType::Float32),
      _backend(backend),
      _enableValidation(enableValidation),
      _dx12Renderer(dx12Renderer),
      _presentMode(dx12Renderer ? PresentMode::Dx12 : PresentMode::Gl) {

    initLuisaContext();
    if (_presentMode == PresentMode::Dx12) {
        _dxPresent = std::make_unique<DxPresent>();
        if (!_dxPresent->init(_dx12Renderer))
            CI_LOG_E("DxPresent initialization failed - no frames will be presented");
    }
    updateRenderSize();
    const char* backendName = (_backend == Backend::DirectX) ? "DX12" : "CUDA";
    const char* presentName = (_presentMode == PresentMode::Dx12) ? "D3D12 swap chain" : "GL interop";
    CI_LOG_I("NewTypeRenderer: Created " << width << "x" << height << " renderer ("
             << backendName << ", " << presentName << ")");
}

Renderer::Renderer(ci::ivec2 size, Backend backend, bool enableValidation,
                   ci::app::RendererD3d12* dx12Renderer)
    : _width(size.x),
      _height (size.y),
      _type   (TextureType::Float32),
      _backend(backend),
      _enableValidation(enableValidation),
      _dx12Renderer(dx12Renderer),
      _presentMode(dx12Renderer ? PresentMode::Dx12 : PresentMode::Gl) {

    initLuisaContext();
    if (_presentMode == PresentMode::Dx12) {
        _dxPresent = std::make_unique<DxPresent>();
        if (!_dxPresent->init(_dx12Renderer))
            CI_LOG_E("DxPresent initialization failed - no frames will be presented");
    }
    updateRenderSize();
    const char* backendName = (_backend == Backend::DirectX) ? "DX12" : "CUDA";
    const char* presentName = (_presentMode == PresentMode::Dx12) ? "D3D12 swap chain" : "GL interop";
    CI_LOG_I("NewTypeRenderer: Created " << size.x << "x" << size.y << " renderer ("
             << backendName << ", " << presentName << ")");
}

Renderer::~Renderer() {
    // Drains Cinder's queue + releases command objects before Luisa teardown
    _dxPresent.reset();
    _recorder.reset();
    _captureFbo.reset();
    destroyFrameResources();

    if (_context) {
        delete _context;
        _context = nullptr;
    }
}

void Renderer::initLuisaContext() {
    // Mirror Luisa's runtime log (device errors, assert backtraces with
    // file:line before abort()) into a file. The default console sink is lost
    // in windowed runs and in stderr redirects (Cinder's console support
    // reopens stdout); err-level messages flush before abort so the assert
    // trail survives crashes.
    static bool sLuisaFileSinkAdded = false;
    if (!sLuisaFileSinkAdded) {
        sLuisaFileSinkAdded = true;
        try {
            luisa::detail::default_logger_add_sink(
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                    "build/luisa_log.txt", /*truncate=*/true));
        } catch (...) {
            CI_LOG_W("Could not attach Luisa file log sink (build/luisa_log.txt)");
        }
    }

    // Get the full executable path for LuisaCompute Context
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);

    // Initialize LuisaCompute with full executable path
    _context = new (std::nothrow) compute::Context(exePath);
    CI_ASSERT_MSG(_context != nullptr, "Failed to create LuisaCompute Context");

    // Select backend based on configuration
#if NT_ALLOW_CUDA
    const char* backendName = (_backend == Backend::DirectX) ? "dx" : "cuda";
#else
    const char* backendName = "dx";
    _backend = Backend::DirectX;
#endif

#if NT_ENABLE_VALIDATION
    // Opt-in LC validation layer (Config.h); the compile profiler bound
    // below works without it.
    _enableValidation = true;
#endif

    try {
        luisa::compute::DeviceConfig config{};
        config.inqueue_buffer_limit = false;
#if NT_PROFILING
        config.profiler = &g_compile_profiler;
#endif

        // Dx12 present mode: Luisa adopts Cinder's ID3D12Device (its
        // production external-device path, used by the Unreal integration).
        // Single device → the per-frame handoff to the swap chain is a plain
        // CopyTextureRegion on Cinder's queue (no shared heap, no GL fence).
        struct CinderDeviceExt final : public luisa::compute::DirectXDeviceConfigExt {
            ci::app::RendererD3d12* renderer = nullptr;
            [[nodiscard]] luisa::optional<ExternalDevice> CreateExternalDevice() noexcept override {
                if (ID3D12Device* device = renderer->getDevice()) {
                    ExternalDevice ext{};
                    ext.device  = device;
                    ext.adapter = nullptr; // backend resolves by device LUID
                    ext.factory = nullptr; // backend creates its own factory
                    return ext;
                }
                return {};
            }
        };
        if (_presentMode == PresentMode::Dx12 && _backend == Backend::DirectX) {
            auto ext = luisa::make_unique<CinderDeviceExt>();
            ext->renderer = _dx12Renderer;
            config.extension = std::move(ext);
        }

        _device = _context->create_device(backendName, &config, _enableValidation);
        if (_device) {
            _stream = _device.create_stream();
            ShaderManager::setDevice(&_device);
            if (_presentMode == PresentMode::Dx12) {
                // Keep the adopted device alive past Luisa's static teardown
                g_AdoptedDeviceRef = _dx12Renderer->getDevice();
                CI_LOG_I("LuisaCompute " << backendName
                    << " backend initialized on Cinder RendererD3d12 device");
            } else {
                CI_LOG_I("LuisaCompute " << backendName << " backend initialized successfully");
            }
        } else {
            CI_LOG_E("Failed to create " << backendName << " device");
            return;
        }
    } catch (const std::exception &e) {
        CI_LOG_EXCEPTION("Exception creating device", e);
    } catch (...) {
        CI_LOG_E("Unknown exception creating device");
    }
}

bool Renderer::appendInteropTexture(std::string_view name, const ci::uvec2 size, ci::gl::Texture::Format& type) {
    string strname = string(name);
    if (_sharedTextures.find(strname) != _sharedTextures.end())
        return false;

    _sharedTextures[strname] = std::move(createFrameInterop(ci::uvec2(_width, _height), type));
    return true;
}

[[nodiscard]] const ci::gl::Texture2dRef Renderer::getInteropTex(std::string_view name) {
    string strname = string(name);
    auto it = _sharedTextures.find(strname);
    if (it  == _sharedTextures.end())
        return nullptr;
    return it->second->texture;
}

[[nodiscard]] const luisa::compute::Image<float>& Renderer::getInteropImage(std::string_view name) {
    string strname = string(name);
    // todo: throw error
    auto it = _sharedTextures.find(strname);
    return it->second->render_target;
}


void Renderer::initialize(TextureType type) {
    _type = type;
    createFrameResources(type);

    // Create visual recorder for screenshot/video capture
    // (GL readback only under the GL present path; Dx12 downloads via Luisa.
    // Sized to the PRESENT dims — render-res capture while a size override
    // is active, window-res otherwise.)
    _recorder = std::make_unique<util::VisualRecorder>(
        ci::ivec2(_presentWidth, _presentHeight), _presentMode == PresentMode::Gl);
}

FrmRscPtr Renderer::createFrameInterop(ci::uvec2 size, ci::gl::Texture2d::Format& fmt) {
    auto ciFmt = fmt.getInternalFormat();
    auto luFmt = luisa::compute::PixelStorage::FLOAT4;
    if (ciFmt == GL_RGBA8)
        luFmt = luisa::compute::PixelStorage::BYTE4;
    else if(ciFmt == GL_RGBA16F)
        luFmt = luisa::compute::PixelStorage::HALF4;

    auto frm = std::make_unique<FrameResource>();

    // Create LuisaCompute Image as render target
    frm->render_target = _device.create_image<float>(
        luFmt, luisa::uint2(size.x, size.y));

    if (_backend == Backend::DirectX && _presentMode == PresentMode::Gl) {
        // DX12-GL path: create shared texture + GL import via DxGLInterop
        // (Dx12-present mode needs none of this — DxPresent copies on-device)
        auto dxDevice = static_cast<ID3D12Device*>(_device.native_handle());

        // Map GL format to DXGI format
        DXGI_FORMAT dxFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
        if (ciFmt == GL_RGBA16F)       dxFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
        else if (ciFmt == GL_RGBA8)    dxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

        frm->dx_interop = std::make_unique<gl_interop::DxGLInterop>(
            dxDevice, size.x, size.y, dxFormat, ciFmt);
        frm->texture = frm->dx_interop->gl_texture();

        CI_LOG_D("Created DX12-GL interop: " << size.x << "x" << size.y);
    }
#if NT_ALLOW_CUDA
    else {
        // CUDA path: create Cinder texture + register with CUDA
        frm->texture = ci::gl::Texture2d::create(size.x, size.y, fmt);
        frm->cuda_interop = std::make_unique<gl_interop::LuisaGLInterop>(
            frm->texture, cudaGraphicsMapFlagsWriteDiscard);

        CI_LOG_D("Created CUDA-GL texture=" << frm->texture->getId() <<
                 "<->" << frm->render_target.handle());
    }
#endif

    return frm;
}

void Renderer::createFrameResources(TextureType type) {
    static const vector<uint32_t> ciFmt = {
        GL_RGBA32F,
        GL_RGBA16F,
        GL_RGBA8
    };

    const auto glFmt  = ciFmt[static_cast<int>(type)];

    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        if (type == TextureType::Int8) {
            // RGBA8 dual-path: FLOAT4 render target (no interop) + BYTE4 display target (with interop).
            // Render target is at RENDER resolution (display * renderScale) — the upscaler
            // stage reads it and produces the display-res HDR frame the tonemap consumes.
            auto frm = std::make_unique<FrameResource>();

            // 1. FLOAT4 render target — pipeline renders into this in full precision
            frm->render_target = _device.create_image<float>(
                luisa::compute::PixelStorage::FLOAT4, luisa::uint2(_renderWidth, _renderHeight));

            // 2. BYTE4 display target — tone-mapped output for the interop
            //    copy. PRESENT dims: window-res normally, render-res while a
            //    render-size override is active (recorder then captures at
            //    render resolution; DxPresent letterboxes the preview).
            frm->display_target = _device.create_image<float>(
                luisa::compute::PixelStorage::BYTE4, luisa::uint2(_presentWidth, _presentHeight));

            // 3. GL_RGBA8 texture + interop for display_target
            //    (Dx12-present mode: display_target stays a plain Luisa image;
            //    DxPresent copies it into the swap chain back buffer)
            if (_backend == Backend::DirectX && _presentMode == PresentMode::Gl) {
                auto dxDevice = static_cast<ID3D12Device*>(_device.native_handle());
                frm->dx_display_interop = std::make_unique<gl_interop::DxGLInterop>(
                    dxDevice, _presentWidth, _presentHeight, DXGI_FORMAT_R8G8B8A8_UNORM, GL_RGBA8);
                frm->display_texture = frm->dx_display_interop->gl_texture();
            }
#if NT_ALLOW_CUDA
            else {
                ci::gl::Texture2d::Format rgba8Fmt;
                rgba8Fmt.internalFormat(GL_RGBA8).dataType(GL_UNSIGNED_BYTE)
                    .wrap(GL_CLAMP_TO_EDGE)
                    .minFilter(GL_LINEAR).magFilter(GL_LINEAR);
                frm->display_texture = ci::gl::Texture2d::create(_presentWidth, _presentHeight, rgba8Fmt);
                frm->cuda_display_interop = std::make_unique<gl_interop::LuisaGLInterop>(
                    frm->display_texture, cudaGraphicsMapFlagsWriteDiscard);
            }
#endif

            // currentTexture() returns this — point to the RGBA8 display texture
            frm->texture = frm->display_texture;

            _frames[i] = std::move(frm);
            CI_LOG_D("Created RGBA8 dual-path frame " << i);
        } else {
            // Float32/Float16 path: single resource with interop on render_target
            ci::gl::Texture2d::Format fmt;
            fmt.internalFormat(glFmt).dataType(GL_FLOAT)
                .wrap(GL_CLAMP_TO_EDGE)
                .minFilter(GL_LINEAR).magFilter(GL_LINEAR);
            _frames[i] = std::move(createFrameInterop(ci::uvec2(_width, _height), fmt));
        }
    }

    CI_LOG_D("NewTypeRenderer: Created " << FRAME_COUNT << " frame resources (type=" << static_cast<int>(type) << ")");

    // Create capture FBO wrapping the display texture for PBO readback
    // (GL present mode only — Dx12 recorder reads the Luisa display target)
    if (_presentMode == PresentMode::Gl) {
        auto& displayTex = _frames[0]->texture;
        auto fboFmt = ci::gl::Fbo::Format()
            .attachment(GL_COLOR_ATTACHMENT0, displayTex)
            .disableDepth();
        _captureFbo = ci::gl::Fbo::create(_presentWidth, _presentHeight, fboFmt);
    }
}

void Renderer::destroyFrameResources() {
    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        if (_frames[i]) {
            // interop is automatically cleaned up by unique_ptr
            // render_target is automatically cleaned up by Image destructor
            // texture is reference-counted by Cinder
            _frames[i].reset();
        }
    }
}

void Renderer::updateRenderSize() noexcept {
    // Round to nearest, never below 1 pixel. Scale is clamped defensively;
    // the Pipeline owns the "renderScale < 1 requires an active upscaler"
    // invariant (docs/upscaling_feasibility_report.md Phase 0).
    _renderScale  = std::clamp(_renderScale, 1.0f / 3.0f, 1.0f);
    if (_renderSizeOverride) {
        // Explicit decoupled resolution (non-perspective projections):
        // renderScale is meaningless here — the override wins.
        _renderWidth  = std::max(1u, _renderSizeOverride->x);
        _renderHeight = std::max(1u, _renderSizeOverride->y);
    } else {
        uint32_t rw = std::max(1u, static_cast<uint32_t>(_width  * _renderScale + 0.5f));
        uint32_t rh = std::max(1u, static_cast<uint32_t>(_height * _renderScale + 0.5f));
        // DLSS feature creation fails on odd render dimensions (the network's
        // 2-pixel phase grid — NVSDK create returns an error and the backends
        // would sit in the failed state). Align down to even whenever the
        // render size is decoupled from the window; FSR and the native path
        // are parity-agnostic, so the uniform rule costs them at most 1 px.
        if (_renderScale < 1.0f) {
            rw = std::max(2u, rw & ~1u);
            rh = std::max(2u, rh & ~1u);
        }
        _renderWidth  = rw;
        _renderHeight = rh;
    }
    // Display/capture target: render-res while decoupled (recorder captures
    // at render resolution; the present path letterboxes), window-res else.
    if (_renderSizeOverride) {
        _presentWidth  = _renderWidth;
        _presentHeight = _renderHeight;
    } else {
        _presentWidth  = _width;
        _presentHeight = _height;
    }
}

void Renderer::setRenderSizeOverride(const std::optional<luisa::uint2>& size) {
    std::optional<luisa::uint2> sane;
    if (size) sane = luisa::uint2{ std::max(1u, size->x), std::max(1u, size->y) };
    const bool same = sane.has_value() == _renderSizeOverride.has_value() &&
        (!sane || (sane->x == _renderSizeOverride->x &&
                   sane->y == _renderSizeOverride->y));
    if (same)
        return;

    // Same safety envelope as resize()/setRenderScale(): sync in-flight work
    // before destroying frame resources.
    _stream << luisa::compute::synchronize();
    _renderSizeOverride = sane;
    _renderScale = 1.0f; // mutually exclusive with scaled rendering
    updateRenderSize();

    destroyFrameResources();
    createFrameResources(_type);
    _presentSkip = true;
    if (_recorder) {
        _recorder.reset();
        _recorder = std::make_unique<util::VisualRecorder>(
            ci::ivec2(_presentWidth, _presentHeight), _presentMode == PresentMode::Gl);
    }

    CI_LOG_I("NewTypeRenderer: Render size override "
        << (sane ? std::to_string(sane->x) + "x" + std::to_string(sane->y) : std::string("off"))
        << " -> render " << _renderWidth << "x" << _renderHeight
        << ", present/capture " << _presentWidth << "x" << _presentHeight
        << ", display " << _width << "x" << _height);
}

void Renderer::setRenderScale(float scale) {
    const float clamped = std::clamp(scale, 1.0f / 3.0f, 1.0f);
    if (clamped == _renderScale)
        return;

    // Same safety envelope as resize(): sync in-flight work before
    // destroying frame resources.
    _stream << luisa::compute::synchronize();
    _renderScale = clamped;
    updateRenderSize();

    destroyFrameResources();
    createFrameResources(_type);
    _presentSkip = true;

    CI_LOG_I("NewTypeRenderer: Render scale " << _renderScale
        << " -> render " << _renderWidth << "x" << _renderHeight
        << ", display " << _width << "x" << _height);
}

void Renderer::beginUiFrame(float deltaTime, uint32_t width, uint32_t height) {
    if (_dxPresent)
        _dxPresent->beginUiFrame(deltaTime, width, height);
}

void Renderer::updateRecorder() {
    if (!_recorder)
        return;
    if (_presentMode == PresentMode::Dx12) {
        if (_type == TextureType::Int8) {
            // Downloads the last-presented (ready) frame. Must run BEFORE this
            // frame's render()/beginFrame(): the blocking synchronize then only
            // waits for work beginFrame would have waited for anyway.
            _recorder->updateDx(_frames[_ready_frame_index]->display_target);
        }
    } else {
        _recorder->update(_captureFbo);
    }
}

FrameResource& Renderer::beginFrame() {
    _current_frame_index = (_current_frame_index + 1) % FRAME_COUNT;
    return *_frames[_current_frame_index];
}

void Renderer::markFrameReady() {
    _ready_frame_index = _current_frame_index;// (_current_frame_index + FRAME_COUNT - 1) % FRAME_COUNT;
}

void Renderer::endFrame() {
    // First endFrame after frame-resource recreation: _ready_frame_index is
    // stale and may alias the frame render() is writing right now — copying
    // it on the interop queue would race the render stream (see
    // _presentSkip). Skip; the next frame resumes the normal ready cadence.
    if (_presentMode == PresentMode::Dx12) {
        if (!_dxPresent || !_dxPresent->valid())
            return;
        if (_presentSkip) {
            _presentSkip = false;
            // Still record a clear + UI: with FLIP_DISCARD, executing nothing
            // would present an undefined back buffer.
            _dxPresent->presentFrame(nullptr, 0, 0);
            return;
        }
        auto& ready_frame = *_frames[_ready_frame_index];
        if (_type == TextureType::Int8) {
            // RGBA8 path: present the BYTE4 display_target (tone-mapped by
            // Pipeline) — same format as the swap chain back buffer. Source
            // dims are the PRESENT dims: equal to the back buffer normally;
            // under a render-size override DxPresent letterboxes the smaller/
            // larger image into the window.
            _dxPresent->presentFrame(
                static_cast<ID3D12Resource*>(ready_frame.display_target.native_handle()),
                _presentWidth, _presentHeight);
        } else {
            // Swap chain is R8G8B8A8; only the tonemapped BYTE4 display target
            // matches. Float display modes are not wired for Dx12 present.
            static bool sFloatModeWarned = false;
            if (!sFloatModeWarned) {
                sFloatModeWarned = true;
                CI_LOG_W("endFrame: Dx12 present supports TextureType::Int8 only - showing clear");
            }
            _dxPresent->presentFrame(nullptr, 0, 0);
        }
        return;
    }

    if (_presentSkip) {
        _presentSkip = false;
        return;
    }
    auto& ready_frame = *_frames[_ready_frame_index];

    if (_type == TextureType::Int8) {
        // RGBA8 path: copy from BYTE4 display_target (tone-mapped by Pipeline)
        if (_backend == Backend::DirectX) {
            auto source = static_cast<ID3D12Resource*>(ready_frame.display_target.native_handle());
            ready_frame.dx_display_interop->copy_to_gl(source, _width, _height);
            ready_frame.dx_display_interop->wait_for_copy();
        }
#if NT_ALLOW_CUDA
        else {
            auto map        = ready_frame.cuda_display_interop->map(_stream);
            auto cuda_array = map.get_mipmapped_array(0, 0);
            copyImageToGlArray(_stream, ready_frame.display_target,
                               cuda_array.get(), _width, _height, 4);
        }
#endif
    } else {
        // Float32/Float16 path: copy from render_target directly
        if (_backend == Backend::DirectX) {
            auto source = static_cast<ID3D12Resource*>(ready_frame.render_target.native_handle());
            ready_frame.dx_interop->copy_to_gl(source, _width, _height);
            ready_frame.dx_interop->wait_for_copy();
        }
#if NT_ALLOW_CUDA
        else {
            auto map        = ready_frame.cuda_interop->map(_stream);
            auto cuda_array = map.get_mipmapped_array(0, 0);
            copyImageToGlArray(_stream, ready_frame.render_target,
                               cuda_array.get(), _width, _height);
        }
#endif
    }
}

const ci::gl::Texture2dRef& Renderer::currentTexture() const {
    return _frames[_ready_frame_index]->texture;
}

const luisa::compute::Image<float>& Renderer::currentRenderTarget() const {
    return _frames[_current_frame_index]->render_target;
}

void Renderer::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height)
        return;  // No resize needed

    CI_LOG_D("NewTypeRenderer: Resizing to " << width << "x" << height);

    // Sync any in-flight GPU work before destroying frame resources
    _stream << luisa::compute::synchronize();

    _width  = width;
    _height = height;
    updateRenderSize();

    // Recreate all frame resources with new size
    destroyFrameResources();
    createFrameResources (_type);
    _presentSkip = true;

    // Recreate visual recorder for new dimensions (present dims — render-res
    // while a size override is active)
    _recorder = std::make_unique<util::VisualRecorder>(
        ci::ivec2(_presentWidth, _presentHeight), _presentMode == PresentMode::Gl);
}

#if NT_ALLOW_CUDA
void copyImageToGlArray(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height,
    size_t bytes_per_pixel) {

    // The native_handle() returns the CUDA array (CUarray) directly, not a pointer to CUDATexture
    // From cuda_device.cpp: native_handle = reinterpret_cast<void *>(p->handle())
    // where handle() returns _base_array, which for single-level images IS the CUarray
    CUarray src_array = reinterpret_cast<CUarray>(source.native_handle());
    CI_ASSERT_MSG(src_array != nullptr,
        "Warning: LuisaCompute Image has invalid CUDA array (native_handle is null)");

    // bytes_per_pixel: 16 for RGBA32F, 8 for RGBA16F, 4 for RGBA8
    size_t width_in_bytes = width * bytes_per_pixel;

    // Get CUDA stream handle from LuisaCompute Stream
    CUstream cuda_stream = reinterpret_cast<CUstream>(stream.native_handle());

    // Use CUDA Driver API to copy from CUDA array (LuisaCompute) to CUDA array (OpenGL)
    CUDA_MEMCPY2D copy_params   = {};
    copy_params.srcMemoryType   = CU_MEMORYTYPE_ARRAY;
    copy_params.srcArray        = src_array;
    copy_params.srcXInBytes     = 0;
    copy_params.srcY            = 0;
    copy_params.dstMemoryType   = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray        = reinterpret_cast<CUarray>(dest_array);
    copy_params.dstXInBytes     = 0;
    copy_params.dstY            = 0;
    copy_params.WidthInBytes    = width_in_bytes;
    copy_params.Height          = height;
    CUresult result = cuMemcpy2DAsync(&copy_params, cuda_stream);

    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        std::cerr << "cuMemcpy2DAsync failed: " << error_str << " (result=" << result << ")" << std::endl;
    }
}
#endif // NT_ALLOW_CUDA

} // namespace newtype::core
