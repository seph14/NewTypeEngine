#include "newtype/core/Renderer.h"
#include "newtype/util/Recorder.h"
#include "newtype/core/ShaderManager.h"
#include "cinder/Log.h"
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
#include "newtype/util/CompileProfiler.h"

namespace newtype::core {
using namespace luisa;

#if NT_PROFILING
namespace {
    newtype::util::CompileProfiler g_compile_profiler;
}
#endif

// Define static members
luisa::compute::Device   Renderer::_device;
luisa::compute::Context* Renderer::_context = nullptr;
luisa::compute::Stream   Renderer::_stream;

Renderer::Renderer(uint32_t width, uint32_t height, Backend backend, bool enableValidation)
    : _width (width),
      _height(height),
      _type  (TextureType::Float32),
      _backend(backend),
      _enableValidation(enableValidation) {

    initLuisaContext();
    const char* backendName = (_backend == Backend::DirectX) ? "DX12" : "CUDA";
    CI_LOG_I("NewTypeRenderer: Created " << width << "x" << height << " renderer (" << backendName << ")");
}

Renderer::Renderer(ci::ivec2 size, Backend backend, bool enableValidation)
    : _width(size.x),
    _height (size.y),
    _type   (TextureType::Float32),
    _backend(backend),
    _enableValidation(enableValidation) {

    initLuisaContext();
    const char* backendName = (_backend == Backend::DirectX) ? "DX12" : "CUDA";
    CI_LOG_I("NewTypeRenderer: Created " << size.x << "x" << size.y << " renderer (" << backendName << ")");
}

Renderer::~Renderer() {
    _recorder.reset();
    _captureFbo.reset();
    destroyFrameResources();
    
    if (_context) {
        delete _context;
        _context = nullptr;
    }
}

void Renderer::initLuisaContext() {
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

    // Force validation to be on so we can use profiler
#if NT_PROFILING
    _enableValidation = true;
#endif

    try {
        luisa::compute::DeviceConfig config{};
        config.inqueue_buffer_limit = false;
#if NT_PROFILING
        config.profiler = &g_compile_profiler;
#endif
        _device = _context->create_device(backendName, &config, _enableValidation);
        if (_device) {
            _stream = _device.create_stream();
            ShaderManager::setDevice(&_device);
            CI_LOG_I("LuisaCompute " << backendName << " backend initialized successfully");
        } else {
            CI_LOG_E("Failed to create " << backendName << " device");
            return;
        }
    } catch (const std::exception& e) {
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
    _recorder = std::make_unique<util::VisualRecorder>(ci::ivec2(_width, _height));
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

    if (_backend == Backend::DirectX) {
        // DX12 path: create shared texture + GL import via DxGLInterop
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
            // RGBA8 dual-path: FLOAT4 render target (no interop) + BYTE4 display target (with interop)
            auto frm = std::make_unique<FrameResource>();

            // 1. FLOAT4 render target — pipeline renders into this in full precision
            frm->render_target = _device.create_image<float>(
                luisa::compute::PixelStorage::FLOAT4, luisa::uint2(_width, _height));

            // 2. BYTE4 display target — tone-mapped output for the interop copy
            frm->display_target = _device.create_image<float>(
                luisa::compute::PixelStorage::BYTE4, luisa::uint2(_width, _height));

            // 3. GL_RGBA8 texture + interop for display_target
            if (_backend == Backend::DirectX) {
                auto dxDevice = static_cast<ID3D12Device*>(_device.native_handle());
                frm->dx_display_interop = std::make_unique<gl_interop::DxGLInterop>(
                    dxDevice, _width, _height, DXGI_FORMAT_R8G8B8A8_UNORM, GL_RGBA8);
                frm->display_texture = frm->dx_display_interop->gl_texture();
            }
#if NT_ALLOW_CUDA
            else {
                ci::gl::Texture2d::Format rgba8Fmt;
                rgba8Fmt.internalFormat(GL_RGBA8).dataType(GL_UNSIGNED_BYTE)
                    .wrap(GL_CLAMP_TO_EDGE)
                    .minFilter(GL_LINEAR).magFilter(GL_LINEAR);
                frm->display_texture = ci::gl::Texture2d::create(_width, _height, rgba8Fmt);
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
    auto& displayTex = _frames[0]->texture;
    auto fboFmt = ci::gl::Fbo::Format()
        .attachment(GL_COLOR_ATTACHMENT0, displayTex)
        .disableDepth();
    _captureFbo = ci::gl::Fbo::create(_width, _height, fboFmt);
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

FrameResource& Renderer::beginFrame() {
    _current_frame_index = (_current_frame_index + 1) % FRAME_COUNT;
    return *_frames[_current_frame_index];
}

void Renderer::markFrameReady() {
    _ready_frame_index = _current_frame_index;// (_current_frame_index + FRAME_COUNT - 1) % FRAME_COUNT;
}

void Renderer::endFrame() {
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

    CI_LOG_I("NewTypeRenderer: Resizing to " << width << "x" << height);

    // Sync any in-flight GPU work before destroying frame resources
    _stream << luisa::compute::synchronize();

    _width  = width;
    _height = height;

    // Recreate all frame resources with new size
    destroyFrameResources();
    createFrameResources (_type);

    // Recreate visual recorder for new dimensions
    _recorder = std::make_unique<util::VisualRecorder>(ci::ivec2(_width, _height));
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
