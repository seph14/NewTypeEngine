#include "newtype/core/Config.h"

#if NT_ALLOW_CUDA

#include "newtype/core/LuisaGLInterop.h"
#include <iostream>
#include "cinder/Log.h"

namespace newtype::gl_interop {

LuisaGLInterop::LuisaGLInterop(const ci::gl::Texture2dRef& texture, unsigned int flags)
    : _textureId(texture->getId()), _target(texture->getTarget()) {

    // Ensure CUDA Runtime API is initialized
    CUDA_runtime_initializer::ensure_initialized();

    // Register the OpenGL texture with CUDA
    LuisaGLInteropException::check(
        cudaGraphicsGLRegisterImage(&_resource_handle, _textureId, _target, flags),
        "cudaGraphicsGLRegisterImage");

    CI_LOG_I("LuisaGLInterop: Registered GL texture " << _textureId <<
             " (target: 0x" << std::hex << _target << std::dec << ") with CUDA");
}

LuisaGLInterop::LuisaGLInterop(GLuint textureId, GLenum target, unsigned int flags)
    : _textureId(textureId), _target(target) {

    // Ensure CUDA Runtime API is initialized
    CUDA_runtime_initializer::ensure_initialized();

    // Register the OpenGL texture with CUDA
    LuisaGLInteropException::check(
        cudaGraphicsGLRegisterImage(&_resource_handle, _textureId, _target, flags),
        "cudaGraphicsGLRegisterImage");

    CI_LOG_I("LuisaGLInterop: Registered GL texture " << _textureId <<
             " (target: 0x" << std::hex << _target << std::dec << ") with CUDA");
}

LuisaGLInterop::~LuisaGLInterop() {
    if (_resource_handle)
        cudaGraphicsUnregisterResource(_resource_handle);
}

Scoped_cuda_graphics_map LuisaGLInterop::map(void* stream) noexcept {
    return Scoped_cuda_graphics_map(_resource_handle, static_cast<cudaStream_t>(stream));
}

Scoped_cuda_graphics_map LuisaGLInterop::map(luisa::compute::Stream& luisa_stream) noexcept {
    // Get the native stream handle from LuisaCompute Stream
    return Scoped_cuda_graphics_map(_resource_handle, static_cast<cudaStream_t>(luisa_stream.native_handle()));
}

} // namespace newtype::gl_interop

#endif // NT_ALLOW_CUDA
