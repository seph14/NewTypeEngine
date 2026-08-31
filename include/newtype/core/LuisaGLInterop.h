#pragma once

#include "newtype/core/Config.h"

#if NT_ALLOW_CUDA

// IMPORTANT: Include Cinder headers BEFORE CUDA to avoid Windows/OpenGL conflicts
#include "cinder/gl/Texture.h"

// Windows.h conflicts with GL.h - define WIN32_LEAN_AND_MEAN first
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <luisa/luisa-compute.h>
#include <memory>
#include <stdexcept>

namespace newtype::gl_interop {

/**
 * @brief Exception class for CUDA-OpenGL interop errors
 */
class LuisaGLInteropException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    static LuisaGLInteropException from(cudaError_t result, const char* operation) {
        return LuisaGLInteropException(
            std::string("CUDA-OpenGL interop failed: ") + operation + " - " + cudaGetErrorString(result));
    }

    static void check(cudaError_t result, const char* operation) {
        if (result != cudaSuccess) {
            throw from(result, operation);
        }
    }
};

/**
 * @brief RAII wrapper for CUDA array pointer
 *
 * Manages a mapped CUDA array from a graphics resource subresource.
 */
class CUDA_array_view {
public:
    CUDA_array_view() noexcept = default;

    explicit CUDA_array_view(cudaArray_t array) noexcept
        : _array(array) {}

    [[nodiscard]] cudaArray_t get() const noexcept { return _array; }
    [[nodiscard]] explicit operator bool() const noexcept { return _array != nullptr; }

private:
    cudaArray_t _array = nullptr;
};

/**
 * @brief Scoped mapping of a CUDA graphics resource
 *
 * Maps a graphics resource for CUDA access and automatically unmaps on destruction.
 * Stores a pointer to the resource handle (cudaGraphicsResource*) for proper API usage.
 */
class Scoped_cuda_graphics_map {
public:
    // Construct from pointer to resource handle
    Scoped_cuda_graphics_map(cudaGraphicsResource* resource_ptr, cudaStream_t stream) noexcept
        : _resource_ptr(resource_ptr), _stream(stream) {
        // Map the graphics resource for CUDA access
        // Note: cudaGraphicsMapResources takes cudaGraphicsResource* (pointer to handle)
        LuisaGLInteropException::check(
            cudaGraphicsMapResources(1, &_resource_ptr, _stream),
            "cudaGraphicsMapResources");
        _mapped = true;
    }

    ~Scoped_cuda_graphics_map() noexcept {
        if (_mapped) {
            cudaGraphicsUnmapResources(1, &_resource_ptr, _stream);
        }
    }

    // Non-copyable, non-movable
    Scoped_cuda_graphics_map(const Scoped_cuda_graphics_map&) = delete;
    Scoped_cuda_graphics_map& operator=(const Scoped_cuda_graphics_map&) = delete;

    /**
     * @brief Get the mapped CUDA array for a specific mipmap level
     * @param arrayIndex Index of the array (for texture arrays, or 0 for single texture)
     * @param mipLevel Mipmap level to retrieve
     * @return CUDA_array_view for the requested subresource
     */
    [[nodiscard]] CUDA_array_view get_mipmapped_array(uint32_t arrayIndex, uint32_t mipLevel) const {
        cudaArray_t array = nullptr;
        // Dereference to get the actual handle, then pass to CUDA
        LuisaGLInteropException::check(
            cudaGraphicsSubResourceGetMappedArray(&array, _resource_ptr, arrayIndex, mipLevel),
            "cudaGraphicsSubResourceGetMappedArray");
        return CUDA_array_view(array);
    }

private:
    cudaGraphicsResource* _resource_ptr = nullptr;  // Pointer to the handle
    cudaStream_t _stream = nullptr;
    bool _mapped = false;
};

/**
 * @brief High-performance CUDA-OpenGL interop helper for LuisaCompute + Cinder
 *
 * This class manages the registration and mapping of OpenGL textures as CUDA resources,
 * enabling zero-copy data sharing between LuisaCompute (CUDA) and Cinder (OpenGL).
 *
 * Usage:
 * @code
 *   // Create Cinder texture (RGBA32F format for HDR)
 *   auto texture = ci::gl::Texture2d::create(1920, 1080,
 *       ci::gl::Texture2d::Format().dataType(GL_FLOAT).internalFormat(GL_RGBA32F));
 *
 *   // Create interop handler
 *   LuisaGLInterop interop(texture);
 *
 *   // In render loop:
 *   {
 *       auto map = interop.map(cuda_stream);
 *       auto cuda_array = map.get_mipmapped_array(0, 0);
 *
 *       // TODO: Use cuda_array with LuisaCompute (Phase 2)
 *       // Current limitation: Need to create LuisaCompute Image from CUDA array
 *   }
 *   // Automatically unmapped when scope ends
 *
 *   // Render texture in Cinder
 *   ci::gl::draw(texture);
 * @endcode
 *
 * @note Requires CUDA Runtime API and CUDA-OpenGL interop support
 */
class LuisaGLInterop {
public:
    /**
     * @brief Construct from a Cinder texture
     * @param texture Cinder texture to register with CUDA
     * @param flags CUDA graphics register flags (default: write-discard for render targets)
     * @throws LuisaGLInteropException if registration fails
     */
    explicit LuisaGLInterop(const ci::gl::Texture2dRef& texture,
                            unsigned int flags = cudaGraphicsMapFlagsWriteDiscard);

    /**
     * @brief Construct from an OpenGL texture ID
     * @param textureId OpenGL texture name
     * @param target Texture target (e.g., GL_TEXTURE_2D)
     * @param flags CUDA graphics register flags
     */
    LuisaGLInterop(GLuint textureId, GLenum target,
                    unsigned int flags = cudaGraphicsMapFlagsWriteDiscard);

    ~LuisaGLInterop();

    // Non-copyable
    LuisaGLInterop(const LuisaGLInterop&) = delete;
    LuisaGLInterop& operator=(const LuisaGLInterop&) = delete;

    // Non-movable (resource handle must stay at fixed address)
    LuisaGLInterop(LuisaGLInterop&&) = delete;
    LuisaGLInterop& operator=(LuisaGLInterop&&) = delete;

    /**
     * @brief Map the resource for CUDA access
     * @param stream CUDA stream (as cudaStream_t)
     * @return Scoped map object that auto-unmaps on destruction
     */
    [[nodiscard]] Scoped_cuda_graphics_map map(void* stream) noexcept;

    /**
     * @brief Map the resource for CUDA access (LuisaCompute Stream)
     * @param stream LuisaCompute Stream object
     * @return Scoped map object that auto-unmaps on destruction
     */
    [[nodiscard]] Scoped_cuda_graphics_map map(luisa::compute::Stream& stream) noexcept;

    /**
     * @brief Get the original OpenGL texture ID
     */
    [[nodiscard]] GLuint texture_id() const noexcept { return _textureId; }

    /**
     * @brief Check if the interop is valid
     */
    [[nodiscard]] explicit operator bool() const noexcept { return _resource_handle != nullptr; }

private:
    GLuint _textureId = 0;
    GLenum _target = GL_TEXTURE_2D;
    cudaGraphicsResource* _resource_handle = nullptr;  // The opaque handle
};

/**
 * @brief RAII wrapper for CUDA stream event synchronization
 */
class CUDA_event {
public:
    CUDA_event() {
        LuisaGLInteropException::check(
            cudaEventCreateWithFlags(&_event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
    }

    ~CUDA_event() noexcept {
        if (_event) {
            cudaEventDestroy(_event);
        }
    }

    // Non-copyable, non-movable
    CUDA_event(const CUDA_event&) = delete;
    CUDA_event& operator=(const CUDA_event&) = delete;

    void record(cudaStream_t stream) const {
        LuisaGLInteropException::check(
            cudaEventRecord(_event, stream),
            "cudaEventRecord");
    }

    void synchronize() const {
        LuisaGLInteropException::check(
            cudaEventSynchronize(_event),
            "cudaEventSynchronize");
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return _event; }

private:
    cudaEvent_t _event = nullptr;
};

/**
 * @brief Helper to initialize CUDA Runtime API
 *
 * Call this once at application startup to ensure CUDA is initialized.
 * Note: The first call to any CUDA Runtime API function implicitly initializes CUDA.
 */
class CUDA_runtime_initializer {
public:
    CUDA_runtime_initializer() {
        // CUDA is automatically initialized on first API call
        // This constructor ensures cudaFree(0) is called to trigger initialization
        void* ptr = nullptr;
        cudaError_t result = cudaMalloc(&ptr, 0);
        if (result == cudaSuccess) {
            cudaFree(ptr);
        }
        // Ignore errors - CUDA might already be initialized
    }

    static bool ensure_initialized() {
        static CUDA_runtime_initializer init;
        return true;
    }
};

} // namespace newtype::gl_interop

#endif // NT_ALLOW_CUDA
