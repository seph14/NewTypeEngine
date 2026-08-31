#pragma once

#include "newtype/core/Config.h"

#if NT_ALLOW_CUDA
#include "newtype/core/LuisaGLInterop.h"
#endif

#include "newtype/util/Recorder.h"
#include <luisa/luisa-compute.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Fbo.h>
#include <memory>
#include <unordered_map>

// Forward declaration — full include only in Renderer.cpp
// Avoids pulling <d3d12.h> / <dxgi1_4.h> into every translation unit
namespace newtype::gl_interop { class DxGLInterop; }
namespace newtype::util { class VisualRecorder; }

namespace newtype::core {

/**
 * @brief Frame resources for triple-buffered rendering
 *
 * Holds a Cinder texture, LuisaCompute render target, and CUDA-OpenGL interop handler
 * for a single frame buffer. Use multiple instances for pipelined rendering.
 */
struct FrameResource {
    ci::gl::Texture2dRef            texture;        // Cinder GL texture for display
    luisa::compute::Image<float>    render_target;  // LuisaCompute Image for rendering (always FLOAT4)

    // RGBA8 display path (only populated when TextureType::Int8)
    luisa::compute::Image<float>    display_target;  // BYTE4 storage for tone-mapped output
    ci::gl::Texture2dRef            display_texture; // GL_RGBA8 texture for display
#if NT_ALLOW_CUDA
    std::unique_ptr<gl_interop::LuisaGLInterop>  cuda_display_interop;  // interop for display_target (CUDA)
#endif
    std::unique_ptr<gl_interop::DxGLInterop>     dx_display_interop;   // interop for display_target (DX)

    // Backend-specific interop (only one is active at a time)
#if NT_ALLOW_CUDA
    std::unique_ptr<gl_interop::LuisaGLInterop>  cuda_interop;  // CUDA-GL interop
#endif
    std::unique_ptr<gl_interop::DxGLInterop>     dx_interop;   // DX12-GL interop

    uint64_t                        fence_value = 0;// For synchronization
};

typedef std::unique_ptr<FrameResource> FrmRscPtr;

enum class TextureType {
    Float32 = 0, Float16 = 1, Int8 = 2
};

enum class Backend {
    CUDA, DirectX
};

class Renderer;
typedef std::unique_ptr<Renderer> RenderPtr;

/**
 * @brief Manages LuisaCompute to Cinder rendering pipeline
 *
 * This class handles:
 * 1. Creating Cinder textures with proper format for CUDA interop
 * 2. Registering textures with CUDA for zero-copy access
 * 3. Managing LuisaCompute Image render targets
 * 4. Copying rendered data from LuisaCompute to Cinder textures
 *
 * Usage:
 * @code
 *   NewTypeRenderer renderer(1920, 1080, luisa_device);
 *
 *   // In setup:
 *   renderer.initialize();
 *
 *   // In render loop:
 *   auto& frame = renderer.begin_frame(luisa_stream);
 *
 *   // Render to frame.render_target using LuisaCompute
 *   // ...
 *
 *   renderer.end_frame(luisa_stream);
 *
 *   // Draw to screen:
 *   ci::gl::draw(frame.texture);
 * @endcode
 */
class Renderer {
public:
    /**
     * @brief Construct renderer with specified dimensions
     * @param width Render target width
     * @param height Render target height
     * @param device LuisaCompute device
     */
    Renderer(uint32_t width, uint32_t height,
             Backend backend = Backend::CUDA, bool enableValidation = false);
    Renderer(ci::ivec2 size,
             Backend backend = Backend::CUDA, bool enableValidation = false);

    static RenderPtr create(uint32_t width, uint32_t height,
                            Backend backend = Backend::CUDA, bool enableValidation = false) {
        return RenderPtr(new Renderer(width, height, backend, enableValidation));
    }

    static RenderPtr create(ci::ivec2 size,
                            Backend backend = Backend::CUDA, bool enableValidation = false) {
        return RenderPtr(new Renderer(size, backend, enableValidation));
    }

    ~Renderer();

    // Non-copyable, non-movable (manages GPU resources)
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    /**
     * @brief Initialize all GPU resources
     * Call this after construction and before any rendering
     */
    void initialize(TextureType type = TextureType::Float32);

    /**
     * @brief Begin a new frame, returns the frame resources to use
     * @return FrameResource containing render target and texture
     */
    [[nodiscard]] FrameResource& beginFrame();

    /**
     * @brief Mark the previous frame as ready for display
     * Called after Pipeline syncs the previous frame's GPU work in beginFrame()
     */
    void markFrameReady();

    /**
     * @brief End the current frame, performs copy from LuisaCompute to GL texture
     */
    void endFrame();

    /**
     * @brief Get the current frame's texture for rendering in Cinder
     */
    [[nodiscard]] const ci::gl::Texture2dRef& currentTexture() const;

    /**
     * @brief Get the current frame's LuisaCompute render target
     */
    [[nodiscard]] const luisa::compute::Image<float>& currentRenderTarget() const;

    /**
     * @brief Get the capture FBO (wraps display texture for PBO readback)
     */
    [[nodiscard]] ci::gl::FboRef captureFbo() const noexcept { return _captureFbo; }

    /**
     * @brief Get the visual recorder component (may be null before initialize)
     */
    [[nodiscard]] util::VisualRecorder* recorder() const noexcept { return _recorder.get(); }

    // custom interop
    bool appendInteropTexture(std::string_view name, const ci::uvec2 size, 
                              ci::gl::Texture::Format& type);
    [[nodiscard]] const ci::gl::Texture2dRef          getInteropTex  (std::string_view name);
    [[nodiscard]] const luisa::compute::Image<float>& getInteropImage(std::string_view name);

    /**
     * @brief Resize all frame resources
     * @param width New width
     * @param height New height
     */
    void resize(uint32_t width, uint32_t height);

    /**
     * @brief Get render width
     */
    [[nodiscard]] uint32_t width() const noexcept { return _width; }

    /**
     * @brief Get render height
     */
    [[nodiscard]] uint32_t height() const noexcept { return _height; }

    [[nodiscard]] Backend backend() const noexcept { return _backend; }

    /**
     * @brief Check if renderer is in RGBA8 (Int8) output mode
     */
    [[nodiscard]] bool isRGBA8() const noexcept { return _type == TextureType::Int8; }

    /**
     * @brief Get the LuisaCompute device
     */
    [[nodiscard]] static luisa::compute::Device& device() noexcept { return _device; }
    /**
     * @brief Get the LuisaCompute context
     */
    [[nodiscard]] static luisa::compute::Context* ctx() noexcept { return _context; }
    /**
     * @brief Get the LuisaCompute stream
     */
    [[nodiscard]] static luisa::compute::Stream& stream() noexcept { return _stream; }

private:
    void initLuisaContext();
    void createFrameResources(TextureType type);
    void destroyFrameResources  ();
    FrmRscPtr createFrameInterop(ci::uvec2 size, ci::gl::Texture2d::Format& fmt);
    
    // dimension
    uint32_t _width  = 0;
    uint32_t _height = 0;
    Backend  _backend = Backend::CUDA;
    bool     _enableValidation = false;

    // Double buffering for pipelined rendering
    static constexpr size_t FRAME_COUNT = 2;
    std::array       <FrmRscPtr, FRAME_COUNT> _frames;
    std::unordered_map<luisa::string, FrmRscPtr> _sharedTextures;

    TextureType     _type;
    size_t          _current_frame_index = 0;
    size_t          _ready_frame_index   = 0;  // Frame ready for display

    // Visual recorder
    ci::gl::FboRef                          _captureFbo;     // FBO wrapping display texture for PBO readback
    std::unique_ptr<util::VisualRecorder>   _recorder;       // Screenshot + video recording component

protected:
    // LuisaCompute context and device
    static luisa::compute::Device   _device;
    static luisa::compute::Context* _context;
    static luisa::compute::Stream   _stream;
};

#if NT_ALLOW_CUDA
/**
 * @brief Helper to copy a LuisaCompute Image to a CUDA-OpenGL mapped array
 *
 * This function handles the copy operation from a LuisaCompute Image to
 * a CUDA array obtained from CUDA-OpenGL graphics interop.
 *
 * @param stream LuisaCompute stream
 * @param source Source LuisaCompute Image
 * @param dest_array Destination CUDA array (from CUDA-OpenGL interop)
 * @param width Image width
 * @param height Image height
 */
void copyImageToGlArray(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height,
    size_t bytes_per_pixel = 16);
#endif // NT_ALLOW_CUDA

} // namespace newtype::core
