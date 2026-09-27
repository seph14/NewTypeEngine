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
#include <optional>
#include <unordered_map>

// Forward declaration — full include only in Renderer.cpp
// Avoids pulling <d3d12.h> / <dxgi1_4.h> into every translation unit
namespace cinder { namespace app { class RendererD3d12; } }
namespace newtype::gl_interop { class DxGLInterop; }
namespace newtype::util { class VisualRecorder; }
namespace newtype::core { class DxPresent; }

namespace newtype::core {

/**
 * @brief Frame resources for triple-buffered rendering
 *
 * Holds a Cinder texture, LuisaCompute render target, and CUDA-OpenGL interop handler
 * for a single frame buffer. Use multiple instances for pipelined rendering.
 */
struct FrameResource {
    ci::gl::Texture2dRef            texture;        // Cinder GL texture for display
    luisa::compute::Image<float>    render_target;  // LuisaCompute Image for rendering (always FLOAT4, at RENDER resolution)

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

// Display path. Gl = classic path (Luisa DX12 -> DxGLInterop -> Cinder GL
// renderer). Dx12 = Luisa adopts Cinder's RendererD3d12 device and presents
// through its swap chain (see DxPresent) — no GL context involved.
enum class PresentMode {
    Gl, Dx12
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
     * @param backend Compute backend (Luisa CUDA/DX12)
     * @param enableValidation Enable Luisa validation layer
     * @param dx12Renderer Cinder D3D12 renderer to present through. Non-null
     *                    switches the display path to PresentMode::Dx12: the
     *                    Luisa device adopts this renderer's ID3D12Device
     *                    (DirectXDeviceConfigExt) and endFrame() copies into
     *                    its back buffers. Must outlive this Renderer (the
     *                    Cinder app owns it).
     */
    Renderer(uint32_t width, uint32_t height,
             Backend backend = Backend::CUDA, bool enableValidation = false,
             ci::app::RendererD3d12* dx12Renderer = nullptr);
    Renderer(ci::ivec2 size,
             Backend backend = Backend::CUDA, bool enableValidation = false,
             ci::app::RendererD3d12* dx12Renderer = nullptr);

    static RenderPtr create(uint32_t width, uint32_t height,
                            Backend backend = Backend::CUDA, bool enableValidation = false,
                            ci::app::RendererD3d12* dx12Renderer = nullptr) {
        return RenderPtr(new Renderer(width, height, backend, enableValidation, dx12Renderer));
    }

    static RenderPtr create(ci::ivec2 size,
                            Backend backend = Backend::CUDA, bool enableValidation = false,
                            ci::app::RendererD3d12* dx12Renderer = nullptr) {
        return RenderPtr(new Renderer(size, backend, enableValidation, dx12Renderer));
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
     *        (GL present mode only; null under Dx12)
     */
    [[nodiscard]] const ci::gl::Texture2dRef& currentTexture() const;

    /**
     * @brief Get the current frame's LuisaCompute render target
     */
    [[nodiscard]] const luisa::compute::Image<float>& currentRenderTarget() const;

    /**
     * @brief Get the capture FBO (wraps display texture for PBO readback)
     *        (GL present mode only; null under Dx12)
     */
    [[nodiscard]] ci::gl::FboRef captureFbo() const noexcept { return _captureFbo; }

    /**
     * @brief Get the visual recorder component (may be null before initialize)
     */
    [[nodiscard]] util::VisualRecorder* recorder() const noexcept { return _recorder.get(); }

    // ---- DX12 present mode (no-ops under PresentMode::Gl) ----

    /** @brief CPU side: start the ImGui frame (DxPresent::beginUiFrame) */
    void beginUiFrame(float deltaTime, uint32_t width, uint32_t height);

    /**
     * @brief Per-frame recorder tick. GL mode: PBO readback of captureFbo().
     * Dx12 mode: blocking Luisa download of the ready display target. Call
     * BEFORE render()/beginFrame() — the download's synchronize then overlaps
     * the wait Pipeline::beginFrame() performs anyway.
     */
    void updateRecorder();

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
     * @brief Set the render scale (render resolution = display size * scale).
     * Display size stays at the window size; only the render targets shrink.
     * Requires RGBA8 (Int8) mode — the upscale output path only exists there.
     * Recreates frame resources; call outside the frame loop.
     */
    void setRenderScale(float scale);

    /**
     * @brief Decouple the render resolution from the window entirely
     * (non-perspective projections: equirect 2:1, room-rig atlases...).
     * When set, render targets AND the display/capture target (recorder,
     * present source) run at this size; the window shows a letterboxed
     * preview. Implies renderScale = 1. nullopt restores window-derived
     * sizing. Recreates frame resources; call outside the frame loop.
     */
    void setRenderSizeOverride(const std::optional<luisa::uint2>& size);
    [[nodiscard]] std::optional<luisa::uint2> renderSizeOverride() const noexcept {
        return _renderSizeOverride;
    }

    // Present/capture dimensions: the size of the display target the
    // tonemap writes, the recorder captures and the present path blits.
    // Equals the render size while an override is active (capture at render
    // resolution), the window size otherwise.
    [[nodiscard]] uint32_t presentWidth()  const noexcept { return _presentWidth; }
    [[nodiscard]] uint32_t presentHeight() const noexcept { return _presentHeight; }

    /**
     * @brief Get render width
     */
    [[nodiscard]] uint32_t width() const noexcept { return _width; }

    /**
     * @brief Get render height
     */
    [[nodiscard]] uint32_t height() const noexcept { return _height; }

    // Render-resolution accessors (display size * renderScale, min 1)
    [[nodiscard]] uint32_t renderWidth()  const noexcept { return _renderWidth; }
    [[nodiscard]] uint32_t renderHeight() const noexcept { return _renderHeight; }
    [[nodiscard]] float    renderScale()  const noexcept { return _renderScale; }

    [[nodiscard]] Backend backend() const noexcept { return _backend; }

    /** @brief Active display path (GL interop vs D3D12 swap chain) */
    [[nodiscard]] PresentMode presentMode() const noexcept { return _presentMode; }
    [[nodiscard]] bool isDx12Present() const noexcept { return _presentMode == PresentMode::Dx12; }

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
    void updateRenderSize() noexcept;

    // dimension — _width/_height are DISPLAY (window) dimensions; the render
    // targets use _renderWidth/_renderHeight = display * _renderScale, or the
    // explicit _renderSizeOverride when one is active (non-perspective).
    // _presentWidth/_presentHeight size the display/capture target: override
    // size when active (capture at render res), display size otherwise.
    uint32_t _width  = 0;
    uint32_t _height = 0;
    float    _renderScale = 1.0f;
    uint32_t _renderWidth  = 0;
    uint32_t _renderHeight = 0;
    std::optional<luisa::uint2> _renderSizeOverride;
    uint32_t _presentWidth  = 0;
    uint32_t _presentHeight = 0;
    Backend  _backend = Backend::CUDA;
    bool     _enableValidation = false;

    // Display path (Gl unless a Cinder RendererD3d12 was supplied)
    ci::app::RendererD3d12* _dx12Renderer = nullptr; // not owned (Cinder app owns)
    PresentMode             _presentMode  = PresentMode::Gl;
    std::unique_ptr<DxPresent> _dxPresent;

    // Double buffering for pipelined rendering
    static constexpr size_t FRAME_COUNT = 2;
    std::array       <FrmRscPtr, FRAME_COUNT> _frames;
    std::unordered_map<luisa::string, FrmRscPtr> _sharedTextures;

    TextureType     _type;
    size_t          _current_frame_index = 0;
    size_t          _ready_frame_index   = 0;  // Frame ready for display
    // Set when frame resources were just (re)created: the stale
    // _ready_frame_index can alias the frame render() is about to write, and
    // endFrame's interop copy would then race the render stream on a
    // non-simultaneous-access display target (D3D12 cross-queue hazard →
    // TDR; crash log 2026-09-16). Skip one present; there is no valid
    // synced frame to show anyway.
    bool            _presentSkip = false;

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
