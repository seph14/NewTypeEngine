#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

// Cinder BEFORE Windows/D3D headers (see CLAUDE.md, DxGLInterop.h for the rule).
#include "cinder/Log.h"

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

#include <cstdint>
#include <vector>
#include <string_view>
#include <filesystem>

struct ID3D11Texture2D;          // forward decl
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11VideoDevice;
struct ID3D11VideoContext;
struct ID3D11VideoProcessor;
struct ID3D11VideoProcessorEnumerator;
struct ID3D11VideoProcessorInputView;
struct ID3D11VideoProcessorOutputView;
struct ID3D11Query;
struct ID3D12Device;
struct ID3D12Resource;
struct IDXGIKeyedMutex;
struct IDXGIAdapter1;
struct IMFSourceReader;
struct IMFMediaType;
struct IMFDXGIDeviceManager;
struct IMFSample;

namespace newtype::media {

//
// Hardware video decoder using Media Foundation + (later phases) D3D11 DXVA.
//
// Phase 1: synchronous IMFSourceReader with RGB32 CPU output. Pure software path —
//          validates MF plumbing, sample/buffer lifetime, frame timing, EOS handling,
//          PNG round-trip — before any GPU resources enter the picture.
// Phase 2: switches to D3D11-device-backed HW decode (NV12 texture array output).
//          D3D11 device is LUID-matched to the caller-supplied D3D12 device so that
//          later phases can share textures across the two APIs.
// Phase 3+: NV12→RGBA8 conversion, shared handle to D3D12, Luisa image wrap.
//
class VideoDecoderD3D11 {
public:
    enum class Mode {
        CpuRgb32,      // Phase 1: software decode, RGB32 CPU buffer
        HardwareNV12,  // Phase 2+: D3D11 DXVA, NV12 texture array (decoder-owned)
    };

    VideoDecoderD3D11() = default;
    ~VideoDecoderD3D11();

    VideoDecoderD3D11(VideoDecoderD3D11 const&) = delete;
    VideoDecoderD3D11& operator=(VideoDecoderD3D11 const&) = delete;
    VideoDecoderD3D11(VideoDecoderD3D11&&) = delete;
    VideoDecoderD3D11& operator=(VideoDecoderD3D11&&) = delete;

    // luisaDevice is required for Mode::HardwareNV12 (D3D11 device must be LUID-paired
    // with the D3D12 device for OpenSharedHandle1 to succeed in later phases).
    // Ignored for Mode::CpuRgb32.
    bool open(std::filesystem::path const& path,
              struct ID3D12Device* luisaDevice = nullptr,
              Mode mode = Mode::CpuRgb32);

    // Read the next decoded frame. Returns false on EOS or error.
    // After a successful call:
    //   - CpuRgb32 mode:    cpu_pixels_bgra() / cpu_stride() expose the decoded BGRA8 bytes
    //   - HardwareNV12 mode: hw_nv12_texture() / hw_subresource_index() expose the decoder
    //                        texture (decoder-owned — do NOT release; copy if persistent)
    bool read_next_frame();

    // Seek to a presentation time (seconds). Keyframe-granular on some containers —
    // the next read_next_frame() may deliver a frame slightly BEFORE the target and
    // its PTS becomes the authoritative position. Clears EOF. Seeking to ≥ duration
    // is clamped to the last frame's span (exact-duration seek would hit EOS instantly).
    bool seek(double sec);

    // Drain any queued samples, release MF resources, MFShutdown on last instance.
    void close();

    [[nodiscard]] Mode    mode()        const noexcept { return _mode; }
    [[nodiscard]] bool    is_open()     const noexcept { return _reader != nullptr; }
    [[nodiscard]] bool    is_eof()      const noexcept { return _eof; }

    [[nodiscard]] uint32_t frame_width()  const noexcept { return _width; }
    [[nodiscard]] uint32_t frame_height() const noexcept { return _height; }
    [[nodiscard]] int32_t  cpu_stride()   const noexcept { return _cpuStride; }

    // CpuRgb32 mode only. Returns the most recent decoded frame as BGRA8 (matches
    // MFVideoFormat_RGB32 memory layout: B in low byte). May be bottom-up or
    // top-down depending on stride sign — see cpu_stride() and is_top_down().
    [[nodiscard]] uint8_t const* cpu_pixels_bgra() const noexcept { return _cpuPixels.data(); }
    [[nodiscard]] size_t         cpu_pixels_size() const noexcept { return _cpuPixels.size(); }
    [[nodiscard]] bool           is_top_down()      const noexcept { return _cpuStride < 0; }

    // HardwareNV12 mode only. The returned texture is owned by the MF decoder and
    // reused across frames; the pointer is valid only until the next read_next_frame().
    // subresource_index identifies the array slice inside the texture.
    [[nodiscard]] struct ID3D11Texture2D* hw_nv12_texture()      const noexcept { return _hwTexture; }
    [[nodiscard]] uint32_t                hw_subresource_index() const noexcept { return _hwSubresourceIndex; }

    // HardwareNV12 mode only. The Video Processor output — RGBA8 converted from
    // the latest NV12 frame via ID3D11VideoProcessor. Always reflects the last
    // read_next_frame() call. Decoder-owned, persists across frames.
    [[nodiscard]] struct ID3D11Texture2D* hw_rgba8_texture()     const noexcept { return _rgba8Texture; }

    // Phase 3 verification helper. Copies the current RGBA8 texture to a CPU
    // buffer for inspection / PNG dump. Returns RGBA8 in row-major top-down order.
    // Stride is bytes-per-row, always a positive multiple of 4.
    bool readback_rgba8_to_cpu(/*out*/ std::vector<uint8_t>& outPixels,
                               /*out*/ int32_t& outStride);

    // Access to the D3D11 device/context — needed by Phase 3 (Video Processor) and
    // Phase 4 (shared texture creation).
    [[nodiscard]] struct ID3D11Device*        d3d11_device()  const noexcept { return _d3d11Device; }
    [[nodiscard]] struct ID3D11DeviceContext* d3d11_context() const noexcept { return _d3d11Context; }

    [[nodiscard]] double duration_sec()       const noexcept { return _durationSec; }
    [[nodiscard]] double current_time_sec()   const noexcept { return _currentTimeSec; }
    // Average frame duration from MF_MT_FRAME_RATE (1/30 fallback when absent).
    [[nodiscard]] double frame_duration_sec() const noexcept { return _frameDurationSec; }

    // Phase 4+ accessors — shared texture state for cross-API bridging.
    // The shared RGBA8 texture is _rgba8Texture with SHARED_NTHANDLE | SHARED_KEYEDMUTEX;
    // D3D12 opens it via OpenSharedHandle1 on the same-adapter device.
    [[nodiscard]] struct ID3D11Texture2D* shared_texture() const noexcept { return _rgba8Texture; }
    [[nodiscard]] struct IDXGIKeyedMutex* keyed_mutex()    const noexcept { return _keyedMutex; }
    [[nodiscard]] void*                   shared_handle()  const noexcept { return _sharedHandle; }

private:
    bool _open_cpu_rgb32(std::filesystem::path const& path);
    bool _open_hardware_nv12(std::filesystem::path const& path, ID3D12Device* luisaDevice);
    bool _create_d3d11_device_on_luid(ID3D12Device* luisaDevice, IDXGIAdapter1** outAdapter);
    bool _configure_rgb32_cpu_output();
    bool _configure_nv12_hw_output();
    bool _query_frame_metrics();
    bool _init_video_processor();         // Phase 3: NV12 → RGBA8 via VideoProcessorBlt
    bool _convert_nv12_to_rgba8();        // Phase 3: per-frame blt; uses _hwTexture + _hwSubresourceIndex
    bool _extract_hw_texture(struct IMFSample* sample);
    bool _copy_cpu_buffer(struct IMFSample* sample);

    // MF objects — opaque to header readers; full defs live in the .cpp
    struct IMFSourceReader*      _reader       = nullptr;
    struct IMFMediaType*         _outputType   = nullptr;
    struct IMFDXGIDeviceManager* _dxgiManager  = nullptr;

    // D3D11 ownership (HardwareNV12 mode only)
    struct ID3D11Device*                         _d3d11Device      = nullptr;
    struct ID3D11DeviceContext*                  _d3d11Context     = nullptr;
    struct ID3D11VideoDevice*                    _videoDevice      = nullptr;
    struct ID3D11VideoContext*                   _videoContext     = nullptr;
    struct ID3D11VideoProcessorEnumerator*       _videoEnum        = nullptr;
    struct ID3D11VideoProcessor*                 _videoProcessor   = nullptr;
    struct ID3D11VideoProcessorOutputView*       _rgba8OutputView  = nullptr;
    struct ID3D11Texture2D*                      _rgba8Texture     = nullptr;
    // VP output is staged here first (no shared flags — VP rejects keyed-mutex
    // textures as output target with E_INVALIDARG on some drivers). Per frame
    // we CopyResource from this to _rgba8Texture under the keyed mutex.
    struct ID3D11Texture2D*                      _rgba8TextureInternal = nullptr;
    // Stable single-slice NV12 copy of the latest decoder frame. The decoder
    // texture is BIND_DECODER and multi-sliced; reading it directly from the
    // VP hangs the GPU on some drivers. Copying to a clean NV12 texture breaks
    // the bad state interaction. The input view on this stable texture is
    // created once and reused — per-frame view creation churns the runtime's
    // view table and triggers spurious E_INVALIDARG.
    struct ID3D11Texture2D*                      _nv12Copy         = nullptr;
    struct ID3D11VideoProcessorInputView*        _nv12InputView    = nullptr;

    // Phase 4: cross-API sharing state. _rgba8Texture is created with
    // SHARED_NTHANDLE | SHARED_KEYEDMUTEX; the handle is opened on Luisa's
    // D3D12 device via OpenSharedHandle. The keyed mutex coordinates writes
    // (D3D11 VP) vs reads (D3D12) across the two APIs.
    struct IDXGIKeyedMutex*                      _keyedMutex       = nullptr;
    void*                                        _sharedHandle     = nullptr;
    // GPU completion query — End'd after CopyResource+Flush, polled until done
    // before keyed-mutex release. Without this, D3D12 (separate command queue)
    // can read the shared texture before D3D11's CopyResource actually executes,
    // returning initial zeros (all-black image).
    struct ID3D11Query*                          _gpuDoneQuery     = nullptr;

    // Latest decoded NV12 texture (non-owning; MF owns it). Valid only between
    // read_next_frame() and the next call.
    struct ID3D11Texture2D* _hwTexture           = nullptr;
    uint32_t                _hwSubresourceIndex  = 0;

    // Cached staging texture for readback_rgba8_to_cpu (reused across calls).
    struct ID3D11Texture2D* _stagingTexture      = nullptr;
    uint32_t                _stagingWidth        = 0;
    uint32_t                _stagingHeight       = 0;

    // Frame state
    uint32_t _width          = 0;
    uint32_t _height         = 0;
    int32_t  _cpuStride      = 0;          // bytes per row; negative ⇒ top-down
    double   _durationSec    = 0.0;
    double   _currentTimeSec = 0.0;
    double   _frameDurationSec = 1.0 / 30.0;
    bool     _eof            = false;
    Mode     _mode           = Mode::CpuRgb32;

    std::vector<uint8_t> _cpuPixels;       // BGRA8, row-major, stride = _cpuStride

    static int s_instanceCount;            // for MFStartup/Shutdown refcount
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
