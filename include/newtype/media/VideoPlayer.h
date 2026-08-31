#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/VideoDecoderD3D11.h"
#include "newtype/media/VideoTextureBridge.h"

#include <luisa/luisa-compute.h>

#include <filesystem>

struct ID3D12Device;

namespace newtype::media {

//
// Orchestrates a single video: owns a VideoDecoderD3D11 + a VideoTextureBridge.
// Advances the media clock from ci::app::getElapsedSeconds() (NOT mDt — mDt is
// gated by mTick in NewTypeEngine::update and collapses to 0 when tick is off,
// which would freeze video playback; see [[feedback-mtick-mdt-conflation]]).
//
class VideoPlayer {
public:
    VideoPlayer() = default;
    ~VideoPlayer();

    VideoPlayer(VideoPlayer const&) = delete;
    VideoPlayer& operator=(VideoPlayer const&) = delete;
    VideoPlayer(VideoPlayer&&) = delete;
    VideoPlayer& operator=(VideoPlayer&&) = delete;

    // luisaDevice + d3d12Device are both required — the decoder needs the D3D12
    // device for LUID matching, the bridge needs it to open the shared handle.
    bool open(std::filesystem::path const& path,
              luisa::compute::Device& luisaDevice,
              struct ID3D12Device* d3d12Device) noexcept;

    void close() noexcept;

    // Advances the media clock from ci::app::getElapsedSeconds() and decodes
    // frames until the currently presented frame's [pts, pts+frameDur) span
    // contains the clock. Returns true when ≥1 frame was decoded this tick.
    // Call once per render-thread update.
    bool update() noexcept;

    void play()  noexcept { _playing = true; _lastWallTimeSec = -1.0; }
    void pause() noexcept { _playing = false; }
    [[nodiscard]] bool is_playing() const noexcept { return _playing; }
    [[nodiscard]] bool is_open() const noexcept    { return _decoder.is_open(); }
    [[nodiscard]] bool is_eof() const noexcept     { return _decoder.is_eof(); }

    // Loop: when reached, EOF rewinds to the head and playback continues.
    void set_looping(bool on) noexcept { _looping = on; }
    [[nodiscard]] bool is_looping() const noexcept { return _looping; }

    // Seek the playhead (seconds) and present the post-seek frame immediately.
    // Works while paused. rewind() is seek-to-head.
    void seek(double sec) noexcept;
    void rewind() noexcept { seek(0.0); }

    [[nodiscard]] uint32_t frame_width()  const noexcept { return _decoder.frame_width(); }
    [[nodiscard]] uint32_t frame_height() const noexcept { return _decoder.frame_height(); }
    [[nodiscard]] double   duration_sec() const noexcept { return _decoder.duration_sec(); }
    // Media playhead (wall-clock-driven), clamped to [0, duration]. NOT the
    // decoder PTS — this is what UI should display.
    [[nodiscard]] double current_time_sec() const noexcept;

    // The Luisa image — sample this from DX kernels via the bindless slot.
    [[nodiscard]] luisa::compute::Image<float>& image() noexcept { return _bridge.image(); }

    // Bridge access — needed for per-frame acquire/release around Luisa work
    // if tighter sync than the natural render-thread ordering is required.
    [[nodiscard]] VideoTextureBridge& bridge() noexcept { return _bridge; }

private:
    VideoDecoderD3D11 _decoder;
    VideoTextureBridge _bridge;
    double            _lastWallTimeSec = -1.0;
    double            _mediaTimeSec    = 0.0;
    bool              _playing         = true;
    bool              _looping         = false;
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
