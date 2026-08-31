#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/VideoPlayer.h"

#include "cinder/app/App.h"

#include <algorithm>

namespace newtype::media {

VideoPlayer::~VideoPlayer() {
    close();
}

bool VideoPlayer::open(std::filesystem::path const& path,
                       luisa::compute::Device& luisaDevice,
                       ID3D12Device* d3d12Device) noexcept {
    if (!d3d12Device) {
        CI_LOG_E("VideoPlayer::open: null D3D12 device");
        return false;
    }
    if (!_decoder.open(path, d3d12Device, VideoDecoderD3D11::Mode::HardwareNV12)) {
        return false;
    }
    if (!_bridge.init(luisaDevice, d3d12Device,
                      _decoder.shared_handle(),
                      _decoder.frame_width(), _decoder.frame_height(),
                      _decoder.keyed_mutex())) {
        _decoder.close();
        return false;
    }
    _lastWallTimeSec = -1.0;
    _mediaTimeSec    = 0.0;

    // Present frame 0 immediately — the decode loop only fires once the clock
    // advances past the first frame's span, so without this the shared texture
    // would show initial zeros for one frame interval.
    _decoder.read_next_frame();

    CI_LOG_I("VideoPlayer: opened '" << path.string()
             << "' " << _decoder.frame_width() << "x" << _decoder.frame_height()
             << ", duration=" << _decoder.duration_sec() << "s"
             << ", frameDur=" << _decoder.frame_duration_sec() << "s");
    return true;
}

void VideoPlayer::close() noexcept {
    _bridge.close();
    _decoder.close();
    _lastWallTimeSec = -1.0;
    _mediaTimeSec    = 0.0;
}

void VideoPlayer::seek(double sec) noexcept {
    if (!_decoder.is_open()) return;
    if (!_decoder.seek(sec)) return;
    _lastWallTimeSec = -1.0;   // dt = 0 on next update — no time jump
    // Present the post-seek frame now and rebase the clock to its delivered
    // PTS: SetCurrentPosition is keyframe-granular, so the frame may sit
    // before the target; rebasing skips re-playing that pre-roll.
    _decoder.read_next_frame();
    _mediaTimeSec = _decoder.current_time_sec();
}

bool VideoPlayer::update() noexcept {
    if (!_decoder.is_open() || !_playing) return false;

    // Latched EOF: end state for non-loop playback, or resume point when the
    // user enables looping after an EOF-pause.
    if (_decoder.is_eof()) {
        if (!_looping) return false;
        seek(0.0);
        if (_decoder.is_eof()) return false;   // seek failed
    }

    // Media clock from wall-clock seconds, NOT mDt. mDt is gated by mTick in
    // NewTypeEngine::update and collapses to 0 when tick is off — feeding that
    // to a media clock would freeze playback. See [[feedback-mtick-mdt-conflation]].
    double const now = ci::app::getElapsedSeconds();
    double const dt  = (_lastWallTimeSec > 0.0) ? std::min(now - _lastWallTimeSec, 0.1)
                                                : 0.0;
    _lastWallTimeSec = now;
    _mediaTimeSec += dt;

    // Decode while the presented frame's [pts, pts+frameDur) span has ended.
    // The dt clamp bounds catch-up to ≤ 0.1 s worth of frames per update,
    // so a window drag or long stall can't trigger a decode storm.
    bool decoded = false;
    while (_decoder.current_time_sec() + _decoder.frame_duration_sec() <= _mediaTimeSec) {
        if (!_decoder.read_next_frame()) {
            if (_decoder.is_eof()) {
                if (_looping) {
                    seek(0.0);   // presents frame 0, resets the clock
                } else {
                    CI_LOG_I("VideoPlayer: EOF reached at "
                             << _decoder.current_time_sec() << "s");
                    _playing = false;
                }
            }
            return decoded;
        }
        decoded = true;
    }
    return decoded;
}

double VideoPlayer::current_time_sec() const noexcept {
    double const dur = _decoder.duration_sec();
    if (dur > 0.0 && _mediaTimeSec > dur) return dur;
    return _mediaTimeSec < 0.0 ? 0.0 : _mediaTimeSec;
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
