#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/AudioRingBuffer.h"
#include "newtype/media/VideoAudioStream.h"
#include "newtype/media/VideoDecoderD3D11.h"
#include "newtype/media/VideoTextureBridge.h"

#include <luisa/luisa-compute.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

struct ID3D12Device;

namespace cinder { namespace audio { class GainNode; } }
namespace newtype::media { class VideoAudioNode; }

namespace newtype::media {

//
// Orchestrates a single video: owns a VideoDecoderD3D11 + a VideoTextureBridge,
// plus (when the file has an audio track and NT_ENABLE_AUDIO is on) an
// audio side-channel: VideoAudioStream (MF float PCM decode, pump thread)
// → AudioRingBuffer (SPSC) → VideoAudioNode (cinder audio graph).
//
// Threading: a decode thread owns the video decoder between parks — it runs
// ReadSample + the NV12→RGBA blt (into rotating decoder-internal slots) ahead
// of the media clock by ~kVideoPrerollSec, then hands off through a prepared-
// frame queue. The render thread promotes the newest due frame into the
// D3D12-shared texture from publish_due_frame(), which the engine calls at
// the GPU-drained point right after its per-frame synchronize — the only
// moment the copy + GPU-completion wait are cheap and race-free. Doing all
// of this on the render thread (the previous design) stalls it behind the
// in-flight path-traced frame every time a video frame decodes. The decode
// ahead happens while the audio pre-roll gate holds the clock, so playback
// starts with both sides buffered instead of video chasing a running clock
// through cold decoders.
//
// Clock: audio-master when the audio path is live — the raw audio position
// (VideoAudioNode::position_sec() minus a constant sync offset) is the
// target, but the media clock advances at wall rate between corrections
// (clamped to never lead the target, so audio starvation still freezes
// video) — the raw position steps one ~10.7 ms audio block at a time, which
// jitters frame swaps. The audio node is created autoEnable(false): it
// stays silent until update()'s pre-roll gate sees kAudioPrerollSeconds
// buffered, then the epoch is primed at the decode anchor and playback
// (audio + video together) starts. Video-only files (or NT_ENABLE_AUDIO=0
// builds) run the clock off ci::app::getElapsedSeconds() (NOT mDt — mDt is
// gated by mTick in NewTypeEngine::update and collapses to 0 when tick is
// off, which would freeze video playback; see
// [[feedback-mtick-mdt-conflation]]).
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

    // Advances the media clock (audio-master smoothing / wall clock) and
    // mirrors it to the decode thread. Frame decoding happens on the decode
    // thread; publishing into the shared texture happens in
    // publish_due_frame(). Returns true when a frame was published since the
    // last call. Call once per render-thread update.
    bool update() noexcept;

    // Promote the decode thread's prepared frame into the D3D12-shared
    // texture. MUST be called from the render thread at the point where the
    // previous GPU frame has synchronized and the current frame's kernels
    // are not yet dispatched (the engine wires this into Pipeline::beginFrame
    // right after its synchronize). Non-blocking; returns true if a frame
    // was published.
    bool publish_due_frame() noexcept;

    void play()  noexcept;
    void pause() noexcept;
    [[nodiscard]] bool is_playing() const noexcept { return _playing; }
    [[nodiscard]] bool is_open() const noexcept    { return _decoder.is_open(); }
    // Mirror of the decode thread's EOF state (the decoder itself is
    // thread-owned).
    [[nodiscard]] bool is_eof() const noexcept     { return _videoEof.load(std::memory_order_relaxed); }

    // Loop: when reached, EOF rewinds to the head and playback continues.
    // With durations known, the rewind is SEAMLESS: each side rewinds on its
    // own thread before its buffer drains (audio pads silence to the boundary
    // first if its track is shorter), so looping never starves or stalls.
    void set_looping(bool on) noexcept;
    [[nodiscard]] bool is_looping() const noexcept { return _looping; }

    // Seek the playhead (seconds) and present the post-seek frame immediately.
    // Works while paused. rewind() is seek-to-head.
    void seek(double sec) noexcept;
    void rewind() noexcept { seek(0.0); }

    [[nodiscard]] uint32_t frame_width()  const noexcept { return _decoder.frame_width(); }
    [[nodiscard]] uint32_t frame_height() const noexcept { return _decoder.frame_height(); }
    [[nodiscard]] double   duration_sec() const noexcept { return _durationCache; }
    // Media playhead (audio-clock- or wall-clock-driven), clamped to [0, duration].
    // NOT the decoder PTS — this is what UI should display.
    [[nodiscard]] double current_time_sec() const noexcept;

    // The Luisa image — sample this from DX kernels via the bindless slot.
    [[nodiscard]] luisa::compute::Image<float>& image() noexcept { return _bridge.image(); }

    // Bridge access — needed for per-frame acquire/release around Luisa work
    // if tighter sync than the natural render-thread ordering is required.
    [[nodiscard]] VideoTextureBridge& bridge() noexcept { return _bridge; }

    // ---------- audio ----------
    // False when the file has no audio track, or NT_ENABLE_AUDIO is off.
    // Clock stays audio-master while muted (mute is a gain, not a stop), so
    // seeking/looping stay sample-accurate with sound off.
    [[nodiscard]] bool has_audio() const noexcept { return _audioActive; }
    void set_audio_volume(float v) noexcept;
    [[nodiscard]] float audio_volume() const noexcept { return _audioVolume; }
    void set_audio_muted(bool on) noexcept;
    [[nodiscard]] bool audio_muted() const noexcept { return _audioMuted; }
    // Seconds of graph pipeline latency compensated when reading the audio
    // clock (node position leads the speakers by one block + endpoint buffer).
    void set_audio_sync_offset(double sec) noexcept { _audioSyncOffsetSec = sec; }
    [[nodiscard]] double audio_sync_offset() const noexcept { return _audioSyncOffsetSec; }
    // Debug: last audio block underflowed (producer couldn't keep up).
    [[nodiscard]] bool audio_starved() const noexcept;

private:
    VideoDecoderD3D11 _decoder;
    VideoTextureBridge _bridge;
    double            _lastWallTimeSec = -1.0;
    double            _mediaTimeSec    = 0.0;
    bool              _playing         = true;
    bool              _looping         = false;

    // ---- video decode side-channel ----
    // ALL decoder calls (ReadSample, D3D11 VP blt, shared-texture publish)
    // run either on the decode thread below or on the render thread while it
    // is parked (open / seek / teardown). The render thread's update() only
    // advances the clock; publish_due_frame() does the shared-texture copy.
    void _video_decode_loop();
    [[nodiscard]] bool _park_video_thread(double timeoutSec) noexcept;

    std::thread        _videoThread;
    std::atomic<bool>  _videoRunning  = false;
    std::atomic<bool>  _videoPauseReq = false;
    std::atomic<bool>  _videoParked   = false;
    // Mirror of the decoder's EOF for the render thread; duration is cached
    // at open (immutable, decoder state is thread-owned). _videoEofPts is the
    // final frame's PTS — the eof branch waits for its display span to end
    // before looping/pausing, matching the last-frame behavior of the old
    // in-loop decoder.
    std::atomic<bool>  _videoEof      = false;
    std::atomic<double> _videoEofPts  = 0.0;
    double             _durationCache = 0.0;
    // 1-deep→K-deep prepared queue: the decode thread prepares frames (VP blt
    // into rotating decoder-internal slots) and pushes {pts, slot}; the render
    // thread publishes from publish_due_frame(). Entries gate on the media
    // clock so frames never display early. Depth = the decoder's slot count,
    // sized from kVideoPrerollSec at open — video banks ~kVideoPrerollSec of
    // decoded frames BEFORE the audio pre-roll gate releases the clock, and
    // keeps that cushion in steady state (absorbs decode hiccups mid-play).
    struct PreparedFrame { double pts; uint32_t slot; };
    static constexpr size_t kPreparedCapacity = 8;   // ≥ decoder's max slot count
    PreparedFrame            _prepared[kPreparedCapacity] = {};
    std::atomic<uint64_t>    _preparedRead  = 0;      // SPSC: render thread
    std::atomic<uint64_t>    _preparedWrite = 0;      // SPSC: decode thread
    uint32_t                 _videoSlots    = 1;      // immutable after open
    [[nodiscard]] size_t prepared_count() const noexcept {
        return static_cast<size_t>(_preparedWrite.load(std::memory_order_acquire)
                                 - _preparedRead.load(std::memory_order_acquire));
    }
    // Media clock published to the decode thread; _playingMirror gates the
    // decode thread during pause.
    std::atomic<double> _clockSec       = 0.0;
    std::atomic<bool>   _playingMirror  = true;
    bool                _publishedSinceUpdate = false;  // render thread only

    // ---- seamless looping ----
    // _loopBoundarySec = max(video, audio) duration, set once at open. At EOF
    // (while looping) the audio pump pads silence up to the boundary then
    // seeks to 0 without draining the ring, and the video decode thread seeks
    // to 0 and keeps publishing with PTS shifted by _videoLoopEpoch (+= the
    // boundary per wrap) — both media clocks stay monotonic and aligned, no
    // render-side seek stall. Boundary <= 0 (unknown durations) falls back to
    // the render-side seek(0) loop path in update().
    std::atomic<bool>   _loopingMirror   = false;
    std::atomic<double> _loopBoundarySec = 0.0;
    std::atomic<double> _videoLoopEpoch  = 0.0;  // decode thread +=, seek() resets (parked)

    // Clock smoother (render thread only): advances at wall rate between
    // corrections toward the raw audio position, clamped to never lead it by
    // more than kClockMaxLeadSec (audio starvation still freezes the video,
    // a couple ms later) nor lag it by more than kClockMaxLagSec (bounded
    // catch-up after hitches).
    double _smoothedSec = 0.0;

    static constexpr double kVideoPrerollSec = 0.12; // video decode-ahead depth (clock-time)
    static constexpr double kClockMaxLeadSec     = 0.002;
    static constexpr double kClockMaxLagSec      = 0.050;

#if NT_ENABLE_AUDIO
    void _wire_audio(std::filesystem::path const& path);
    void _teardown_audio() noexcept;
    // True when the audio track is fully consumed and the ring is drained —
    // the clock falls back to wall time for any video tail longer than audio.
    [[nodiscard]] bool _audio_exhausted() const noexcept;

    // Background decode loop (owns the VideoAudioStream + ring writes). Keeps
    // synchronous MF ReadSample calls OFF the render thread — a cold ReadSample
    // can cost ~10 ms, which at 84 Hz is the whole frame budget. Render thread
    // only checks the node enable gate.
    void _pump_loop();
    // Park the pump thread between iterations (producer quiesce) so the audio
    // thread can drain the ring and the stream can be re-anchored race-free.
    [[nodiscard]] bool _park_pump(double timeoutSec) noexcept;

    VideoAudioStream                   _audioStream;
    AudioRingBuffer                    _audioRing;
    std::shared_ptr<VideoAudioNode>    _audioNode;
    std::shared_ptr<cinder::audio::GainNode> _audioGain;
    std::vector<float>                 _pcmScratch;      // one decode chunk, interleaved float (pump thread only)
    std::thread                        _pumpThread;
    // Pump protocol flags. _pumpPauseReq/_pumpParked: render thread parks the
    // pump before flush/seek; the pump acks via _pumpParked from its parked
    // state, guaranteeing no in-flight ring writes. _cmdSeek executes on the
    // pump thread (it owns the MF reader).
    std::atomic<bool>                  _pumpRunning   = false;
    std::atomic<bool>                  _pumpPauseReq  = false;
    std::atomic<bool>                  _pumpParked    = false;
    std::atomic<bool>                  _cmdSeek       = false;
    std::atomic<double>                _cmdSeekTarget = 0.0;
    bool                               _audioActive          = false;
    bool                               _audioNeedsResumeSeek = false;
    bool                               _audioStarveLogged    = false;  // render thread; one-shot underrun diagnostic
    float                              _audioVolume          = 1.0f;
    bool                               _audioMuted           = false;
    double                             _audioSyncOffsetSec   = 0.03;

    static constexpr double kAudioRingSeconds      = 1.0;  // ring capacity (pow2-rounded)
    static constexpr double kAudioPrerollSeconds   = 0.40; // clock starts once this much is buffered
    static constexpr double kAudioHighWaterSeconds = 0.5;  // pump idles above this level
    static constexpr double kAudioChunkSeconds     = 0.04; // per-read_next decode granularity
#endif
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
