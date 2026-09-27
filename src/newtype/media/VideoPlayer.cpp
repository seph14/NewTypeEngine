#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER

#include "newtype/media/VideoPlayer.h"

#if NT_ENABLE_AUDIO
#include "newtype/media/VideoAudioNode.h"

#include "cinder/audio/Context.h"
#include "cinder/audio/GainNode.h"
#endif

#include "cinder/app/App.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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
    // kVideoPrerollSec of internal slots: the decode thread banks that much
    // video before the audio pre-roll gate releases the clock (and keeps it
    // as steady-state hiccup slack).
    if (!_decoder.open(path, d3d12Device, VideoDecoderD3D11::Mode::HardwareNV12,
                       kVideoPrerollSec)) {
        return false;
    }
    _videoSlots = _decoder.internal_slot_count();
    if (!_bridge.init(luisaDevice, d3d12Device,
                      _decoder.shared_handle(),
                      _decoder.frame_width(), _decoder.frame_height(),
                      _decoder.keyed_mutex())) {
        _decoder.close();
        return false;
    }

#if NT_ENABLE_AUDIO
    // Audio side-channel — optional by design: no audio track or MF commit
    // failure leaves a fully working video-only player (wall-clock master).
    _wire_audio(path);
#endif

    _lastWallTimeSec = -1.0;
    _mediaTimeSec    = 0.0;
    _smoothedSec     = 0.0;
    _durationCache   = _decoder.duration_sec();
    _videoLoopEpoch.store(0.0, std::memory_order_relaxed);
#if NT_ENABLE_AUDIO
    _loopBoundarySec.store(
        std::max(_durationCache, _audioActive ? _audioStream.duration_sec() : 0.0),
        std::memory_order_relaxed);
#else
    _loopBoundarySec.store(_durationCache, std::memory_order_relaxed);
#endif

#if NT_ENABLE_AUDIO
    char const* audioState = _audioActive ? "yes" : "no";
#else
    char const* audioState = "n/a (NT_ENABLE_AUDIO=0)";
#endif
    // Present frame 0 immediately (on the render thread, before the decode
    // thread exists — no concurrent decoder user). The decode loop takes over
    // from frame 1; without this the shared texture would show initial zeros
    // for one frame interval.
    _decoder.read_next_frame();
    _publishedSinceUpdate = true;
    _videoEof.store(false, std::memory_order_relaxed);
    _videoEofPts.store(0.0, std::memory_order_relaxed);
    _preparedRead.store(0, std::memory_order_relaxed);
    _preparedWrite.store(0, std::memory_order_relaxed);
    _playingMirror.store(_playing, std::memory_order_relaxed);
    _clockSec.store(0.0, std::memory_order_relaxed);
    _videoRunning.store(true, std::memory_order_relaxed);
    _videoThread = std::thread([this] { _video_decode_loop(); });

    CI_LOG_D("VideoPlayer: opened '" << path.string()
             << "' " << _decoder.frame_width() << "x" << _decoder.frame_height()
             << ", duration=" << _decoder.duration_sec() << "s"
             << ", frameDur=" << _decoder.frame_duration_sec() << "s"
             << ", audio=" << audioState);
    return true;
}

void VideoPlayer::close() noexcept {
#if NT_ENABLE_AUDIO
    _teardown_audio();
#endif
    // Stop the decode thread before touching decoder/bridge — between parks
    // it owns the reader and the D3D11 context.
    if (_videoThread.joinable()) {
        _videoRunning.store(false, std::memory_order_relaxed);
        _videoPauseReq.store(false, std::memory_order_relaxed);  // unstick a parked wait
        _videoThread.join();
    }
    _videoPauseReq.store(false, std::memory_order_relaxed);
    _videoParked.store(false, std::memory_order_relaxed);
    _preparedRead.store(_preparedWrite.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);   // drain
    _bridge.close();
    _decoder.close();
    _lastWallTimeSec = -1.0;
    _mediaTimeSec    = 0.0;
    _smoothedSec     = 0.0;
}

#if NT_ENABLE_AUDIO

void VideoPlayer::_wire_audio(std::filesystem::path const& path) {
    try {
        auto* ctx = ci::audio::Context::master();
        if (!ctx->isEnabled()) ctx->enable();

        uint32_t const rate     = static_cast<uint32_t>(ctx->getSampleRate());
        uint32_t constexpr channels = 2;

        if (!_audioStream.open(path, rate, channels)) return;

        // Ring capacity: ≥1 s, power of two (AudioRingBuffer requirement).
        size_t cap = 1024;
        while (cap < static_cast<size_t>(rate * kAudioRingSeconds)) cap <<= 1;
        _audioRing.configure(channels, cap);

        _pcmScratch.resize(static_cast<size_t>(rate * kAudioChunkSeconds) * channels);

        // autoEnable(false) is load-bearing: the master context is already
        // running, so an auto-enabled node would start pulling from an empty
        // ring while the MF pump is still cold — the audio clock would limp
        // along at decode speed and stagger playback. The pre-roll gate in
        // update() owns the enable instead.
        ci::audio::Node::Format nodeFormat;
        nodeFormat.autoEnable(false);
        _audioNode = ctx->makeNode<VideoAudioNode>(_audioRing, rate, channels, nodeFormat);
        _audioGain = ctx->makeNode(new ci::audio::GainNode(_audioMuted ? 0.0f : _audioVolume));
        _audioNode >> _audioGain >> ctx->getOutput();

        // Start the background decode pump. It fills the ring to the high-water
        // level and then idles; the node (and with it the audio clock) stays
        // disabled until update()'s enable gate sees the pre-roll buffered.
        _pumpRunning.store(true, std::memory_order_relaxed);
        _pumpThread = std::thread([this] { _pump_loop(); });

        _audioActive = true;
    }
    catch (std::exception const& e) {
        // Audio graph unavailable — degrade to video-only, don't fail the open.
        CI_LOG_W("VideoPlayer: audio wiring failed, playing video-only: " << e.what());
        _teardown_audio();
    }
}

void VideoPlayer::_teardown_audio() noexcept {
    _audioActive = false;
    if (_pumpThread.joinable()) {
        _pumpRunning.store(false, std::memory_order_relaxed);
        _pumpPauseReq.store(false, std::memory_order_relaxed);  // unstick the parked wait
        _pumpThread.join();
    }
    _pumpPauseReq.store(false, std::memory_order_relaxed);
    _pumpParked.store(false, std::memory_order_relaxed);
    _cmdSeek.store(false, std::memory_order_relaxed);
    if (_audioNode) {
        try {
            _audioNode->setEnabled(false);
            _audioNode->disconnectAll();
        }
        catch (...) {} // audio context may already be torn down at app exit
        _audioNode.reset();
    }
    if (_audioGain) {
        try { _audioGain->disconnectAll(); } catch (...) {}
        _audioGain.reset();
    }
    _audioStream.close();
}

void VideoPlayer::_pump_loop() {
    size_t const highWaterFrames = static_cast<size_t>(
        _audioStream.sample_rate() * kAudioHighWaterSeconds);
    size_t const chunkFrames = static_cast<size_t>(
        _audioStream.sample_rate() * kAudioChunkSeconds);
    if (_pcmScratch.size() < chunkFrames * _audioStream.channels())
        _pcmScratch.resize(chunkFrames * _audioStream.channels());

    // Seamless-loop state (pump-local). Silence padding keeps the node clock
    // running linearly across the wrap when the audio track is shorter than
    // the video; re-armed after each rewind and after explicit seeks.
    size_t const boundaryFrames = static_cast<size_t>(
        _audioStream.sample_rate() * _loopBoundarySec.load(std::memory_order_relaxed));
    size_t const trackFrames = static_cast<size_t>(
        _audioStream.sample_rate() * _audioStream.duration_sec());
    size_t const padFramesPerLoop =
        (boundaryFrames > trackFrames) ? (boundaryFrames - trackFrames) : 0;
    std::vector<float> const silence(chunkFrames * _audioStream.channels(), 0.0f);
    size_t padRemaining = padFramesPerLoop;

    while (_pumpRunning.load(std::memory_order_relaxed)) {
        // Parked state: render thread is (or may be) draining the ring or
        // re-homing the clock; touch nothing until unparked.
        if (_pumpPauseReq.load(std::memory_order_acquire)) {
            _pumpParked.store(true, std::memory_order_release);
            while (_pumpPauseReq.load(std::memory_order_acquire) &&
                   _pumpRunning.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            _pumpParked.store(false, std::memory_order_release);
            if (!_pumpRunning.load(std::memory_order_relaxed)) break;
            if (_cmdSeek.exchange(false, std::memory_order_acq_rel)) {
                _audioStream.seek(_cmdSeekTarget.load(std::memory_order_relaxed));
                padRemaining = padFramesPerLoop;
            }
            continue;
        }
        if (_cmdSeek.exchange(false, std::memory_order_acq_rel)) {
            _audioStream.seek(_cmdSeekTarget.load(std::memory_order_relaxed));
            padRemaining = padFramesPerLoop;
            continue;
        }

        if (_audioStream.is_eof()) {
            if (_loopingMirror.load(std::memory_order_relaxed)
                && _loopBoundarySec.load(std::memory_order_relaxed) > 0.0) {
                // Seamless loop: pad silence up to the boundary, then rewind
                // on this thread. The ring never drains while looping, so the
                // node clock keeps advancing and no starvation gap opens.
                if (padRemaining > 0) {
                    size_t const n = std::min({chunkFrames, _audioRing.free_frames(), padRemaining});
                    if (n > 0) {
                        _audioRing.write(silence.data(), n);
                        padRemaining -= n;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                } else if (_audioStream.seek(0.0)) {
                    padRemaining = padFramesPerLoop;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }
        if (_audioRing.buffered_frames() >= highWaterFrames) {
            // 5 ms idle poll: rare wakeups, still refills fast after a drain
            // (a block is ~10.7 ms, so the cushion absorbs the latency).
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        size_t produced = 0;
        if (!_audioStream.read_next(_pcmScratch.data(), chunkFrames, produced)) {
            if (!_audioStream.is_eof()) {
                // Read error (not EOF): back off instead of hot-spinning.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            continue;
        }
        if (produced > 0) {
            _audioRing.write(_pcmScratch.data(), produced);
        }
    }
}

bool VideoPlayer::_park_pump(double timeoutSec) noexcept {
    if (!_pumpThread.joinable()) return false;
    _pumpParked.store(false, std::memory_order_relaxed);
    _pumpPauseReq.store(true, std::memory_order_release);
    double waited = 0.0;
    while (!_pumpParked.load(std::memory_order_acquire)) {
        if (waited >= timeoutSec) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited += 0.001;
    }
    return true;
}

bool VideoPlayer::_audio_exhausted() const noexcept {
    return _audioActive && _audioStream.is_eof() && _audioRing.buffered_frames() == 0;
}

bool VideoPlayer::audio_starved() const noexcept {
    return _audioActive && _audioNode && _audioNode->is_starved();
}

void VideoPlayer::set_audio_volume(float v) noexcept {
    _audioVolume = std::clamp(v, 0.0f, 1.0f);
    if (_audioGain && !_audioMuted) _audioGain->setValue(_audioVolume);
}

void VideoPlayer::set_audio_muted(bool on) noexcept {
    _audioMuted = on;
    if (_audioGain) _audioGain->setValue(on ? 0.0f : _audioVolume);
}

#else // !NT_ENABLE_AUDIO

void VideoPlayer::set_audio_volume(float) noexcept {}
void VideoPlayer::set_audio_muted(bool) noexcept {}
bool VideoPlayer::audio_starved() const noexcept { return false; }

#endif // NT_ENABLE_AUDIO

void VideoPlayer::set_looping(bool on) noexcept {
    _looping = on;
    _loopingMirror.store(on, std::memory_order_relaxed);
}

void VideoPlayer::play() noexcept {
    _playing = true;
    _playingMirror.store(true, std::memory_order_relaxed);
    _lastWallTimeSec = -1.0;
#if NT_ENABLE_AUDIO
    if (_audioActive && _audioNode) {
        // After a pause the ring was flushed and the pump is parked; the MF
        // reader still sits AHEAD of the audible position (pause dropped the
        // unconsumed backlog). Re-anchor the reader at the clock — the seek
        // executes on the pump thread as it unparks.
        if (_audioNeedsResumeSeek) {
            // position_sec() leads the speakers by the pipeline depth; back
            // off the sync offset so the reader re-decodes from what was
            // actually heard — otherwise every pause/resume permanently
            // skips that much audio. While looping, the node clock may have
            // wrapped past the track: fold the target back into [0, boundary)
            // so the MF seek lands inside the file.
            double resumeSec = std::max(0.0, _audioNode->position_sec() - _audioSyncOffsetSec);
            double const boundary = _loopBoundarySec.load(std::memory_order_relaxed);
            if (boundary > 0.0) resumeSec = std::fmod(resumeSec, boundary);
            _cmdSeekTarget.store(resumeSec, std::memory_order_relaxed);
            _cmdSeek.store(true, std::memory_order_release);
            _audioNeedsResumeSeek = false;
        }
        _pumpPauseReq.store(false, std::memory_order_release);
    }
#endif
}

void VideoPlayer::pause() noexcept {
    _playing = false;
    _playingMirror.store(false, std::memory_order_relaxed);
#if NT_ENABLE_AUDIO
    if (_audioActive && _audioNode) {
        if (!_audioNode->isEnabled()) {
            // Never started (still pre-rolling or before first play): nothing
            // is audible and there is no backlog to drop — leave the pump
            // filling so the next play() starts with a full pre-roll.
            return;
        }
        // Drop the unconsumed backlog so audio stops promptly; the clock keeps
        // the position the speakers actually reached. Park the producer first
        // so the audio thread can drain race-free; the reader re-anchors in
        // play(). If the park times out, leave things running — the backlog
        // (≤ high-water) simply plays out.
        if (_park_pump(0.1)) {
            _audioNode->request_flush();
            _audioNeedsResumeSeek = true;
        } else {
            CI_LOG_W("VideoPlayer::pause: pump park timed out; audio backlog will play out");
        }
    }
#endif
}

void VideoPlayer::seek(double sec) noexcept {
    if (!_decoder.is_open()) return;
    // Park the decode thread first — everything below touches the decoder
    // (MF reader + D3D11 context) from the render thread. Unlike the audio
    // side, a failed park must abort: concurrent decoder access corrupts it.
    if (!_park_video_thread(0.2)) {
        CI_LOG_W("VideoPlayer::seek: decode thread park timed out; seek skipped");
        return;
    }
    if (!_decoder.seek(sec)) {
        _videoPauseReq.store(false, std::memory_order_release);
        return;
    }
    // Drop any prepared-but-unpublished frames; the inline present below is
    // the authoritative post-seek image. Explicit seeks collapse any loop
    // epochs — PTS are file-local again from here on.
    _preparedRead.store(_preparedWrite.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);   // drain (decode thread parked)
    _videoEof.store(false, std::memory_order_relaxed);
    _videoEofPts.store(0.0, std::memory_order_relaxed);
    _videoLoopEpoch.store(0.0, std::memory_order_relaxed);
    _lastWallTimeSec = -1.0;   // dt = 0 on next update — no time jump

#if NT_ENABLE_AUDIO
    if (_audioActive && _audioNode) {
        if (_park_pump(0.1)) {
            if (_audioNode->isEnabled()) {
                // Jump the audio clock and drain the ring while the producer
                // is parked, then let the pump re-anchor the reader at the
                // target as it resumes. wait_flushed quiesces until the audio
                // thread applied the rebase, so the pump's post-seek writes
                // land after the drain.
                _audioNode->rebase_to(sec);
                (void)_audioNode->wait_flushed(0.1);
            } else {
                // Node never ran, so cinder never pulls its process(): no
                // consumer exists and the ring can be drained directly. The
                // epoch is primed from the decoder's post-seek anchor when
                // the pre-roll gate enables the node in update().
                _audioRing.reset();
            }
            _cmdSeekTarget.store(sec, std::memory_order_relaxed);
            _cmdSeek.store(true, std::memory_order_release);
            // Stay parked while paused: un-parking would let the pump refill
            // and the enabled node consume it, drifting the clock past the
            // seek target while the video stands still. play() releases the
            // park and the pending seek executes on the pump thread.
            if (_playing) _pumpPauseReq.store(false, std::memory_order_release);
            _audioNeedsResumeSeek = false;
            _audioStarveLogged = false;
        } else {
            CI_LOG_W("VideoPlayer::seek: pump park timed out; audio desyncs until next seek");
        }
    }
#endif

    // Present the post-seek frame now and rebase the clock to its delivered
    // PTS: SetCurrentPosition is keyframe-granular, so the frame may sit
    // before the target; rebasing skips re-playing that pre-roll. (With audio
    // live, the next update() overwrites _mediaTimeSec from the audio clock at
    // the seek target — same skip-the-preroll semantics.) The decode thread is
    // parked, so this synchronous decode+publish is the only decoder user.
    _decoder.read_next_frame();
    _mediaTimeSec = _smoothedSec = _decoder.current_time_sec();
    _clockSec.store(_mediaTimeSec, std::memory_order_relaxed);
    _publishedSinceUpdate = true;
    _videoPauseReq.store(false, std::memory_order_release);
}

bool VideoPlayer::update() noexcept {
    if (!_decoder.is_open() || !_playing) return false;
    bool const published = _publishedSinceUpdate;
    _publishedSinceUpdate = false;

    // Latched EOF (non-loop playback, or seamless-loop rewind failure):
    // end state for non-loop playback; loop mode falls back to a render-side
    // seek(0) here only when the decode thread could not self-rewind.
    // Deferred until the final frame was published AND its display span
    // ended — acting at publish time would clip the last frame short by up
    // to one frame duration.
    if (_videoEof.load(std::memory_order_relaxed) && prepared_count() == 0
        && _mediaTimeSec >= _videoEofPts.load(std::memory_order_relaxed)
                            + _decoder.frame_duration_sec()) {
        if (!_looping) {
            _playing = false;
            _playingMirror.store(false, std::memory_order_relaxed);
            return published;
        }
        seek(0.0);
        if (_videoEof.load(std::memory_order_relaxed)) return published;   // seek failed
    }

    double now = 0.0;
    double dt  = 0.0;
    {
        // Wall-clock bookkeeping runs every tick so a mid-playback fallback to
        // wall time (audio exhausted, below) doesn't inherit a stale baseline.
        now = ci::app::getElapsedSeconds();
        dt  = (_lastWallTimeSec > 0.0) ? std::min(now - _lastWallTimeSec, 0.1) : 0.0;
        _lastWallTimeSec = now;
    }

    // Media clock. Three modes: audio-master (smoothed toward the raw audio
    // position), wall (video-only files, and video tails past audio EOF), and
    // hold (pre-roll gate pending — keep where open()/seek() left it).
    enum class ClockMode { Hold, Wall, AudioMaster };
    ClockMode clockMode = ClockMode::Wall;
#if NT_ENABLE_AUDIO
    if (_audioActive) {
        // Pre-roll gate: the node is created autoEnable(false) and stays
        // silent until the ring holds a full pre-roll (or the audio track hit
        // EOF — tracks shorter than the pre-roll still play). Enabling the
        // node is what starts the audio clock, so audio and video begin
        // together instead of the node draining a half-warm ring while MF
        // spins up.
        if (_audioNode && !_audioNode->isEnabled()) {
            size_t const prerollFrames = static_cast<size_t>(
                _audioStream.sample_rate() * kAudioPrerollSeconds);
            if (_audioRing.buffered_frames() >= prerollFrames || _audioStream.is_eof()) {
                // A disabled node's process() is never pulled, so the clock
                // can be set directly: anchor the epoch at the decoder's
                // first-sample PTS (covers edit lists / encoder priming) and
                // clear any stale flush from a pre-start pause.
                _audioNode->prime_epoch(_audioStream.anchor_sec());
                _audioNode->setEnabled(true);
                _audioStarveLogged = false;
            }
            clockMode = ClockMode::Hold;
        } else if (_audioNode) {
            clockMode = _audio_exhausted() ? ClockMode::Wall : ClockMode::AudioMaster;
        }
    }
#endif

    switch (clockMode) {
        case ClockMode::Hold:
            _mediaTimeSec = _smoothedSec;
            break;
        case ClockMode::Wall:
            // Wall-clock seconds, NOT mDt. mDt is gated by mTick in
            // NewTypeEngine::update and collapses to 0 when tick is off —
            // feeding that to a media clock would freeze playback. See
            // [[feedback-mtick-mdt-conflation]]. The dt clamp bounds catch-up
            // to ≤ 0.1 s worth of frames per update, so a window drag or long
            // stall can't trigger a decode storm.
            _smoothedSec += dt;
            _mediaTimeSec = _smoothedSec;
            break;
        case ClockMode::AudioMaster:
#if NT_ENABLE_AUDIO
        {
            // The raw audio position advances one ~10.7 ms block per audio
            // callback; deriving the media clock directly makes frame swaps
            // land in bursts. Advance at wall rate and correct toward the
            // position instead — clamped to never lead it by more than
            // kClockMaxLeadSec (audio starvation still freezes the video, a
            // couple of ms later) nor trail it by more than kClockMaxLagSec
            // (bounded catch-up after a hitch).
            double const target =
                std::max(0.0, _audioNode->position_sec() - _audioSyncOffsetSec);
            _smoothedSec += dt;
            if (_smoothedSec > target + kClockMaxLeadSec) {
                _smoothedSec = target + kClockMaxLeadSec;
            } else if (target - _smoothedSec > kClockMaxLagSec) {
                _smoothedSec = target - kClockMaxLagSec;
            }
            if (_smoothedSec < 0.0) _smoothedSec = 0.0;
            _mediaTimeSec = _smoothedSec;
            break;
        }
#endif // unreachable (and unselected) with NT_ENABLE_AUDIO=0
    }

#if NT_ENABLE_AUDIO
    // One-shot underrun diagnostic (re-armed per enable/seek): a starved
    // block means an audible gap; decoded-vs-clock tells whether the pump was
    // late or the decoder was slow. Suppressed once the audio track hit EOF —
    // silence past the end of a shorter audio track is expected (video tail
    // playback), not a starvation bug.
    if (_audioActive && _audioNode && _audioNode->isEnabled()
        && !_audioStream.is_eof()
        && !_audioStarveLogged && _audioNode->is_starved()) {
        CI_LOG_W("VideoPlayer: audio underrun at " << _mediaTimeSec
                 << "s (decoded=" << _audioStream.position_sec()
                 << "s, ring=" << _audioRing.buffered_frames()
                 << "f, starvedBlocks=" << _audioNode->starved_blocks() << ")");
        _audioStarveLogged = true;
    }
#endif

    // The decode thread chases this clock; publish_due_frame() — called from
    // the engine's beginFrame right after its GPU synchronize — promotes
    // prepared frames into the shared texture.
    _clockSec.store(_mediaTimeSec, std::memory_order_release);
    return published;
}

bool VideoPlayer::publish_due_frame() noexcept {
    // Render thread, between the previous GPU frame's synchronize and the
    // current frame's kernel dispatches (the engine wires this into
    // Pipeline::beginFrame). The GPU is momentarily drained, so the
    // keyed-mutex copy + completion wait cost a fraction of what they would
    // mid-frame, and no in-flight kernel can sample a half-written texture.
    if (!_playing) return false;

    // Pop every entry whose PTS has been reached; publish the newest one and
    // drop the older ones — they are already past their display span, so
    // rendering them for a single engine frame each would just add judder.
    uint64_t r = _preparedRead.load(std::memory_order_relaxed);
    uint64_t const w = _preparedWrite.load(std::memory_order_acquire);
    uint32_t dueSlot = 0;
    bool     haveDue = false;
    while (r < w) {
        PreparedFrame const& f = _prepared[r & (kPreparedCapacity - 1)];
        if (f.pts > _mediaTimeSec) break;
        dueSlot = f.slot;
        haveDue = true;
        ++r;
    }
    if (!haveDue) return false;
    _preparedRead.store(r, std::memory_order_release);
    _decoder.publish_frame_to_shared(dueSlot);
    _publishedSinceUpdate = true;
    return true;
}

void VideoPlayer::_video_decode_loop() {
    while (_videoRunning.load(std::memory_order_relaxed)) {
        // Parked state: render thread owns the decoder (seek / teardown).
        if (_videoPauseReq.load(std::memory_order_acquire)) {
            _videoParked.store(true, std::memory_order_release);
            while (_videoPauseReq.load(std::memory_order_acquire) &&
                   _videoRunning.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            _videoParked.store(false, std::memory_order_release);
            if (!_videoRunning.load(std::memory_order_relaxed)) break;
            continue;
        }

        if (_videoEof.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (prepared_count() >= _videoSlots) {
            // Prepared queue full: hold the next decode until the render
            // thread published the oldest entries. The count gate (slots,
            // not capacity) guarantees the slot about to be overwritten was
            // already published.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (!_playingMirror.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        double const clock = _clockSec.load(std::memory_order_acquire);
        double const epoch = _videoLoopEpoch.load(std::memory_order_relaxed);
        // Decode ahead of the clock by ~kVideoPrerollSec (PTS are
        // epoch-shifted across loop wraps so the pacing holds on every
        // iteration). While the audio pre-roll gate holds the clock at 0
        // this fills the queue before playback starts; afterwards it is
        // steady-state hiccup slack.
        if (_decoder.current_time_sec() + epoch + _decoder.frame_duration_sec()
                > clock + kVideoPrerollSec) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (!_decoder.decode_next_frame()) {
            if (_decoder.is_eof()) {
                double const boundary = _loopBoundarySec.load(std::memory_order_relaxed);
                if (_loopingMirror.load(std::memory_order_relaxed) && boundary > 0.0) {
                    // Seamless loop: rewind on this thread (we own the
                    // decoder between parks) and keep banking head frames.
                    // PTS shift by the boundary keeps the queue monotonic
                    // and aligned with the audio's boundary; the render-side
                    // seek stall never happens.
                    if (_decoder.seek(0.0)) {
                        _videoLoopEpoch.fetch_add(boundary, std::memory_order_relaxed);
                        continue;
                    }
                    CI_LOG_W("VideoPlayer: loop rewind failed; using render-side seek fallback");
                }
                _videoEofPts.store(
                    _decoder.current_time_sec() + _videoLoopEpoch.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                _videoEof.store(true, std::memory_order_relaxed);
                CI_LOG_D("VideoPlayer: video EOF at " << _decoder.current_time_sec() << "s");
            } else {
                // Transient no-sample event or decode error: back off, retry.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }
        PreparedFrame f{_decoder.current_time_sec()
                            + _videoLoopEpoch.load(std::memory_order_relaxed),
                        _decoder.last_internal_slot()};
        _prepared[_preparedWrite.load(std::memory_order_relaxed) & (kPreparedCapacity - 1)] = f;
        _preparedWrite.fetch_add(1, std::memory_order_release);
    }
}

bool VideoPlayer::_park_video_thread(double timeoutSec) noexcept {
    if (!_videoThread.joinable()) return false;
    _videoParked.store(false, std::memory_order_relaxed);
    _videoPauseReq.store(true, std::memory_order_release);
    double waited = 0.0;
    while (!_videoParked.load(std::memory_order_acquire)) {
        if (waited >= timeoutSec) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited += 0.001;
    }
    return true;
}

double VideoPlayer::current_time_sec() const noexcept {
    // While looping, the media clock is monotonic across wraps — fold it back
    // into [0, boundary) for display/seek UI.
    double t = _mediaTimeSec;
    double const boundary = _loopBoundarySec.load(std::memory_order_relaxed);
    if (_looping && boundary > 0.0) t = std::fmod(t, boundary);
    if (t < 0.0) t = 0.0;
    double const dur = _durationCache;
    if (dur > 0.0 && t > dur) return dur;
    return t;
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER
