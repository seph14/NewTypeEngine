#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO

// Cinder BEFORE Windows/D3D headers (see CLAUDE.md, VideoDecoderD3D11.h for the rule).
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

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <vector>

struct IMFSourceReader;

namespace newtype::media {

//
// Audio-only decode side-channel for VideoPlayer.
//
// A SECOND IMFSourceReader on the same file, with every stream deselected
// except FIRST_AUDIO_STREAM, outputting float32 PCM at a caller-chosen rate
// and channel count (MF inserts the decoder + resampler/channel mixer
// automatically). Audio decode is cheap software work — no D3D manager, no
// hardware transforms needed.
//
// Why a second reader instead of sharing the video reader: the synchronous
// SourceReader is not thread-safe, and the video loop + audio pump have
// different pacing. Two instances demux the same local file independently,
// which is negligible cost and decouples threading completely. (This is also
// why VideoAudioStream must be driven from ONE thread only — the render
// thread, same as VideoDecoderD3D11.)
//
// Position: PTS of the first sample after open/seek anchors an internal
// counter that then advances by frames/rate — more robust than per-sample
// PTS juggling. This reader-side position is diagnostic only; the playback
// clock is VideoAudioNode::position_sec().
//
class VideoAudioStream {
public:
    VideoAudioStream() = default;
    ~VideoAudioStream();
    VideoAudioStream(VideoAudioStream const&) = delete;
    VideoAudioStream& operator=(VideoAudioStream const&) = delete;
    VideoAudioStream(VideoAudioStream&&) = delete;
    VideoAudioStream& operator=(VideoAudioStream&&) = delete;

    // Fails quietly (logs at INFO) when the file has no audio stream —
    // callers should treat a false return as "play video-only".
    bool open(std::filesystem::path const& path,
              uint32_t outputSampleRate, uint32_t outputChannels);

    // Decode the next audio PCM frames into dst (interleaved float, `channels()`
    // stride). MF hands out whole MFT output packets (typically 1024+ frames for
    // AAC) which can exceed maxFrames — the remainder is carried internally and
    // delivered by subsequent calls, so NO data is ever dropped: call with any
    // maxFrames you like. Returns true with outFrames possibly 0 on a no-data
    // event; returns false on error or once EOF was reached.
    bool read_next(/*out*/ float* dst, size_t maxFrames, /*out*/ size_t& outFrames);

    // Demuxer-granular seek. Position re-anchors on the first post-seek PTS.
    bool seek(double sec);

    void close();

    [[nodiscard]] bool     is_open()      const noexcept { return _reader != nullptr; }
    [[nodiscard]] bool     is_eof()       const noexcept { return _eof; }
    [[nodiscard]] uint32_t sample_rate()  const noexcept { return _sampleRate; }
    [[nodiscard]] uint32_t channels()     const noexcept { return _channels; }
    [[nodiscard]] double   duration_sec() const noexcept { return _durationSec; }
    // Reader-side decode position (diagnostics; NOT the playback clock).
    [[nodiscard]] double   position_sec() const noexcept { return _positionSec; }
    // Media-time PTS the current decode run is anchored at: the first sample
    // decoded after open/seek. The pre-roll gate primes the audio node's
    // epoch from this so edit lists / encoder priming don't bias lip sync.
    // Written by the pump thread at anchor time; publication to the render
    // thread rides the ring-write release (the gate only reads it once
    // post-anchor frames are already buffered).
    [[nodiscard]] double   anchor_sec() const noexcept { return _anchorSec.load(std::memory_order_relaxed); }

private:
    struct IMFSourceReader* _reader       = nullptr;
    uint32_t                _sampleRate   = 48000;
    uint32_t                _channels     = 2;
    double                  _durationSec  = 0.0;
    double                  _positionSec  = 0.0;
    // Same publish rule as position_sec()'s anchor: written by the pump
    // thread when the first post-open/seek sample lands, read by the render
    // thread's enable gate (after ring writes publish it).
    std::atomic<double>     _anchorSec    = 0.0;
    bool                    _anchorPending = false;
    // Atomic: written by the decode (pump) thread, polled by the render thread
    // for the audio-clock fallback and the node enable gate.
    std::atomic<bool>       _eof          = false;

    // Leftover frames of the last MFT packet when the caller's maxFrames was
    // smaller than the packet (frame-major interleaved, same layout as dst).
    std::vector<float>      _carry;
    size_t                  _carryFrames  = 0;
    size_t                  _carryOffset  = 0;

    static int s_instanceCount;            // for MFStartup/Shutdown refcount
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO
