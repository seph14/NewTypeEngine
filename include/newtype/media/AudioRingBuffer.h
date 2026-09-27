#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace newtype::media {

//
// Single-producer / single-consumer lock-free ring of interleaved float frames.
//
// Producer = the render-thread MF audio decode pump (VideoPlayer); consumer =
// VideoAudioNode::process() on the cinder audio thread. Free-running 64-bit
// cursors + power-of-two capacity mean no wraparound handling at the API level.
// Indexing uses acquire/release so the consumer's read of _write never passes
// the producer's release of newly written frames (and vice versa).
//
// reset() drains (read cursor = write cursor) rather than zeroing both, so a
// flush racing an in-flight producer write degrades to "those frames are
// dropped" instead of corrupting the cursors — flush semantics are
// "discard everything buffered", which stays true.
//
class AudioRingBuffer {
public:
    // capacityFrames must be a power of two. channels is the interleaved
    // channel count; both sides must agree (we always run stereo float32).
    void configure(uint32_t channels, size_t capacityFramesPow2) {
        _channels = channels;
        _capacity = capacityFramesPow2;
        _mask     = _capacity - 1;
        _data.assign(_capacity * _channels, 0.0f);
        _read.store(0, std::memory_order_relaxed);
        _write.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] size_t buffered_frames() const {
        return _write.load(std::memory_order_acquire) - _read.load(std::memory_order_acquire);
    }
    [[nodiscard]] size_t free_frames() const {
        return _capacity - buffered_frames();
    }
    [[nodiscard]] uint32_t channels() const noexcept { return _channels; }
    [[nodiscard]] size_t   capacity() const noexcept { return _capacity; }

    // Producer thread only. Returns frames actually accepted (never partial
    // frame granularity issues — callers pass whole frames).
    size_t write(float const* src, size_t frames) {
        size_t const w = _write.load(std::memory_order_relaxed);
        size_t const r = _read.load(std::memory_order_acquire);
        size_t const n = std::min(frames, _capacity - (w - r));
        if (n == 0) return 0;
        _copy_in(w, src, n);
        _write.store(w + n, std::memory_order_release);
        return n;
    }

    // Consumer thread only (audio thread).
    size_t read(float* dst, size_t frames) {
        size_t const r = _read.load(std::memory_order_relaxed);
        size_t const w = _write.load(std::memory_order_acquire);
        size_t const n = std::min(frames, w - r);
        if (n == 0) return 0;
        _copy_out(r, dst, n);
        _read.store(r + n, std::memory_order_release);
        return n;
    }

    // Consumer side of the flush handshake (VideoAudioNode::process under the
    // flush flag), with the producer quiesced between request and completion.
    void reset() {
        size_t const w = _write.load(std::memory_order_acquire);
        _read.store(w, std::memory_order_release);
    }

private:
    void _copy_in(size_t cursor, float const* src, size_t frames) {
        size_t const first = std::min(frames, _capacity - (cursor & _mask));
        std::memcpy(&_data[(cursor & _mask) * _channels], src,
                    first * _channels * sizeof(float));
        std::memcpy(&_data[0], src + first * _channels,
                    (frames - first) * _channels * sizeof(float));
    }
    void _copy_out(size_t cursor, float* dst, size_t frames) const {
        size_t const first = std::min(frames, _capacity - (cursor & _mask));
        std::memcpy(dst, &_data[(cursor & _mask) * _channels],
                    first * _channels * sizeof(float));
        std::memcpy(dst + first * _channels, &_data[0],
                    (frames - first) * _channels * sizeof(float));
    }

    std::atomic<uint64_t> _read  = 0;
    std::atomic<uint64_t> _write = 0;
    std::vector<float>    _data;
    uint32_t              _channels = 2;
    size_t                _capacity = 0;
    size_t                _mask     = 0;
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO
