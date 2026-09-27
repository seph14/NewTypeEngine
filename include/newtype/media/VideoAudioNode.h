#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO

#include "cinder/audio/InputNode.h"

#include "newtype/media/AudioRingBuffer.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace newtype::media {

//
// Audio-graph source node that pulls interleaved float frames from an
// AudioRingBuffer (filled by the video's background MF decode pump) and
// tracks a sample-accurate read position, mirroring BufferPlayerNode's
// mReadPos. VideoPlayer masters its media clock off position_sec().
//
// Layout: the ring is INTERLEAVED (MF delivers interleaved PCM); the Buffer
// cinder hands to process() is PLANAR (channel planes stored consecutively —
// see BufferT::getChannel). process() stages through an interleaved scratch
// and de-interleaves on the fly. Writing interleaved data straight into the
// planar buffer scrambles L/R across both planes — full-scale noise.
//
// Threading: process() runs on the audio thread; everything else is called
// from the render thread. While ENABLED, the render thread never mutates
// ring contents — drains/rebases go through the flush handshake:
//
//   rebase_to(sec) / request_flush()   [render thread]
//     stores mPendingEpochFrames, raises mFlushRequested
//   wait_flushed(timeout)              [render thread, bounded spin]
//   process() observes the flag at the next block boundary, latches the new
//     epoch, drains the ring, and outputs silence for that block.
//
// While DISABLED, cinder never calls process() (Node::pullInputs gates on
// mEnabled), so there is no audio-thread side at all: prime_epoch() may set
// the clock and the ring may be drained directly, no handshake needed.
//
// The node is intentionally never disabled during pause/seek — pause just
// stops feeding the ring (position freezes, silence plays), which keeps the
// handshake live and avoids disable/enable races with in-flight blocks.
//
class VideoAudioNode : public cinder::audio::InputNode {
public:
    VideoAudioNode(AudioRingBuffer& ring, uint32_t sampleRate, uint32_t channels,
                   cinder::audio::Node::Format const& format = {});

    // --- render thread ---

    // Jump the clock to `sec` and drop buffered frames. Requires the node to
    // be enabled (the drain happens on the audio thread) — pair with
    // wait_flushed() when the exact switch-over moment matters.
    void rebase_to(double sec);
    // Drop buffered frames without moving the clock (pause semantics).
    void request_flush();
    // Bounded spin until the audio thread applied the pending flush.
    [[nodiscard]] bool wait_flushed(double timeoutSec) const;

    // Set the clock directly while the node is still DISABLED — safe because
    // cinder never pulls process() on a disabled node, so no audio thread
    // exists to race. The pre-roll gate uses this to anchor the epoch at the
    // decoder's first-sample PTS before the first block runs; it also clears
    // any stale flush request so enabling doesn't drain the pre-roll. Must
    // NOT be called once enabled — use rebase_to() then.
    void prime_epoch(double sec);

    // --- any thread ---

    // Audible-playhead estimate: epoch + frames consumed, in seconds. Steps at
    // block granularity and leads the speakers by the graph pipeline depth —
    // VideoPlayer subtracts a constant sync offset to compensate.
    [[nodiscard]] double position_sec() const;
    [[nodiscard]] bool     is_starved()       const { return mStarved.load(std::memory_order_relaxed); }
    [[nodiscard]] uint64_t starved_blocks()   const { return mStarvedCount.load(std::memory_order_relaxed); }

protected:
    void initialize() override;
    void process(cinder::audio::Buffer* buffer) override;

private:
    AudioRingBuffer& mRing;
    uint32_t mSampleRate = 48000;
    uint32_t mChannels   = 2;

    // Clock state. mEpochFrames/mFramesSinceEpoch are advanced ONLY by the
    // audio thread while enabled (and ONLY by the render thread via
    // prime_epoch while disabled); mPendingEpochFrames is written by the
    // render thread before raising the flag, and latched by the audio thread
    // at the flush point (seq_cst ordering makes the two-step publication
    // safe).
    std::atomic<uint64_t> mEpochFrames      = 0;
    std::atomic<uint64_t> mFramesSinceEpoch = 0;
    std::atomic<uint64_t> mPendingEpochFrames = 0;
    std::atomic<bool>     mFlushRequested   = false;

    std::atomic<bool>     mStarved      = false;
    std::atomic<uint64_t> mStarvedCount = 0;

    // Audio-thread-only staging: interleaved ring frames awaiting
    // de-interleave into the planar process buffer. Sized once in
    // initialize() when the block size is known — process() never allocates.
    std::vector<float> mScratch;
    // Click guard at data<->silence transitions (~1.5 ms); steady-state
    // blocks pass through unmodified.
    size_t mFadeFrames = 0;
    // True when the previous block ended in silence (start of stream counts).
    bool mWasDry = true;
};

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO
