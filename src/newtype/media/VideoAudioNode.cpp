#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO

#include "newtype/media/VideoAudioNode.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace newtype::media {

VideoAudioNode::VideoAudioNode(AudioRingBuffer& ring, uint32_t sampleRate, uint32_t channels,
                               cinder::audio::Node::Format const& format)
    : InputNode(format)
    , mRing(ring)
    , mSampleRate(sampleRate)
    , mChannels(channels)
{
    setChannelMode(cinder::audio::Node::ChannelMode::SPECIFIED);
    setNumChannels(mChannels);
}

void VideoAudioNode::initialize() {
    // Audio-thread staging sized here, while the block size is known and we
    // are on the graph thread — process() must never allocate.
    mScratch.assign(getFramesPerBlock() * getNumChannels(), 0.0f);
    // ~1.5 ms, clamped to a sane minimum and to one block.
    mFadeFrames = std::min<size_t>(getFramesPerBlock(),
                                   std::max<size_t>(8, mSampleRate * 3u / 2000u));
    mWasDry = true;
}

void VideoAudioNode::rebase_to(double sec) {
    if (sec < 0.0) sec = 0.0;
    uint64_t const frames = static_cast<uint64_t>(sec * static_cast<double>(mSampleRate) + 0.5);
    mPendingEpochFrames.store(frames, std::memory_order_relaxed);
    mFlushRequested.store(true, std::memory_order_seq_cst);
}

void VideoAudioNode::request_flush() {
    // Epoch unchanged: the position keeps whatever the speaker already heard;
    // only the unconsumed backlog between it and the producer is dropped.
    mPendingEpochFrames.store(mEpochFrames.load(std::memory_order_relaxed) +
                                  mFramesSinceEpoch.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    mFlushRequested.store(true, std::memory_order_seq_cst);
}

void VideoAudioNode::prime_epoch(double sec) {
    // Disabled-only path (asserted by the caller's protocol): no audio thread
    // is pulling process(), so plain stores are race-free.
    if (sec < 0.0) sec = 0.0;
    mEpochFrames.store(static_cast<uint64_t>(sec * static_cast<double>(mSampleRate) + 0.5),
                       std::memory_order_relaxed);
    mFramesSinceEpoch.store(0, std::memory_order_relaxed);
    mPendingEpochFrames.store(0, std::memory_order_relaxed);
    // Drop any flush queued before the first enable — it would drain the
    // pre-roll the gate is about to start playing.
    mFlushRequested.store(false, std::memory_order_relaxed);
}

bool VideoAudioNode::wait_flushed(double timeoutSec) const {
    double waited = 0.0;
    double const step = 0.001; // flush lands within one audio block (~ms)
    while (mFlushRequested.load(std::memory_order_seq_cst)) {
        if (waited >= timeoutSec) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited += step;
    }
    return true;
}

double VideoAudioNode::position_sec() const {
    uint64_t const frames = mEpochFrames.load(std::memory_order_acquire) +
                            mFramesSinceEpoch.load(std::memory_order_acquire);
    return static_cast<double>(frames) / static_cast<double>(mSampleRate);
}

void VideoAudioNode::process(cinder::audio::Buffer* buffer) {
    // Flush handshake first: render thread published (epoch, flag) — drain the
    // ring and latch the new epoch before pulling anything.
    if (mFlushRequested.exchange(false, std::memory_order_seq_cst)) {
        mEpochFrames.store(mPendingEpochFrames.load(std::memory_order_relaxed),
                           std::memory_order_release);
        mFramesSinceEpoch.store(0, std::memory_order_relaxed);
        mRing.reset();
    }

    auto const& frameRange = getProcessFramesRange();
    size_t const numFrames = frameRange.second - frameRange.first;
    uint32_t const channels = getNumChannels();

    // Stage the ring's interleaved frames, then de-interleave into the planar
    // process buffer (per-channel planes — see BufferT::getChannel).
    size_t const filled = mRing.read(mScratch.data(), numFrames);
    bool const dry = filled < numFrames;

    // Click guards at data<->silence boundaries only; interior blocks of
    // continuous data pass through with unity gain.
    size_t const fadeIn  = (mWasDry && filled > 0) ? std::min(mFadeFrames, filled) : 0;
    size_t const fadeOut = (dry && filled > 0)     ? std::min(mFadeFrames, filled) : 0;

    for (uint32_t ch = 0; ch < channels; ++ch) {
        float* plane = buffer->getChannel(ch) + frameRange.first;
        for (size_t i = 0; i < filled; ++i) {
            float g = 1.0f;
            if (i < fadeIn)
                g *= static_cast<float>(i + 1) / static_cast<float>(fadeIn + 1);
            if (fadeOut > 0 && i >= filled - fadeOut)
                g *= static_cast<float>(filled - i) / static_cast<float>(fadeOut + 1);
            plane[i] = mScratch[i * channels + ch] * g;
        }
        if (dry) {
            // Underrun: emit silence but do NOT advance the clock over frames
            // that were never played — video freezes with audio until the
            // producer catches up (audio-master sync behavior).
            std::memset(plane + filled, 0, (numFrames - filled) * sizeof(float));
        }
    }

    if (dry) {
        mStarved.store(true, std::memory_order_relaxed);
        mStarvedCount.fetch_add(1, std::memory_order_relaxed);
    } else {
        mStarved.store(false, std::memory_order_relaxed);
    }
    mWasDry = dry;

    mFramesSinceEpoch.fetch_add(filled, std::memory_order_acq_rel);
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO
