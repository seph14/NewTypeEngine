#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO

#include "cinder/Vector.h"
#include "cinder/audio/Node.h"
#include "cinder/audio/Buffer.h"
#include "phonon.h"

#include <atomic>
#include <memory>
#include <vector>

namespace newtype::audio {

//
// Shared spatial-audio state, published render thread -> audio thread.
//
// The listener pose (world space) and the runtime ambisonic order are read
// by the audio thread inside the nodes' sumInputs(). Individual fields are
// relaxed atomics rather than a seqlock: these are smooth perceptual
// parameters, and a torn read straddling two adjacent frames of a moving
// listener is inaudible — no buffer content depends on their consistency.
// Only SpatialAudioSystem::setListener / setOrder ever store.
//
// Frame conventions (pinned here, validated by the orbit test tone):
//   - world positions and the listener basis are engine world space;
//   - the encode direction is the world-space vector listener -> source
//     (IPLAmbisonicsEncodeEffectParams::direction normalizes it itself);
//   - the bus is world-space B-format; AmbisonicsBinauralNode rotates it
//     into the listener frame with IPLAmbisonicsRotationEffect before the
//     HRTF decode, so both effects always share one (Steam Audio internal)
//     ambisonic ordering/normalization.
//
struct SpatialAudioState {
    // Listener position (world).
    std::atomic<float> posX{ 0.f }, posY{ 0.f }, posZ{ 0.f };
    // Listener basis (world, orthonormal; setListener publishes).
    std::atomic<float> rightX{ 1.f }, rightY{ 0.f }, rightZ{ 0.f };
    std::atomic<float> upX{ 0.f },    upY{ 1.f },   upZ{ 0.f };
    std::atomic<float> aheadX{ 0.f }, aheadY{ 0.f }, aheadZ{ -1.f };
    // Runtime ambisonic order, clamped to [0, maxOrder] on read.
    std::atomic<int>   order{ 3 };

    [[nodiscard]] int loadOrder( int maxOrder ) const {
        int o = order.load( std::memory_order_relaxed );
        return o < 0 ? 0 : ( o > maxOrder ? maxOrder : o );
    }
};

using SpatialAudioStateRef = std::shared_ptr<SpatialAudioState>;

//
// Mono in -> (order+1)^2-channel B-format out, one per spatial voice.
//
// Wraps IPLAmbisonicsEncodeEffect. Channel-layout conversion follows the
// ChannelRouterNode pattern (sumInputs override — the stock Node graph
// cannot express an N!=M channel conversion): each input is pulled into the
// shared summing buffer and folded to mono (a stereo input is L/R-averaged;
// spatial assets should be authored mono), then the IPL effect encodes that
// mono capture into this node's internal B-format buffer, which downstream
// (the binaural bus node) mixes from.
//
// Threading: IPL objects are created in initialize() and released in
// uninitialize() (both main thread, per the VideoAudioNode discipline);
// sumInputs() only applies the effect — no allocation, no locks.
// setPosition() is render-thread, relaxed-atomic.
//
class AmbisonicsEncodeNode : public cinder::audio::Node {
public:
    /// \a format's channel count is overridden to (maxOrder+1)^2.
    AmbisonicsEncodeNode( IPLContext phononContext, SpatialAudioStateRef state,
                          int maxOrder, const Format &format = Format() );
    ~AmbisonicsEncodeNode() override;

    /// World-space source position (render thread).
    void                setPosition( const cinder::vec3 &worldPos );
    [[nodiscard]] cinder::vec3 getPosition() const;
    [[nodiscard]] int   maxOrder() const { return mMaxOrder; }

protected:
    bool supportsInputNumChannels( size_t numChannels ) const override { return numChannels >= 1; }
    bool supportsProcessInPlace() const override { return false; }
    void initialize() override;
    void uninitialize() override;
    void sumInputs() override;

private:
    IPLContext                 mPhonon;      // retained for the effect's lifetime
    SpatialAudioStateRef       mState;
    int                        mMaxOrder;
    IPLAmbisonicsEncodeEffect  mEffect = nullptr;

    cinder::audio::BufferDynamic   mMonoCapture; // frames x 1, sized in initialize()
    std::vector<float *>       mOutChannels; // IPL out-view pointer table
    std::atomic<float>         mSrcX{ 0.f }, mSrcY{ 0.f }, mSrcZ{ 0.f };
};

using AmbisonicsEncodeNodeRef = std::shared_ptr<AmbisonicsEncodeNode>;

//
// (order+1)^2-channel B-format bus in -> stereo binaural out. One per graph.
//
// Bus tail: sums every connected encoder into a staging bus, rotates the
// world-space bus into the listener frame (IPLAmbisonicsRotationEffect,
// listener basis from SpatialAudioState), then renders HRTF binaural to
// stereo (IPLAmbisonicsBinauralEffect with the shared built-in IPLHRTF).
// The stereo result is published into this node's internal buffer for the
// downstream output device to pull. Same ChannelRouter-style sumInputs
// override and threading rules as the encoder.
//
class AmbisonicsBinauralNode : public cinder::audio::Node {
public:
    /// \a hrtf is retained for the effect's lifetime. \a format's channel
    /// count is overridden to 2.
    AmbisonicsBinauralNode( IPLContext phononContext, SpatialAudioStateRef state,
                            IPLHRTF hrtf, int maxOrder, const Format &format = Format() );
    ~AmbisonicsBinauralNode() override;

    [[nodiscard]] int maxOrder() const { return mMaxOrder; }

protected:
    bool supportsInputNumChannels( size_t numChannels ) const override;
    bool supportsProcessInPlace() const override { return false; }
    void initialize() override;
    void uninitialize() override;
    void sumInputs() override;

private:
    IPLContext                    mPhonon;   // retained
    SpatialAudioStateRef          mState;
    IPLHRTF                       mHrtf;     // retained
    int                           mMaxOrder;
    IPLAmbisonicsRotationEffect   mRotation = nullptr;
    IPLAmbisonicsBinauralEffect   mBinaural = nullptr;

    cinder::audio::BufferDynamic      mBus;        // frames x Nch, summed encoders
    cinder::audio::BufferDynamic      mBusRotated; // frames x Nch, listener frame
    std::vector<float *>          mBusChannels, mRotChannels, mOutChannels;
};

using AmbisonicsBinauralNodeRef = std::shared_ptr<AmbisonicsBinauralNode>;

} // namespace newtype::audio

#endif // NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO
