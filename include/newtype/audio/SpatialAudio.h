#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO

#include "newtype/audio/AmbisonicsNodes.h"

#include "cinder/Vector.h"
#include "cinder/audio/GainNode.h"
#include "cinder/audio/GenNode.h"

#include <memory>

namespace newtype::audio {

//
// HOA volumetric audio front-end (Steam Audio phonon, Apache-2.0).
//
// Owns the one-per-graph spatial tail:
//
//   [per-voice] player >> gain >> AmbisonicsEncodeNode --\\
//   [per-voice] ...                                      >> AmbisonicsBinauralNode >> output
//
// Encoders write a world-space 3rd-order B-format bus ((order+1)^2 = 16 ch);
// the binaural node rotates it into the listener frame and renders HRTF
// binaural to the stereo device output. Non-spatial audio keeps routing
// straight to the context output and sums with the spatial tail.
//
// Ambisonic encoding carries no distance attenuation — callers apply
// distanceGain() (inverse with reference-distance clamp) on the voice gain.
//
// Singleton mirroring SoundController: silent-degrades on machines without
// an audio device (isAvailable() == false; addSource returns null). All IPL
// object creation/destruction happens on the main thread; the audio thread
// only runs effect applies inside the nodes.
//
class SpatialAudioSystem {
public:
    [[nodiscard]] static SpatialAudioSystem &get();
    ~SpatialAudioSystem();

    SpatialAudioSystem( const SpatialAudioSystem & )            = delete;
    SpatialAudioSystem &operator=( const SpatialAudioSystem & ) = delete;

    [[nodiscard]] bool isAvailable() const { return mAvailable; }
    [[nodiscard]] int  maxOrder() const { return kMaxOrder; }

    /// Runtime ambisonic order scaling, [0, maxOrder]. Lower orders reduce
    /// CPU on both the encoders and the binaural tail.
    void setOrder( int order );
    [[nodiscard]] int getOrder() const;

    /// Listener pose push (render thread, once per frame after the camera
    /// updates). Vectors are world-space; `front` is the view direction.
    void setListener( const cinder::vec3 &pos, const cinder::vec3 &right,
                      const cinder::vec3 &up, const cinder::vec3 &front );
    [[nodiscard]] cinder::vec3 listenerPosition() const;
    [[nodiscard]] cinder::vec3 listenerAhead() const;

    /// Wire `source` (mono-capable, e.g. a GainNode) into the spatial bus at
    /// `worldPos`. Returns the encoder for later setPosition() calls, or a
    /// null ref when spatial audio is unavailable.
    AmbisonicsEncodeNodeRef addSource( const cinder::audio::NodeRef &source,
                                       const cinder::vec3 &worldPos );

    /// min(1, refDist/dist) — the distance attenuation for a world-space
    /// source (encoding itself carries none).
    [[nodiscard]] float distanceGain( const cinder::vec3 &worldPos ) const;
    void                setReferenceDistance( float d );
    [[nodiscard]] float referenceDistance() const { return mRefDistance; }

    // --- Test tone (spike / debug UI) ---
    /// 440 Hz sine placed in the world; doubles as the L/R-conventions check.
    void setTestToneEnabled( bool enabled );
    [[nodiscard]] bool isTestToneEnabled() const { return mTestToneOn; }
    void setTestTonePolar( float azimuthDeg, float elevationDeg, float distance );
    /// Orbit the test tone around the listener at angularSpeed rad/s.
    void setTestToneOrbit( bool enabled, float angularSpeed = 1.f );
    /// Per-frame (orbit animation). Call from the main update loop.
    void update( double dt );

    /// Spatial debug section (listener pose, order, test tone). Call inside
    /// an active ImGui frame.
    void drawUi();

private:
    SpatialAudioSystem();

    static constexpr int kMaxOrder = 3;

    bool                    mAvailable = false;
    SpatialAudioStateRef    mState;
    IPLContext              mPhonon = nullptr;
    IPLHRTF                 mHrtf   = nullptr;
    int                     mAudioSampleRate = 0;
    int                     mAudioFramesPerBlock = 0;

    AmbisonicsBinauralNodeRef mBinaural;

    // Test tone chain (lazily created on first enable).
    cinder::audio::GenOscNodeRef         mTestOsc;
    cinder::audio::GainNodeRef          mTestGain;
    AmbisonicsEncodeNodeRef         mTestEncoder;
    bool                            mTestToneOn   = false;
    bool                            mTestOrbit    = false;
    float                           mOrbitAngle   = 0.f;
    float                           mOrbitSpeed   = 1.f;
    float                           mTestAzimuthDeg = 0.f, mTestElevationDeg = 0.f;
    float                           mTestDistance   = 2.f;
    float                           mRefDistance    = 1.f;
};

} // namespace newtype::audio

#else
// NT_ENABLE_AUDIO=0 / NT_ENABLE_SPATIAL_AUDIO=0 stub — no-op like
// SoundController's disabled form.
#include "cinder/Vector.h"
#include "cinder/audio/GainNode.h"
#include "cinder/audio/GenNode.h"

namespace newtype::audio {
class SpatialAudioSystem {
    SpatialAudioSystem() = default;
public:
    static SpatialAudioSystem &get() {
        static SpatialAudioSystem instance;
        return instance;
    }
    [[nodiscard]] bool isAvailable() const { return false; }
    [[nodiscard]] int  maxOrder() const { return 0; }
    [[nodiscard]] int  getOrder() const { return 0; }
    template<typename... A> void                setOrder( A &&... ) {}
    template<typename... A> void                setListener( A &&... ) {}
    template<typename... A> cinder::vec3            listenerPosition() const { return {}; }
    [[nodiscard]] cinder::vec3 listenerAhead() const { return { 0.f, 0.f, -1.f }; }
    template<typename... A> AmbisonicsEncodeNodeRef addSource( A &&... ) { return nullptr; }
    template<typename... A> float               distanceGain( A &&... ) const { return 0.f; }
    template<typename... A> void                setReferenceDistance( A &&... ) {}
    template<typename... A> float               referenceDistance() const { return 1.f; }
    template<typename... A> void                setTestToneEnabled( A &&... ) {}
    template<typename... A> bool                isTestToneEnabled() const { return false; }
    template<typename... A> void                setTestTonePolar( A &&... ) {}
    template<typename... A> void                setTestToneOrbit( A &&... ) {}
    template<typename... A> void                update( A &&... ) {}
    template<typename... A> void                drawUi() {}
};
} // namespace newtype::audio

#endif // NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO
