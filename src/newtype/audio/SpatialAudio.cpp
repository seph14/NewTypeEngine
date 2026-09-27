#include "newtype/audio/SpatialAudio.h"

#if NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO

#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "cinder/audio/GenNode.h"
#include "cinder/audio/GainNode.h"
#include "cinder/audio/Context.h"

#include <algorithm>
#include <cmath>

namespace newtype::audio {

namespace {

void IPLCALL phononLogCallback( IPLLogLevel level, const char *message )
{
    switch ( level ) {
    case IPL_LOGLEVEL_ERROR:
        CI_LOG_E( "phonon: " << message );
        break;
    case IPL_LOGLEVEL_WARNING:
        CI_LOG_W( "phonon: " << message );
        break;
    case IPL_LOGLEVEL_DEBUG:
        break; // phonon is chatty at debug level; drop it
    default:
        CI_LOG_I( "phonon: " << message );
        break;
    }
}

} // namespace

// ============================================================================
// Construction / destruction
// ============================================================================

SpatialAudioSystem &SpatialAudioSystem::get()
{
    static SpatialAudioSystem instance;
    return instance;
}

SpatialAudioSystem::SpatialAudioSystem()
{
    mState = std::make_shared<SpatialAudioState>();

    // Headless / no-device machines degrade to silent operation (matches
    // SoundController); spatial voices then route nowhere.
    try {
        auto ctx    = cinder::audio::Context::master();
        auto output = ctx->getOutput();
        ctx->enable();

        IPLContextSettings contextSettings{};
        contextSettings.version     = STEAMAUDIO_VERSION;
        contextSettings.logCallback = &phononLogCallback;
        contextSettings.simdLevel   = IPL_SIMDLEVEL_AVX2;

        IPLerror err = iplContextCreate( &contextSettings, &mPhonon );
        if ( err != IPL_STATUS_SUCCESS || !mPhonon ) {
            CI_LOG_E( "SpatialAudioSystem: iplContextCreate failed ("
                      << static_cast<int>( err ) << ") - spatial audio disabled" );
            return;
        }

        mAudioSampleRate     = static_cast<int>( ctx->getSampleRate() );
        mAudioFramesPerBlock = static_cast<int>( ctx->getFramesPerBlock() );

        IPLAudioSettings audioSettings;
        audioSettings.samplingRate = mAudioSampleRate;
        audioSettings.frameSize    = mAudioFramesPerBlock;

        IPLHRTFSettings hrtfSettings{};
        hrtfSettings.type     = IPL_HRTFTYPE_DEFAULT;
        hrtfSettings.volume   = 1.f;
        hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;

        err = iplHRTFCreate( mPhonon, &audioSettings, &hrtfSettings, &mHrtf );
        if ( err != IPL_STATUS_SUCCESS || !mHrtf ) {
            CI_LOG_E( "SpatialAudioSystem: iplHRTFCreate failed ("
                      << static_cast<int>( err ) << ", sampleRate="
                      << mAudioSampleRate << ", frameSize="
                      << mAudioFramesPerBlock << ") - spatial audio disabled" );
            iplContextRelease( &mPhonon );
            mPhonon = nullptr;
            return;
        }

        mBinaural = ctx->makeNode<AmbisonicsBinauralNode>( mPhonon, mState, mHrtf, kMaxOrder );
        mBinaural >> output;
        mAvailable = true;

        CI_LOG_I( "SpatialAudioSystem: order-" << kMaxOrder
                  << " ambisonics + binaural tail online ("
                  << mAudioSampleRate << " Hz, " << mAudioFramesPerBlock
                  << " frames/block)" );
    }
    catch ( const std::exception &e ) {
        CI_LOG_W( "SpatialAudioSystem: no audio output available - running silent ("
                 << e.what() << ")" );
    }
}

SpatialAudioSystem::~SpatialAudioSystem()
{
    if ( mTestOsc ) {
        mTestOsc->disable();
        mTestOsc->disconnectAll();
    }
    if ( mTestGain )
        mTestGain->disconnectAll();
    if ( mTestEncoder )
        mTestEncoder->disconnectAll();
    mTestOsc.reset();
    mTestGain.reset();
    mTestEncoder.reset();

    if ( mBinaural ) {
        mBinaural->disconnectAll();
        mBinaural.reset();
    }
    if ( mHrtf )
        iplHRTFRelease( &mHrtf );
    if ( mPhonon )
        iplContextRelease( &mPhonon );
}

// ============================================================================
// Listener / ordering
// ============================================================================

void SpatialAudioSystem::setListener( const cinder::vec3 &pos, const cinder::vec3 &right,
                                      const cinder::vec3 &up, const cinder::vec3 &front )
{
    auto relaxed = std::memory_order_relaxed;
    mState->posX.store( pos.x, relaxed );
    mState->posY.store( pos.y, relaxed );
    mState->posZ.store( pos.z, relaxed );
    mState->rightX.store( right.x, relaxed );
    mState->rightY.store( right.y, relaxed );
    mState->rightZ.store( right.z, relaxed );
    mState->upX.store( up.x, relaxed );
    mState->upY.store( up.y, relaxed );
    mState->upZ.store( up.z, relaxed );
    mState->aheadX.store( front.x, relaxed );
    mState->aheadY.store( front.y, relaxed );
    mState->aheadZ.store( front.z, relaxed );
}

cinder::vec3 SpatialAudioSystem::listenerPosition() const
{
    auto relaxed = std::memory_order_relaxed;
    return { mState->posX.load( relaxed ), mState->posY.load( relaxed ),
             mState->posZ.load( relaxed ) };
}

cinder::vec3 SpatialAudioSystem::listenerAhead() const
{
    auto relaxed = std::memory_order_relaxed;
    return { mState->aheadX.load( relaxed ), mState->aheadY.load( relaxed ),
             mState->aheadZ.load( relaxed ) };
}

void SpatialAudioSystem::setOrder( int order )
{
    mState->order.store( std::clamp( order, 0, kMaxOrder ), std::memory_order_relaxed );
}

int SpatialAudioSystem::getOrder() const
{
    return mState->loadOrder( kMaxOrder );
}

// ============================================================================
// Voices
// ============================================================================

AmbisonicsEncodeNodeRef SpatialAudioSystem::addSource( const cinder::audio::NodeRef &source,
                                                       const cinder::vec3 &worldPos )
{
    if ( !mAvailable || !mBinaural || !source )
        return nullptr;

    auto ctx     = cinder::audio::Context::master();
    auto encoder = ctx->makeNode<AmbisonicsEncodeNode>( mPhonon, mState, kMaxOrder );
    encoder->setPosition( worldPos );
    source >> encoder >> mBinaural;
    return encoder;
}

float SpatialAudioSystem::distanceGain( const cinder::vec3 &worldPos ) const
{
    const float d = glm::length( worldPos - listenerPosition() );
    return std::min( 1.f, mRefDistance / std::max( d, 1e-4f ) );
}

void SpatialAudioSystem::setReferenceDistance( float d )
{
    mRefDistance = std::max( d, 0.01f );
}

// ============================================================================
// Test tone
// ============================================================================

namespace {
// World position for (azimuth, elevation, distance) around the listener.
// Azimuth 0 = ahead, +90 = the listener's right; elevation +90 = straight up.
cinder::vec3 polarToWorld( const SpatialAudioStateRef &state, float azDeg, float elDeg,
                       float distance )
{
    const float az = glm::radians( azDeg );
    const float el = glm::radians( elDeg );
    auto          relaxed = std::memory_order_relaxed;
    const cinder::vec3 right{ state->rightX.load( relaxed ), state->rightY.load( relaxed ),
                          state->rightZ.load( relaxed ) };
    const cinder::vec3 up{ state->upX.load( relaxed ), state->upY.load( relaxed ),
                       state->upZ.load( relaxed ) };
    const cinder::vec3 ahead{ state->aheadX.load( relaxed ), state->aheadY.load( relaxed ),
                          state->aheadZ.load( relaxed ) };
    const cinder::vec3 pos{ state->posX.load( relaxed ), state->posY.load( relaxed ),
                        state->posZ.load( relaxed ) };

    const cinder::vec3 dir = std::cos( el ) * ( std::cos( az ) * ahead + std::sin( az ) * right )
                       + std::sin( el ) * up;
    return pos + distance * dir;
}
} // namespace

void SpatialAudioSystem::setTestToneEnabled( bool enabled )
{
    if ( enabled == mTestToneOn )
        return;
    if ( !enabled ) {
        if ( mTestOsc )
            mTestOsc->disable();
        mTestToneOn = false;
        return;
    }

    if ( !mAvailable )
        return;

    if ( !mTestEncoder ) {
        auto     ctx    = cinder::audio::Context::master();
        mTestOsc        = ctx->makeNode<cinder::audio::GenOscNode>( 440.f, cinder::audio::Node::Format().channels( 1 ) );
        mTestGain       = ctx->makeNode<cinder::audio::GainNode>( 1.f, cinder::audio::Node::Format().channels( 1 ) );
        mTestEncoder    = addSource( mTestGain, polarToWorld( mState, mTestAzimuthDeg,
                                                              mTestElevationDeg, mTestDistance ) );
        mTestOsc >> mTestGain;
    }

    const cinder::vec3 pos = polarToWorld( mState, mTestAzimuthDeg, mTestElevationDeg, mTestDistance );
    if ( mTestEncoder )
        mTestEncoder->setPosition( pos );
    if ( mTestGain )
        mTestGain->setValue( distanceGain( pos ) );
    if ( mTestOsc )
        mTestOsc->enable();
    mTestToneOn = true;
}

void SpatialAudioSystem::setTestTonePolar( float azimuthDeg, float elevationDeg, float distance )
{
    mTestAzimuthDeg   = azimuthDeg;
    mTestElevationDeg = elevationDeg;
    mTestDistance     = std::max( distance, 0.1f );

    if ( !mTestToneOn || !mTestEncoder || !mTestGain )
        return;
    const cinder::vec3 pos = polarToWorld( mState, mTestAzimuthDeg, mTestElevationDeg, mTestDistance );
    mTestEncoder->setPosition( pos );
    mTestGain->setValue( distanceGain( pos ) );
}

void SpatialAudioSystem::setTestToneOrbit( bool enabled, float angularSpeed )
{
    mTestOrbit  = enabled;
    mOrbitSpeed = angularSpeed;
}

void SpatialAudioSystem::update( double dt )
{
    if ( !mTestToneOn || !mTestOrbit )
        return;
    mOrbitAngle = std::fmod( mOrbitAngle + static_cast<float>( dt ) * mOrbitSpeed,
                             glm::two_pi<float>() );
    setTestTonePolar( glm::degrees( mOrbitAngle ), mTestElevationDeg, mTestDistance );
}

// ============================================================================
// UI
// ============================================================================

void SpatialAudioSystem::drawUi()
{
    if ( !ImGui::CollapsingHeader( "Spatial Audio" ) )
        return;
    ImGui::ScopedId scope( "spatial" );

    if ( !mAvailable ) {
        ImGui::TextDisabled( "unavailable (no audio device)" );
        return;
    }

    const cinder::vec3 pos = listenerPosition();
    ImGui::Text( "listener pos: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z );

    int order = getOrder();
    if ( ImGui::SliderInt( "order", &order, 0, kMaxOrder ) )
        setOrder( order );

    float refDist = mRefDistance;
    if ( ImGui::SliderFloat( "ref distance", &refDist, 0.1f, 10.f, "%.2f m" ) )
        setReferenceDistance( refDist );

    ImGui::Separator();
    if ( !ImGui::CollapsingHeader( "Test Tone", ImGuiTreeNodeFlags_DefaultOpen ) )
        return;
    ImGui::ScopedId toneScope( "tone" );

    bool tone = mTestToneOn;
    if ( ImGui::Checkbox( "enabled", &tone ) )
        setTestToneEnabled( tone );

    float az = mTestAzimuthDeg, el = mTestElevationDeg, dist = mTestDistance;
    if ( ImGui::SliderFloat( "azimuth", &az, -180.f, 180.f, "%.0f deg" ) )
        setTestTonePolar( az, el, dist );
    if ( ImGui::SliderFloat( "elevation", &el, -89.f, 89.f, "%.0f deg" ) )
        setTestTonePolar( az, el, dist );
    if ( ImGui::SliderFloat( "distance", &dist, 0.25f, 50.f, "%.2f m" ) )
        setTestTonePolar( az, el, dist );

    bool orbit = mTestOrbit;
    if ( ImGui::Checkbox( "orbit", &orbit ) )
        setTestToneOrbit( orbit );
    if ( mTestOrbit ) {
        float speed = mOrbitSpeed;
        if ( ImGui::SliderFloat( "orbit speed", &speed, -3.f, 3.f, "%.2f rad/s" ) )
            setTestToneOrbit( true, speed );
    }
}

} // namespace newtype::audio

#endif // NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO
