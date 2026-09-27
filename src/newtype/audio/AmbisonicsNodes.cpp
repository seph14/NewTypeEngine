#include "newtype/audio/AmbisonicsNodes.h"

#if NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO

#include "cinder/audio/Context.h"
#include "cinder/audio/dsp/Dsp.h"
#include "cinder/Log.h"

#include <algorithm>

namespace newtype::audio {

namespace {

constexpr int numChannelsForOrder( int order ) { return ( order + 1 ) * ( order + 1 ); }

IPLAudioBuffer makeView( int numChannels, IPLint32 numSamples, float **data ) {
    IPLAudioBuffer view;
    view.numChannels = numChannels;
    view.numSamples  = numSamples;
    view.data        = data;
    return view;
}

} // namespace

// ============================================================================
// AmbisonicsEncodeNode
// ============================================================================

AmbisonicsEncodeNode::AmbisonicsEncodeNode( IPLContext phononContext,
                                            SpatialAudioStateRef state, int maxOrder,
                                            const Format &format )
    : Node( [&] {
          Format f = format;
          f.channels( static_cast<size_t>( numChannelsForOrder( maxOrder ) ) );
          f.channelMode( ChannelMode::SPECIFIED );
          return f;
      }() ),
      mPhonon( iplContextRetain( phononContext ) ),
      mState( std::move( state ) ),
      mMaxOrder( maxOrder )
{
}

AmbisonicsEncodeNode::~AmbisonicsEncodeNode()
{
    if ( mEffect )
        iplAmbisonicsEncodeEffectRelease( &mEffect );
    if ( mPhonon )
        iplContextRelease( &mPhonon );
}

void AmbisonicsEncodeNode::setPosition( const cinder::vec3 &worldPos )
{
    mSrcX.store( worldPos.x, std::memory_order_relaxed );
    mSrcY.store( worldPos.y, std::memory_order_relaxed );
    mSrcZ.store( worldPos.z, std::memory_order_relaxed );
}

cinder::vec3 AmbisonicsEncodeNode::getPosition() const
{
    return { mSrcX.load( std::memory_order_relaxed ),
             mSrcY.load( std::memory_order_relaxed ),
             mSrcZ.load( std::memory_order_relaxed ) };
}

void AmbisonicsEncodeNode::initialize()
{
    auto       ctx = getContext();
    IPLAudioSettings audioSettings;
    audioSettings.samplingRate = static_cast<IPLint32>( ctx->getSampleRate() );
    audioSettings.frameSize    = static_cast<IPLint32>( ctx->getFramesPerBlock() );

    IPLAmbisonicsEncodeEffectSettings effectSettings;
    effectSettings.maxOrder = mMaxOrder;

    IPLerror err = iplAmbisonicsEncodeEffectCreate( mPhonon, &audioSettings,
                                                    &effectSettings, &mEffect );
    if ( err != IPL_STATUS_SUCCESS ) {
        CI_LOG_E( "AmbisonicsEncodeNode: iplAmbisonicsEncodeEffectCreate failed ("
                  << static_cast<int>( err ) << ", frameSize="
                  << audioSettings.frameSize << ") - node runs silent" );
        mEffect = nullptr;
    }

    mMonoCapture.setSize( ctx->getFramesPerBlock(), 1 );
    mOutChannels.assign( static_cast<size_t>( numChannelsForOrder( mMaxOrder ) ), nullptr );
}

void AmbisonicsEncodeNode::uninitialize()
{
    if ( mEffect )
        iplAmbisonicsEncodeEffectRelease( &mEffect );
    mEffect = nullptr;
}

void AmbisonicsEncodeNode::sumInputs()
{
    cinder::audio::Buffer *        internal = getInternalBuffer();
    cinder::audio::BufferDynamic * summing  = getSummingBuffer();
    const size_t           numFrames = internal->getNumFrames();

    if ( !mEffect ) {
        internal->zero();
        return;
    }

    // Fold all inputs to mono. Multi-channel inputs are L/R-averaged (spatial
    // assets should be authored mono; SoundController sums stereo buffers at
    // load — this is the safety net).
    mMonoCapture.zero();
    float *capture = mMonoCapture.getChannel( 0 );
    for ( auto &input : getInputs() ) {
        if ( !input )
            continue;

        summing->setNumChannels( input->getNumChannels() );
        input->pullInputs( summing );
        const cinder::audio::Buffer *processed =
            input->getProcessesInPlace() ? summing : input->getInternalBuffer();

        const float *ch0 = processed->getChannel( 0 );
        if ( processed->getNumChannels() >= 2 ) {
            const float *ch1 = processed->getChannel( 1 );
            for ( size_t i = 0; i < numFrames; ++i )
                capture[ i ] += ( ch0[ i ] + ch1[ i ] ) * 0.5f;
        }
        else {
            cinder::audio::dsp::add( ch0, capture, capture, numFrames );
        }
    }

    // Encode: world-space direction listener -> source (IPL normalizes).
    const IPLVector3 dir{
        mSrcX.load( std::memory_order_relaxed ) - mState->posX.load( std::memory_order_relaxed ),
        mSrcY.load( std::memory_order_relaxed ) - mState->posY.load( std::memory_order_relaxed ),
        mSrcZ.load( std::memory_order_relaxed ) - mState->posZ.load( std::memory_order_relaxed ) };

    float *inPtr = capture;
    IPLAudioBuffer inView  = makeView( 1, static_cast<IPLint32>( numFrames ), &inPtr );
    for ( size_t c = 0; c < mOutChannels.size(); ++c )
        mOutChannels[ c ] = internal->getChannel( c );
    IPLAudioBuffer outView = makeView( static_cast<int>( mOutChannels.size() ),
                                       static_cast<IPLint32>( numFrames ),
                                       mOutChannels.data() );

    IPLAmbisonicsEncodeEffectParams params;
    params.direction = dir;
    params.order     = mState->loadOrder( mMaxOrder );

    iplAmbisonicsEncodeEffectApply( mEffect, &params, &inView, &outView );
}

// ============================================================================
// AmbisonicsBinauralNode
// ============================================================================

AmbisonicsBinauralNode::AmbisonicsBinauralNode( IPLContext phononContext,
                                                SpatialAudioStateRef state, IPLHRTF hrtf,
                                                int maxOrder, const Format &format )
    : Node( [&] {
          Format f = format;
          f.channels( 2 );
          f.channelMode( ChannelMode::SPECIFIED );
          return f;
      }() ),
      mPhonon( iplContextRetain( phononContext ) ),
      mState( std::move( state ) ),
      mHrtf( iplHRTFRetain( hrtf ) ),
      mMaxOrder( maxOrder )
{
}

AmbisonicsBinauralNode::~AmbisonicsBinauralNode()
{
    if ( mBinaural )
        iplAmbisonicsBinauralEffectRelease( &mBinaural );
    if ( mRotation )
        iplAmbisonicsRotationEffectRelease( &mRotation );
    if ( mHrtf )
        iplHRTFRelease( &mHrtf );
    if ( mPhonon )
        iplContextRelease( &mPhonon );
}

bool AmbisonicsBinauralNode::supportsInputNumChannels( size_t numChannels ) const
{
    return numChannels == static_cast<size_t>( numChannelsForOrder( mMaxOrder ) );
}

void AmbisonicsBinauralNode::initialize()
{
    auto       ctx = getContext();
    IPLAudioSettings audioSettings;
    audioSettings.samplingRate = static_cast<IPLint32>( ctx->getSampleRate() );
    audioSettings.frameSize    = static_cast<IPLint32>( ctx->getFramesPerBlock() );

    IPLAmbisonicsRotationEffectSettings rotationSettings;
    rotationSettings.maxOrder = mMaxOrder;

    IPLAmbisonicsBinauralEffectSettings binauralSettings;
    binauralSettings.hrtf     = mHrtf;
    binauralSettings.maxOrder = mMaxOrder;

    IPLerror err = iplAmbisonicsRotationEffectCreate( mPhonon, &audioSettings,
                                                      &rotationSettings, &mRotation );
    if ( err != IPL_STATUS_SUCCESS ) {
        CI_LOG_E( "AmbisonicsBinauralNode: rotation effect create failed ("
                  << static_cast<int>( err ) << ")" );
        mRotation = nullptr;
    }

    err = iplAmbisonicsBinauralEffectCreate( mPhonon, &audioSettings,
                                             &binauralSettings, &mBinaural );
    if ( err != IPL_STATUS_SUCCESS ) {
        CI_LOG_E( "AmbisonicsBinauralNode: binaural effect create failed ("
                  << static_cast<int>( err ) << ", frameSize="
                  << audioSettings.frameSize << ") - node runs silent" );
        mBinaural = nullptr;
    }

    const size_t numChannels = static_cast<size_t>( numChannelsForOrder( mMaxOrder ) );
    mBus.setSize( ctx->getFramesPerBlock(), numChannels );
    mBusRotated.setSize( ctx->getFramesPerBlock(), numChannels );
    mBusChannels.assign( numChannels, nullptr );
    mRotChannels.assign( numChannels, nullptr );
    mOutChannels.assign( 2, nullptr );
}

void AmbisonicsBinauralNode::uninitialize()
{
    if ( mBinaural )
        iplAmbisonicsBinauralEffectRelease( &mBinaural );
    mBinaural = nullptr;
    if ( mRotation )
        iplAmbisonicsRotationEffectRelease( &mRotation );
    mRotation = nullptr;
}

void AmbisonicsBinauralNode::sumInputs()
{
    cinder::audio::Buffer *        internal  = getInternalBuffer();
    cinder::audio::BufferDynamic * summing   = getSummingBuffer();
    const size_t           numFrames = mBus.getNumFrames();
    const int              numCh     = static_cast<int>( mBusChannels.size() );

    if ( !mRotation || !mBinaural || numFrames == 0 ) {
        internal->zero();
        return;
    }

    // Sum every encoder's world-space B-format into the staging bus.
    mBus.zero();
    for ( auto &input : getInputs() ) {
        if ( !input )
            continue;

        summing->setNumChannels( input->getNumChannels() );
        input->pullInputs( summing );
        const cinder::audio::Buffer *processed =
            input->getProcessesInPlace() ? summing : input->getInternalBuffer();

        if ( processed->getNumChannels() != mBusChannels.size() )
            continue;

        for ( size_t c = 0; c < mBusChannels.size(); ++c )
            cinder::audio::dsp::add( processed->getChannel( c ), mBus.getChannel( c ),
                             mBus.getChannel( c ), numFrames );
    }

    const int order = mState->loadOrder( mMaxOrder );

    // Rotate the bus into the listener frame.
    IPLCoordinateSpace3 orientation;
    orientation.right = IPLVector3{ mState->rightX.load( std::memory_order_relaxed ),
                                    mState->rightY.load( std::memory_order_relaxed ),
                                    mState->rightZ.load( std::memory_order_relaxed ) };
    orientation.up = IPLVector3{ mState->upX.load( std::memory_order_relaxed ),
                                 mState->upY.load( std::memory_order_relaxed ),
                                 mState->upZ.load( std::memory_order_relaxed ) };
    orientation.ahead = IPLVector3{ mState->aheadX.load( std::memory_order_relaxed ),
                                    mState->aheadY.load( std::memory_order_relaxed ),
                                    mState->aheadZ.load( std::memory_order_relaxed ) };
    orientation.origin = IPLVector3{ 0.f, 0.f, 0.f };

    for ( int c = 0; c < numCh; ++c ) {
        mBusChannels[ c ] = mBus.getChannel( c );
        mRotChannels[ c ] = mBusRotated.getChannel( c );
    }
    mOutChannels[ 0 ] = internal->getChannel( 0 );
    mOutChannels[ 1 ] = internal->getChannel( 1 );

    IPLAudioBuffer busView  = makeView( numCh, static_cast<IPLint32>( numFrames ),
                                        mBusChannels.data() );
    IPLAudioBuffer rotView  = makeView( numCh, static_cast<IPLint32>( numFrames ),
                                        mRotChannels.data() );
    IPLAudioBuffer outView  = makeView( 2, static_cast<IPLint32>( numFrames ),
                                        mOutChannels.data() );

    IPLAmbisonicsRotationEffectParams rotationParams;
    rotationParams.orientation = orientation;
    rotationParams.order       = order;
    iplAmbisonicsRotationEffectApply( mRotation, &rotationParams, &busView, &rotView );

    IPLAmbisonicsBinauralEffectParams binauralParams;
    binauralParams.hrtf  = mHrtf;
    binauralParams.order = order;
    iplAmbisonicsBinauralEffectApply( mBinaural, &binauralParams, &rotView, &outView );
}

} // namespace newtype::audio

#endif // NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO
