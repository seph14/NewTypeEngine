#include "newtype/plugin/MovieWriter.h"
#include <atlbase.h>

namespace cinder {
	namespace wmf {
		MovieWriter::Format::Format()
			: mCodec( H264 ), mFileType( QUICK_TIME_MOVIE ), mFPS( 30 ), mVideoInputFormat( MFVideoFormat_RGB32 ), mVideoBitRate( 5120000 ), mHardwareAcceleration( false ) {
			mDefaultFrameDuration = 10 * 1000 * 1000 / mFPS;
		}

		namespace {
			const GUID  codecToMFVideoCodecKey( MovieWriter::Codec codec ) {
				switch( codec ) {
					case MovieWriter::Codec::H264:
						return MFVideoFormat_H264;
					case MovieWriter::Codec::WMV3:
						return MFVideoFormat_WMV3;
					case MovieWriter::Codec::H265:
						return MFVideoFormat_HEVC;
					default:
						return MFVideoFormat_WMV3;
				}
			}
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// MovieWriter
		MovieWriter::MovieWriter( const fs::path& path, UINT32 width, UINT32 height, const Format& format )
			: mWidth( width ), mHeight( height ), mFormat( format ), mFinished( false ), mNumFrames( 0 ), mCurrentSeconds( 0 ) {
			/*mFbo = gl::Fbo::create( width, height, true, false, false );

			ci::geom::Rect rect( mFbo->getBounds() );
			
			mGlsl = ci::gl::GlslProg::create( ci::gl::GlslProg::Format()
			                                  .vertex( CI_GLSL( 330,
			                                          uniform mat4 ciModelViewProjection;
			                                          in vec4 ciPosition;
			                                          in vec2 ciTexCoord0;
			                                          out highp vec2 TexCoord0;

			void main( void ) {
				gl_Position = ciModelViewProjection *  ciPosition;
				TexCoord0 = vec2(0.,1.) + vec2(1.,-1.) * ciTexCoord0;
			} ) )
			.fragment( CI_GLSL( 330,
			                    uniform sampler2D uTexture;
			                    uniform bool uFlipY = false;

			                    in vec2 TexCoord0;

			void main( void ) {
				vec2 st = TexCoord0;

				//if( uFlipY ) {
				//	st.y = 1 - st.y;
				//}

				//vec4 colorTex = texture( uTexture, st );
				gl_FragColor = texture(uTexture, st).bgra;
			} ) ) );

			mBatch = ci::gl::Batch::create( rect, mGlsl );
			*/

			HRESULT hr = MFStartup( MF_VERSION );
			mSinkWriter = NULL;
			mStreamIndex = NULL;

			if( SUCCEEDED( hr ) ) {
				IMFSinkWriter*   pSinkWriter = NULL;
				IMFMediaType*    pVideoTypeOut = NULL;
				IMFMediaType*    pVideoTypeIn = NULL;

				DWORD           streamIndex;
				LPCWSTR			file = path.c_str();

				// make sure directories exist
				std::string pathString = path.string();
				std::filesystem::create_directories( ci::fs::path( pathString ).remove_filename() );

				HRESULT hr;
				if (mFormat.getHardwareAccelerationEnabled()) {
					CComPtr<IMFAttributes> attrs;
					MFCreateAttributes(&attrs, 1);
					attrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
					hr = MFCreateSinkWriterFromURL(file, NULL, attrs, &pSinkWriter);
				} else {
					hr = MFCreateSinkWriterFromURL(file, NULL, NULL, &pSinkWriter);
				}

				// Set the output media type.
				if( SUCCEEDED( hr ) ) {
					hr = MFCreateMediaType( &pVideoTypeOut );
				} else {
					throw( new MovieWriterExcInvalidPath() );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeOut->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Video );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeOut->SetGUID( MF_MT_SUBTYPE, codecToMFVideoCodecKey( mFormat.getCodec() ) );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeOut->SetUINT32( MF_MT_AVG_BITRATE, mFormat.getVideoBitRate() );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeOut->SetUINT32( MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeSize( pVideoTypeOut, MF_MT_FRAME_SIZE, mWidth, mHeight );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeRatio( pVideoTypeOut, MF_MT_FRAME_RATE, mFormat.getFrameRate(), 1 );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeRatio( pVideoTypeOut, MF_MT_PIXEL_ASPECT_RATIO, 1, 1 );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pSinkWriter->AddStream( pVideoTypeOut, &streamIndex );
				}

				// Set the input media type.
				if( SUCCEEDED( hr ) ) {
					hr = MFCreateMediaType( &pVideoTypeIn );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeIn->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Video );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeIn->SetGUID( MF_MT_SUBTYPE, mFormat.getVideoInputFormat() );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pVideoTypeIn->SetUINT32( MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeSize( pVideoTypeIn, MF_MT_FRAME_SIZE, mWidth, mHeight );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeRatio( pVideoTypeIn, MF_MT_FRAME_RATE, mFormat.getFrameRate(), 1 );
				}

				if( SUCCEEDED( hr ) ) {
					hr = MFSetAttributeRatio( pVideoTypeIn, MF_MT_PIXEL_ASPECT_RATIO, 1, 1 );
				}

				if( SUCCEEDED( hr ) ) {
					hr = pSinkWriter->SetInputMediaType( streamIndex, pVideoTypeIn, NULL );

					if (hr == MF_E_INVALIDMEDIATYPE) {
						app::console() << "The underlying media sink does not support the format" << std::endl;
					} else if (hr == MF_E_INVALIDSTREAMNUMBER) {
						app::console() << "The dwStreamIndex parameter is invalid." << std::endl;
					} else if (hr == MF_E_TOPO_CODEC_NOT_FOUND) {
						app::console() << "Could not find an encoder for the encoded format." << std::endl;
					}
				}

				// Tell the sink writer to start accepting data.
				if( SUCCEEDED( hr ) ) {
					hr = pSinkWriter->BeginWriting();
				}

				// Return the pointer to the caller.
				if( SUCCEEDED( hr ) ) {
					mSinkWriter = pSinkWriter;
					mSinkWriter->AddRef();
					mStreamIndex = streamIndex;
					mSinkCorrect = true;
				}

				SafeRelease( &pSinkWriter );
				SafeRelease( &pVideoTypeOut );
				SafeRelease( &pVideoTypeIn );
			}
		}

		MovieWriter::~MovieWriter()
		{
			if( !mFinished ) {
				finish();
			}
		}

		void MovieWriter::addFrame( Surface imageSource, float duration )
		{
			if( mFinished ) {
				throw MovieWriterExcAlreadyFinished();
			}

			if( !mSinkCorrect ) {
				throw MovieWriterExc();
			}

			LONGLONG frameDuration = duration > 0 ? duration * 1000.0 * 1000.0 * 10.0 : mFormat.getDefaultFrameDuration();

			IMFSample* pSample = NULL;
			IMFMediaBuffer* pBuffer = NULL;

			const LONG cbWidth = 4 * mWidth;
			const DWORD cbBuffer = cbWidth * mHeight;

			BYTE* pData = NULL;

			// Create a new memory buffer.
			HRESULT hr = MFCreateMemoryBuffer( cbBuffer, &pBuffer );

			// Lock the buffer and copy the video frame to the buffer.
			if( SUCCEEDED( hr ) ) {
				hr = pBuffer->Lock( &pData, NULL, NULL );
			}

			if( SUCCEEDED( hr ) ) {
				hr = MFCopyImage(
				         pData,                 // Destination buffer.
				         cbWidth,               // Destination stride.
				         imageSource.getData(),
				         cbWidth,               // Source stride.
				         cbWidth,               // Image width in bytes.
				         mHeight                // Image height in pixels.
				     );
			}

			if( pBuffer ) {
				pBuffer->Unlock();
			}

			// Set the data length of the buffer.
			if( SUCCEEDED( hr ) ) {
				hr = pBuffer->SetCurrentLength( cbBuffer );
			}

			// Create a media sample and add the buffer to the sample.
			if( SUCCEEDED( hr ) ) {
				hr = MFCreateSample( &pSample );
			}

			if( SUCCEEDED( hr ) ) {
				hr = pSample->AddBuffer( pBuffer );
			}

			// Set the time stamp and the duration.
			if( SUCCEEDED( hr ) ) {
				hr = pSample->SetSampleTime( mCurrentSeconds );
			}

			if( SUCCEEDED( hr ) ) {
				hr = pSample->SetSampleDuration( duration );

			}

			// Send the sample to the Sink Writer.
			if( SUCCEEDED( hr ) ) {
				hr = mSinkWriter->WriteSample( mStreamIndex, pSample );
			}

			mCurrentSeconds += frameDuration;
			++mNumFrames;
			SafeRelease( &pSample );
			SafeRelease( &pBuffer );
		}

		void cinder::wmf::MovieWriter::addFrame( gl::TextureRef textureSource, float duration )
		{
			if( mFinished ) {
				throw MovieWriterExcAlreadyFinished();
			}

			if( !mSinkCorrect ) {
				throw MovieWriterExc();
			}

			gl::pushMatrices();
			gl::ScopedFramebuffer fbScp( mFbo );
			gl::clear( Color( 1.0f, 1.0f, 1.0f ) );
			gl::setMatricesWindow( mFbo->getSize() );
			gl::color( Color( 1.0f, 1.0f, 1.0f ) );
			gl::ScopedViewport scpVp( ivec2( 0 ), mFbo->getSize() );

			if( textureSource ) {
				textureSource->bind();
				mGlsl->uniform( "uTexture", 0 );
			}

			mGlsl->uniform( "uFlipY", mFormat.getCodec() == H264 );

			mBatch->draw();

			gl::popMatrices();

			ci::Surface composite = mFbo->getColorTexture()->createSource();

			addFrame( composite, duration );
		}

		void MovieWriter::finish()
		{
			if( mFinished ) {
				return;
			}

			if( mSinkCorrect ) {
				HRESULT hr = mSinkWriter->Finalize();
			}

			SafeRelease( &mSinkWriter );
			MFShutdown();
			mSinkCorrect = false;
			mFinished = true;
		}
	}
}