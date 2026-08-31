#pragma once

#include "cinder/gl/gl.h"
#include "cinder/Surface.h"
#include "cinder/app/App.h"

#include <Windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <Mfreadwrite.h>
#include <mferror.h>
#include <shlwapi.h>
#include <wrl.h>

#pragma comment(lib, "mfreadwrite")
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mfuuid")

#pragma comment(lib, "mf")
#pragma comment(lib, "shlwapi")

namespace cinder {
	namespace wmf {
		template <class T> void SafeRelease( T** ppT ) {
			if( *ppT ) {
				( *ppT )->Release();
				*ppT = NULL;
			}
		}

		class MovieWriter;
		typedef std::shared_ptr<MovieWriter> MovieWriterRef;

		class MovieWriter {
			public:
				typedef enum Codec {
					H264, WMV3, H265
				} Codec;
				typedef enum FileType { QUICK_TIME_MOVIE, MPEG4, M4V, WMV } FileType;

				//! Defines the encoding parameters of a MovieWriter
				class Format {
					public:
						Format();

						//! Returns the codec of type MovieWriter::Codec. Default is \c Codec::H264.
						Codec		getCodec() const { return mCodec; }
						//! Sets the encoding codec. Default is \c Codec::H264
						Format&		codec( Codec codec ) { mCodec = codec; return *this; }
						//! Sets the encoding bitrate. Default is \c 5120000.
						Format&		bitrate( int bitrate ) { mVideoBitRate = bitrate; return *this; }

						GUID		getVideoInputFormat() { return mVideoInputFormat; };
						UINT32		getVideoBitRate() { return mVideoBitRate; };
						UINT32		getFrameRate() const { return mFPS; };
						Format&		fps( int fps ) { mFPS = fps; mDefaultFrameDuration = 10 * 1000 * 1000 / mFPS; return *this; }
						//! Returns the output file type. Default is \c FileType::QUICK_TIME_MOVIE.
						FileType	getFileType() const { return mFileType; }
						//! Sets the output file type, defined in MovieWriter::FileType. Default is \c FileType::QUICK_TIME_MOVIE.
						Format&		fileType( FileType fileType ) { mFileType = fileType; return *this; }

						Format&		hardwareAcceleration(const bool& enabled) { mHardwareAcceleration = enabled; return *this; }
						bool		getHardwareAccelerationEnabled() { return mHardwareAcceleration; }

						//! Returns the overall quality for encoding in the range of [\c 0,\c 1.0] for the JPEG codec. Defaults to \c 0.99. \c 1.0 corresponds to lossless.
						//float		getJpegQuality() const { return mJpegQuality; }
						//! Sets the overall quality for encoding when using the JPEG codec. Must be in a range of [\c 0,\c 1.0]. Defaults to \c 0.99. \c 1.0 corresponds to lossless.
						//Format&		jpegQuality(float quality);
						//! Returns the standard duration of a frame, measured in seconds. Defaults to \c 1/30 sec, meaning \c 30 FPS.
						float		getDefaultFrameDuration() const { return mDefaultFrameDuration; }
						//! Sets the default duration of a frame, measured in seconds. Defaults to \c 1/30 sec, meaning \c 30 FPS.
						Format&		defaultFrameDuration( float defaultDuration ) { mDefaultFrameDuration = defaultDuration; return *this; }
						//! Returns the integer base value for the encoding time scale. Defaults to \c 600
						//long		getTimeScale() const { return mTimeBase; }
						//! Sets the integer base value for encoding time scale. Defaults to \c 600.
						//Format&		setTimeScale(long timeScale) { mTimeBase = timeScale; return *this; }

						//! Returns the average bits per second for H.264 encoding. Defaults to no limit.
						//float		getAverageBitsPerSecond() const { return mH264AverageBitsPerSecond; }
						//! Sets the average bits per second for H.264 encoding. Defaults to no limit.
						//Format&		averageBitsPerSecond(float avgBitsPerSecond) { mH264AverageBitsPerSecondSpecified = true; mH264AverageBitsPerSecond = avgBitsPerSecond; return *this; }
						//! Resets the average bits per second to no limit.
						//void		unspecifyAverageBitsPerSecond() { mH264AverageBitsPerSecondSpecified = false; }
						//! Returns whether the average bits per second for H.264 have been specified. If \c false, no limit is imposed.
						//bool		isAverageBitsPerSecondSpecified() const { return mH264AverageBitsPerSecond; }

					private:
						Codec		mCodec;
						FileType	mFileType;
						//long		mTimeBase;
						float		mDefaultFrameDuration;
						//float		mJpegQuality;
						UINT32		mFPS;
						GUID		mVideoInputFormat;
						UINT32		mVideoBitRate;
						bool		mHardwareAcceleration;
						// H.264 only
						//bool		mFrameReorderingEnabled;
						//bool		mH264AverageBitsPerSecondSpecified;
						//float		mH264AverageBitsPerSecond;

						friend class MovieWriter;
				};

				~MovieWriter();
				static MovieWriterRef	create( const fs::path& path, int32_t width, int32_t height, const Format& format = Format::Format() ) {
					return std::shared_ptr<MovieWriter>( new MovieWriter( path, width, height, format ) );
				}

				//! Returns the Movie's default frame duration measured in seconds. You can also think of this as the Movie's frameRate.
				float	getDefaultDuration() const { return mFormat.mDefaultFrameDuration; }
				//! Returns the width of the Movie in pixels
				int32_t	getWidth() const { return mWidth; }
				//! Returns the height of the Movie in pixels
				int32_t	getHeight() const { return mHeight; }
				//! Returns the size of the Movie in pixels
				ivec2	getSize() const { return ivec2( getWidth(), getHeight() ); }
				//! Returns the Movie's aspect ratio, which is its width / height
				float	getAspectRatio() const { return getWidth() / ( float )getHeight(); }
				//! Returns the bounding Area of the Movie in pixels: [0,0]-(width,height)
				Area	getBounds() const { return Area( 0, 0, getWidth(), getHeight() ); }

				//! Returns the Movie's Format
				const Format&	getFormat() const { return mFormat; }

				//! Appends a frame to the Movie. The optional \a duration parameter allows a frame to be inserted for a durationtime other than the Format's default duration.
				void addFrame( Surface imageSource, float duration = -1.0f );
				void addFrame( gl::TextureRef textureSource, float duration = -1.0f );
				//void addFrame(float duration = -1.0f);

				//! Returns the number of frames in the movie
				uint32_t	getNumFrames() const { return mNumFrames; }

				//! Completes the encoding of the movie and closes the file. Calling finish() more than once has no effect. It is an error to call addFrame() after calling finish().
				void finish();

			private:
				MovieWriter( const fs::path& path, UINT32 width, UINT32 height, const Format& format );
				IMFSinkWriter*	mSinkWriter;
				DWORD			mStreamIndex;
				bool			mSinkCorrect;
				//LONGLONG		rtStart;

				uint32_t		mNumFrames;
				LONGLONG		mCurrentSeconds;
				UINT32			mWidth, mHeight;
				Format			mFormat;
				bool			mFinished;

				ci::gl::FboRef	mFbo;
				ci::gl::BatchRef mBatch;
				ci::gl::GlslProgRef		mGlsl;

		};

		class MovieWriterExc : public Exception
		{
		};
		class MovieWriterExcInvalidPath : public MovieWriterExc
		{
		};
		class MovieWriterExcFrameEncode : public MovieWriterExc
		{
		};
		class MovieWriterExcAlreadyFinished : public MovieWriterExc
		{
		};
	}
}
