#pragma once
#include "newtype/core/Config.h"
#include "cinder/gl/Pbo.h"
#include "cinder/gl/Fbo.h"
#include "cinder/Surface.h"
#if NT_ENABLE_VIDEO_RECORDER
#include "newtype/plugin/MovieWriter.h"
#endif

namespace newtype {
namespace util {
class VisualRecorder;
typedef std::shared_ptr<VisualRecorder> VisualRecorderRef;


class VisualRecorder {
public:
	ci::gl::PboRef	mPbo[2];
	ci::SurfaceRef	mSurface[2], mFlipSurface;
	uint8_t			mIdx{ 1 };
	bool			mRecording;
#if NT_ENABLE_SCREENSHOT
	bool			mFlip;
#endif
	ci::ivec2		mSize;
#if NT_ENABLE_VIDEO_RECORDER
	ci::wmf::MovieWriterRef mRecorder;
#endif

	VisualRecorder(const ci::ivec2& size);
	static VisualRecorderRef create(const ci::ivec2& size);

	static const std::string getTimestamp();
	ci::Surface8uRef fetchLastSurface();

	const bool& isRecording() { return mRecording; }
	void update			(ci::gl::FboRef buffer);
#if NT_ENABLE_VIDEO_RECORDER
	void startRecording	(const float& fps = 30.f, const int& bitrate = 25000000);
	void stopRecording	();
#endif
#if NT_ENABLE_SCREENSHOT
	void saveScreenshot ();
#endif
	bool drawUi			();
};

}
}
