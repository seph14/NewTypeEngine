#pragma once
#include "newtype/core/Config.h"
#include "cinder/gl/Pbo.h"
#include "cinder/gl/Fbo.h"
#include "cinder/Surface.h"
#include <filesystem>
#if NT_ENABLE_VIDEO_RECORDER
#include "newtype/plugin/MovieWriter.h"
#endif

// Forward declaration — full Luisa include only in Recorder.cpp
namespace luisa::compute { template<typename T> class Image; }

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
	bool			mFlip;
	bool			mScreenshotPending = false; // one-shot: captured by the next update()
	ci::ivec2		mSize;
#if NT_ENABLE_VIDEO_RECORDER
	ci::wmf::MovieWriterRef mRecorder;
	bool			mFirstVideoFrame = false; // update() pulls the just-filled PBO once (other PBO is stale after the gated readback)
#endif

	// glReadback=false (Dx12 present mode): no PBOs/GL access — captures go
	// through updateDx(), a blocking Luisa download of the display target.
	VisualRecorder(const ci::ivec2& size, bool glReadback = true);
	static VisualRecorderRef create(const ci::ivec2& size, bool glReadback = true);

	static const std::string getTimestamp();
	ci::Surface8uRef fetchLastSurface();

	const bool& isRecording() { return mRecording; }
	void update			(ci::gl::FboRef buffer);
	void updateDx		(const luisa::compute::Image<float>& displayTarget);
	// Always declared — no-ops when their feature macro is 0, so callers
	// (key handlers, UI) never need to guard on the macro themselves.
	void startRecording	(const float& fps = 30.f, const int& bitrate = 25000000);
	void stopRecording	();
	void saveScreenshot ();
	// Same deferred one-shot capture, but written to `path` (extension picks
	// the encoder, e.g. .png for lossless panoramic stills) instead of the
	// timestamped screenshots/ default. Used by the --shot CLI harness.
	void saveScreenshotTo(const std::filesystem::path& path);
	bool drawUi			();

private:
	bool			mGlReadback = true;
	std::vector<uint8_t> mHostBuffer; // Dx12 staging: RGBA8 bytes from Luisa download
	std::filesystem::path mScreenshotPath; // explicit one-shot destination (empty = timestamped default)
};

}
}
