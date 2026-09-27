#include "newtype/util/Recorder.h"
#include "newtype/core/Renderer.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/Utilities.h"
#include "cinder/gl/gl.h"
#include <luisa/luisa-compute.h>
#if NT_ENABLE_VIDEO_RECORDER
#include "newtype/plugin/MovieWriter.h"
#endif

using namespace ci;
using namespace std;

namespace newtype::util {

VisualRecorderRef VisualRecorder::create(const ci::ivec2& size, bool glReadback) {
	return VisualRecorderRef(new VisualRecorder(size, glReadback));
}

VisualRecorder::VisualRecorder(const ci::ivec2& size, bool glReadback) {
	mRecording	= false;
	mFlip		= true;
	mGlReadback	= glReadback;

	if (mGlReadback) {
		mPbo[0]		= ci::gl::Pbo::create(GL_PIXEL_PACK_BUFFER, (size_t)size.x * size.y * 4, 0, GL_STREAM_READ);
		mPbo[1]		= ci::gl::Pbo::create(GL_PIXEL_PACK_BUFFER, (size_t)size.x * size.y * 4, 0, GL_STREAM_READ);
	} else {
		// Dx12 present mode: staging for the Luisa display-target download
		mHostBuffer.assign((size_t)size.x * size.y * 4, 0);
	}
	mSurface[0] = ci::Surface::create(size.x, size.y, true, ci::SurfaceChannelOrder::BGRA);
	mSurface[1] = ci::Surface::create(size.x, size.y, true, ci::SurfaceChannelOrder::BGRA);
	mFlipSurface= ci::Surface::create(size.x, size.y, false, ci::SurfaceChannelOrder::RGB);
	mSize		= size;

#if NT_ENABLE_VIDEO_RECORDER
	auto recordingPath = app::getAssetPath("") / "recordings";
	if (!fs::exists(recordingPath)) fs::create_directory(recordingPath);
#endif
#if NT_ENABLE_SCREENSHOT
	auto screenshotPath = app::getAssetPath("") / "screenshots";
	if (!fs::exists(screenshotPath)) fs::create_directory(screenshotPath);
#endif
}

const string VisualRecorder::getTimestamp() {
	auto t	= std::time(nullptr);
	auto tm = *std::localtime(&t);
	stringstream ss;
	ss << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S");
	return ss.str();
}

void VisualRecorder::update(ci::gl::FboRef buffer) {
	// The full-window glReadPixels is expensive; only pay it when the data
	// has a consumer this frame — the video encoder or a pending one-shot
	// screenshot capture. (This used to run unconditionally every frame even
	// when idle.)
#if NT_ENABLE_SCREENSHOT
	const bool needCapture = mRecording || mScreenshotPending;
#else
	const bool needCapture = mRecording;
#endif
	if (!needCapture) return;

	{
		ci::gl::ScopedFramebuffer backFbo(buffer);
		ci::gl::ScopedBuffer      frontPbo(mPbo[mIdx]);
		ci::gl::readBuffer(GL_COLOR_ATTACHMENT0);
		ci::gl::readPixels(0, 0, mSize.x, mSize.y, GL_BGRA, GL_UNSIGNED_BYTE, 0);
	}

#if NT_ENABLE_VIDEO_RECORDER
	if (mRecording && mRecorder) {
		if (mFirstVideoFrame) {
			// The gated readback no longer keeps both PBOs fresh while idle:
			// on the first frame of a session the "other" PBO still holds
			// data from the previous session (or is uninitialized). Pull the
			// PBO filled above instead — blocking, but a one-shot cost on a
			// user action (same class as the screenshot capture).
			mFirstVideoFrame = false;
			mPbo[mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
		} else {
			mPbo[1 - mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
		}
		mRecorder->addFrame(*mSurface[mIdx]);
	}
#endif

#if NT_ENABLE_SCREENSHOT
	if (mScreenshotPending) {
		mScreenshotPending = false;
		// Blocking pull of the PBO filled above (glGetBufferSubData waits for
		// the async readback) — acceptable for a one-shot user action. The
		// recording path deliberately reads the *other* PBO to avoid this stall.
		mPbo[mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
		const fs::path out = mScreenshotPath.empty()
			? app::getAssetPath("screenshots") / (getTimestamp() + ".jpg")
			: mScreenshotPath;
		mScreenshotPath.clear();
		if (mFlip) {
			mFlipSurface->copyFromFlipped(*mSurface[mIdx], mFlipSurface->getBounds());
			writeImage(writeFile(out), *mFlipSurface);
		} else {
			writeImage(writeFile(out), *mSurface[mIdx]);
		}
	}
#endif

	mIdx = 1 - mIdx;
}

void VisualRecorder::updateDx(const luisa::compute::Image<float>& displayTarget) {
	// Same gating as update(): the full-window download is only worth it when
	// a consumer exists this frame (video encoder or pending screenshot).
#if NT_ENABLE_SCREENSHOT
	const bool needCapture = mRecording || mScreenshotPending;
#else
	const bool needCapture = mRecording;
#endif
	if (!needCapture) return;
	if (mGlReadback || mHostBuffer.empty()) return;
	if (displayTarget.size().x != static_cast<uint>(mSize.x) ||
		displayTarget.size().y != static_cast<uint>(mSize.y))
		return; // transient mismatch during resize; try again next frame

	// Blocking download on Luisa's stream (RGBA8 = 4 B/px). updateRecorder()
	// calls this before the frame's render submit, so the synchronize only
	// waits for work Pipeline::beginFrame() would have waited for anyway.
	auto& stream = newtype::core::Renderer::stream();
	stream << displayTarget.copy_to(luisa::span<uint8_t>{ mHostBuffer.data(), mHostBuffer.size() });
	stream << luisa::compute::synchronize();

	// Luisa BYTE4 download is RGBA; the surfaces (and MovieWriter/JPEG
	// writers) consume BGRA like the glReadPixels path — swizzle while
	// copying. Rows arrive bottom-up (GL row convention, same as
	// glReadPixels), so the screenshot path flips on save exactly like the
	// GL readback path does.
	{
		auto* __restrict dst = mSurface[mIdx]->getData();
		const uint8_t* __restrict src = mHostBuffer.data();
		const size_t pixelCount = (size_t)mSize.x * mSize.y;
		for (size_t i = 0; i < pixelCount; ++i) {
			dst[i * 4 + 0] = src[i * 4 + 2];
			dst[i * 4 + 1] = src[i * 4 + 1];
			dst[i * 4 + 2] = src[i * 4 + 0];
			dst[i * 4 + 3] = src[i * 4 + 3];
		}
	}

#if NT_ENABLE_VIDEO_RECORDER
	if (mRecording && mRecorder)
		mRecorder->addFrame(*mSurface[mIdx]);
#endif

#if NT_ENABLE_SCREENSHOT
	if (mScreenshotPending) {
		mScreenshotPending = false;
		const fs::path out = mScreenshotPath.empty()
			? app::getAssetPath("screenshots") / (getTimestamp() + ".jpg")
			: mScreenshotPath;
		mScreenshotPath.clear();
		if (mFlip) {
			mFlipSurface->copyFromFlipped(*mSurface[mIdx], mFlipSurface->getBounds());
			writeImage(writeFile(out), *mFlipSurface);
		} else {
			writeImage(writeFile(out), *mSurface[mIdx]);
		}
	}
#endif

	mIdx = 1 - mIdx;
}

void VisualRecorder::startRecording(const float& fps, const int& bitrate) {
	if (mRecording) return;

#if NT_ENABLE_VIDEO_RECORDER
	mRecording		= true;
	mFirstVideoFrame= true;
	auto savepath	= app::getAssetPath("recordings") / (getTimestamp() + ".mp4");
	mRecorder		= wmf::MovieWriter::create(
		savepath,
		mSurface[0]->getWidth(), mSurface[0]->getHeight(),
		wmf::MovieWriter::Format()
			.bitrate(bitrate)
			.fps	(fps)
			.codec	(wmf::MovieWriter::Codec::H264)
	);
#endif
}

void VisualRecorder::stopRecording() {
#if NT_ENABLE_VIDEO_RECORDER
	if (mRecording && mRecorder) {
		mRecorder->finish();
		mRecorder.reset();
		mRecording = false;
	}
#endif
}

ci::Surface8uRef VisualRecorder::fetchLastSurface() {
	if (!mGlReadback || !mPbo[1 - mIdx])
		return nullptr; // Dx12 mode: use updateDx()'s surfaces instead
	if (!mRecording)
		mPbo[1 - mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
	return mSurface[mIdx];
}

#if NT_ENABLE_SCREENSHOT
void VisualRecorder::saveScreenshot() {
    // Deferred: update() performs the capture at the end of the next draw()
    // (the unconditional per-frame readback it used to rely on is now gated
    // on recording/screenshot activity). Returns immediately — safe from
    // keyDown handlers and UI buttons alike.
    mScreenshotPending = true;
    mScreenshotPath.clear();
}

void VisualRecorder::saveScreenshotTo(const std::filesystem::path& path) {
    mScreenshotPending = true;
    mScreenshotPath = path;
}
#else
void VisualRecorder::saveScreenshot() {}
void VisualRecorder::saveScreenshotTo(const std::filesystem::path&) {}
#endif

bool VisualRecorder::drawUi() {
	if (ImGui::CollapsingHeader("Recorder")) {
		ImGui::ScopedId scpId("Visual Recorder");

#if NT_ENABLE_SCREENSHOT
		ImGui::Checkbox("Flip Screenshot", &mFlip);
		if (ImGui::Button("Save Screenshot"))
			saveScreenshot();
#endif

#if NT_ENABLE_VIDEO_RECORDER
		if (mRecording && mRecorder) {
			ImGui::Text(("Recording " + toString(mRecorder->getNumFrames()) + " frames").c_str());
			if (ImGui::Button("Stop Recording")) {
				stopRecording();
				return true;
			}
		} else {
			static float fps = glm::min(60.f, app::getFrameRate());
			static int	 bitrate = 25000000;
			ImGui::InputFloat("Fps", &fps);
			ImGui::InputInt("Bitrate", &bitrate);
			if (ImGui::Button("Start Recording")) {
				startRecording(fps, bitrate);
				return true;
			}
		}
#endif
	}

	return false;
}

} // namespace newtype::util
