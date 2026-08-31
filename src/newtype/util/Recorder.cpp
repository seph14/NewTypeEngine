#include "newtype/util/Recorder.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/Utilities.h"
#include "cinder/gl/gl.h"
#if NT_ENABLE_VIDEO_RECORDER
#include "newtype/plugin/MovieWriter.h"
#endif

using namespace ci;
using namespace std;

namespace newtype::util {

VisualRecorderRef VisualRecorder::create(const ci::ivec2& size) {
	return VisualRecorderRef(new VisualRecorder(size));
}

VisualRecorder::VisualRecorder(const ci::ivec2& size) {
	mRecording	= false;
	mFlip		= true;

	mPbo[0]		= ci::gl::Pbo::create(GL_PIXEL_PACK_BUFFER, (size_t)size.x * size.y * 4, 0, GL_STREAM_READ);
	mPbo[1]		= ci::gl::Pbo::create(GL_PIXEL_PACK_BUFFER, (size_t)size.x * size.y * 4, 0, GL_STREAM_READ);
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
	{
		ci::gl::ScopedFramebuffer backFbo(buffer);
		ci::gl::ScopedBuffer      frontPbo(mPbo[mIdx]);
		ci::gl::readBuffer(GL_COLOR_ATTACHMENT0);
		ci::gl::readPixels(0, 0, mSize.x, mSize.y, GL_BGRA, GL_UNSIGNED_BYTE, 0);
	}

#if NT_ENABLE_VIDEO_RECORDER
	if (mRecording && mRecorder) {
		mPbo[1 - mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
		mRecorder->addFrame(*mSurface[mIdx]);
	}
#endif

	mIdx = 1 - mIdx;
}

void VisualRecorder::startRecording(const float& fps, const int& bitrate) {
	if (mRecording) return;

#if NT_ENABLE_VIDEO_RECORDER
	mRecording		= true;
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
	if (!mRecording)
		mPbo[1 - mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());
	return mSurface[mIdx];
}

#if NT_ENABLE_SCREENSHOT
void VisualRecorder::saveScreenshot() {
	if (!mRecording)
		mPbo[1 - mIdx]->getBufferSubData(0, (size_t)mSize.x * mSize.y * 4, mSurface[mIdx]->getData());

	if (mFlip) {
		mFlipSurface->copyFromFlipped(*mSurface[mIdx], mFlipSurface->getBounds());
		writeImage(writeFile(app::getAssetPath("screenshots") / (getTimestamp() + ".jpg")), *mFlipSurface);
	} else
		writeImage(writeFile(app::getAssetPath("screenshots") / (getTimestamp() + ".jpg")), *mSurface[mIdx]);
}
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
			static float fps = app::getFrameRate();
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
