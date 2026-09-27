#pragma once

#include "cinder/BSpline.h"
#include "cinder/Quaternion.h"
#include "cinder/Log.h"
#include "cinder/Json.h"
#include "newtype/core/Config.h"
#include "newtype/timeline/EasingCurve.h"

namespace newtype::timeline {

template<int D>
class PathHelper {
public:
    using vecP = typename ci::VECDIM<D, float>::TYPE;

    PathHelper(std::vector<vecP> path);
    PathHelper();
    ~PathHelper() = default;

    [[nodiscard]] vecP sampleCurve(float ratio) const { return mCurve.getPosition(ratio); }
    [[nodiscard]] vecP getCurveDirection(float r) const { return mCurve.getDerivative(r); }

    [[nodiscard]] ci::vec3 getNormalizedDirection(float ratio) const;
    [[nodiscard]] ci::mat4 getTransform(float ratio, bool includeRotation = true) const;
    [[nodiscard]] ci::mat4 getRotation(float ratio) const;
    [[nodiscard]] vecP     getPosition(float ratio) const;
    [[nodiscard]] float    getLength() const { return mLength; }

    void getCurveData(float ratio, vecP* pos, vecP* dir);
    void getNormalizedData(float ratio, vecP* pos, vecP* dir);
    void setPathData(std::vector<vecP> path);

    ci::BSpline<D, float>& getSpline() { return mCurve; }
    EasingCurve& getEasing() { return mEasing; }

    ci::Json toJson() const;
    void fromJson(const ci::Json& j);

#if NT_ENABLE_TIMELINE_EDITOR
    void drawDebugPoints(float ratio, ci::Color controlPnt,
                         ci::Color currPos = ci::Color::hex(0xff0000));
    bool drawUi(const std::string& name);
#endif

private:
    ci::BSpline<D, float> mCurve;
    EasingCurve mEasing;
    ci::vec3  mRootPos;
    ci::quat  mRootRot;
    float     mLength = 0.f;

#if NT_ENABLE_TIMELINE_EDITOR
    int mNodeIdx = -1;
    std::vector<float> mSplits;
    std::vector<vecP>  mBakedPos;   // SAMPLE_POINTS spline samples (gizmo polyline)

    void bakePoints();
    void updateSelf();
#endif
};

template<int D>
PathHelper<D> loadPath(const ci::Json& j) {
    int dim = j["dim"];
    if (dim != D) CI_LOG_E("Dimension mismatch on path loading");
    using vecP = typename ci::VECDIM<D, float>::TYPE;
    std::vector<vecP> data;
    for (auto& item : j["data"]) {
        vecP vec;
        for (int i = 0; i < D; i++)
            vec[i] = item[i];
        data.push_back(vec);
    }
    auto path = PathHelper<D>(data);
    path.fromJson(j);
    return path;
}

using PathHelperPath = PathHelper<3>;

} // namespace newtype::timeline
