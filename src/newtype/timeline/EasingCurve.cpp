#include "newtype/timeline/EasingCurve.h"
#include "newtype/timeline/CurveUI.h"
#include "imgui/imgui_internal.h"

using namespace ci;

namespace newtype::timeline {

EasingCurve::EasingCurve() {
    mData = {{0.f, 0.f, 1.f, 1.f}, 0.f};
}

EasingCurve::EasingCurve(const EasingCurve& other) {
    mData = other.mData;
}

void EasingCurve::setStartPos(ci::vec2 pos) {
    mData.points[0] = pos.x;
    mData.points[1] = pos.y;
}

void EasingCurve::setEndPos(ci::vec2 pos) {
    mData.points[2] = pos.x;
    mData.points[3] = pos.y;
}

ci::vec2 EasingCurve::sample(float t) const {
    vec2 p[4] = {
        {0, 0},
        {mData.points[0], mData.points[1]},
        {mData.points[2], mData.points[3]},
        {1, 1}
    };

    float it = 1.f - t;
    vec4 b = {
        it * it * it,
        3.f * it * it * t,
        3.f * it * t * t,
        t * t * t
    };

    return b.x * p[0] + b.y * p[1] + b.z * p[2] + b.w * p[3];
}

ci::Json EasingCurve::toJson() const {
    auto j = ci::Json::object();
    j["type"] = mData.type;
    auto arr = ci::Json::array();
    for (int i = 0; i < 4; i++)
        arr.push_back(mData.points[i]);
    j["data"] = arr;
    return j;
}

void EasingCurve::fromJson(const ci::Json& j) {
    mData.type = j["type"];
    for (int i = 0; i < 4; i++)
        mData.points[i] = j["data"][i];
}

bool EasingCurve::drawUi(const std::string& name) {
    return ImGui::Bezier(name.c_str(), mData.points);
}

} // namespace newtype::timeline
