#pragma once

#include "cinder/Json.h"
#include "cinder/Vector.h"

namespace newtype::timeline {

class EasingCurve {
public:
    EasingCurve();
    EasingCurve(const EasingCurve& other);

    ci::vec2 sample(float ratio) const;

    void setStartPos(ci::vec2 pos);
    void setEndPos(ci::vec2 pos);

    ci::Json toJson() const;
    void fromJson(const ci::Json& j);

    bool drawUi(const std::string& name);

private:
    struct CurveData {
        float points[4]; // cubic bezier control points (P1x,P1y,P2x,P2y)
        float type = 0.f;
    } mData;
};

} // namespace newtype::timeline
