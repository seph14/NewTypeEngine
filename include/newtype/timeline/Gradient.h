#pragma once

#include "cinder/CinderImGui.h"
#include "cinder/Surface.h"
#include "cinder/Json.h"
#include <luisa/luisa-compute.h>
#include <list>

namespace newtype::timeline {

struct GradientMark {
    ci::ColorA color;
    float position; // 0 to 1
};

class Gradient {
public:
    Gradient();
    Gradient(const Gradient& other);
    ~Gradient();

    void getColorAt(float position, ci::ColorA* color) const;
    void addMark(float position, ImColor const color);
    void removeMark(GradientMark* mark);
    void refreshCache();
    std::list<GradientMark*>& getMarks() { return m_marks; }

    ci::Surface8uRef genSurface();
    luisa::vector<luisa::half4> genData();

    ci::Json toJson() const;
    void fromJson(const ci::Json& j);

private:
    void computeColorAt(float position, ci::ColorA* color) const;
    std::list<GradientMark*> m_marks;
    ci::ColorA m_cachedValues[256];
};

} // namespace newtype::timeline

namespace ImGui {
    bool GradientButton(newtype::timeline::Gradient* gradient);
    bool GradientEditor(newtype::timeline::Gradient* gradient,
                        newtype::timeline::GradientMark*& draggingMark,
                        newtype::timeline::GradientMark*& selectedMark);
}
