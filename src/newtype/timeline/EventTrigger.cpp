#include "newtype/timeline/EventTrigger.h"
#include "cinder/CinderImGui.h"
#include "cinder/Utilities.h"
#include <algorithm>

using namespace ci;
using namespace std;

namespace newtype::timeline {

static const float FrameDuration = 1.f / 30.f;

// ============================================================================
// Base EventTrigger
// ============================================================================

void EventTrigger::offsetTime(float offset) {
    startTime += offset;
    endTime += offset;
}

float EventTrigger::evaluate(float time) {
    if (!enabled || _triggered) return 0.f;

    if (mode == TriggerMode::Pulse) {
        auto res = time >= startTime ? triggerValue : 0.f;
        if (res > .5f) _triggered = true;
        return res;
    } 
    else if (mode == TriggerMode::Repeat) {
        float dt = (time - startTime) / glm::max(FLT_EPSILON, duration);
        int idx = (int)glm::floor(dt);
        if (time >= startTime && time < endTime &&
            glm::min(glm::fract(dt), glm::fract(1.f - dt)) * duration < FrameDuration) {
            const_cast<EventTrigger*>(this)->_triggerIdx = idx;
            return glm::min(1.f, (.0001f + time - startTime) / (endTime - startTime));
        }
        else if (time >= endTime) _triggered = true;

        return 0.f;
    }
    else if (mode == TriggerMode::Duration) {
        if (time >= startTime && time <= startTime + duration)
            return triggerValue;
        else if (time > startTime + duration) 
            _triggered = true;
        return 0.f;
    }

    return 0.f;
}

void EventTrigger::apply(float time) {
    float progress = evaluate(time);
    if (progress > 0.f && onUpdate)
        onUpdate(progress);
}

void EventTrigger::reset() {
    _triggered = false;
    _triggerIdx = -1;
}

bool EventTrigger::drawUiContent(bool forceOpen) {
    ImGui::InputText("Evt Name", &name);
    ImGui::InputText("Target", &targetName);

    if(forceOpen) ImGui::SetNextItemOpen(forceOpen);
    if (ImGui::CollapsingHeader("Event")) {
        {
            int option = (int)mode;
            static vector<string> optionName = {"Pulse", "Repeat", "Duration"};
            if (ImGui::Combo("Type", &option, optionName)) {
                mode = (TriggerMode)option;
                if (mode == TriggerMode::Repeat)
                    duration = glm::max(1.f, duration);
            }
        }

        ImGui::Checkbox("Enabled", &enabled);
        ImGui::DragFloat("Trigger Value", &triggerValue, 0.01f);
        ImGui::DragFloat("Time", &startTime);
        if (mode == TriggerMode::Repeat || mode == TriggerMode::Duration)
            if (ImGui::DragFloat("Duration", &duration))
                duration = glm::max(.01f, duration);
        if (mode == TriggerMode::Repeat)
            ImGui::DragFloat("End Time", &endTime);
        return true;
    }
    return false;
}

bool EventTrigger::drawUi(uint8_t idx, bool forceOpen) {
    ImGui::ScopedId scpId(to_string(idx).c_str());
    drawUiContent(forceOpen);
    return ImGui::Button("Remove");
}

bool EventTrigger::drawTimeline(ImDrawList* draw_list, float yy, float length,
                                ImVec2 pos, ImVec2 size, int evtIndex) {
    static const int LineHeight = 8;

    if (mode == TriggerMode::Pulse) {
        float start = startTime / length;
        draw_list->AddCircleFilled(
            ImVec2((float)pos.x + start * size.x, yy),
            LineHeight, _triggered ? 0xFFFF0000 : 0xFF606060);
        draw_list->AddText(
            ImVec2((float)pos.x + start * size.x, yy + LineHeight),
            0xFF606060, name.c_str());
    }
    else if (mode == TriggerMode::Repeat) {
        float start = startTime / length;
        float step  = duration / length;
        float end   = endTime / length;
        draw_list->AddText(
            ImVec2((float)pos.x + start * size.x, yy + LineHeight),
            0xfff9ac06, name.c_str());
        int idx = 0;
        while (start < end) {
            draw_list->AddCircleFilled(
                ImVec2((float)pos.x + start * size.x, yy), LineHeight,
                _triggerIdx > idx ? 0xFFFF0000 : 0xfff9ac06);
            start += step;
            idx++;
        }
    }
    else if (mode == TriggerMode::Duration) {
        float start = startTime / length;
        float end   = (startTime + duration) / length;
        draw_list->AddLine(
            ImVec2((float)pos.x + start * size.x, yy),
            ImVec2((float)pos.x + end * size.x, yy),
            _triggered ? 0xFFFF0000 : 0xff04ff04, LineHeight);
        draw_list->AddText(
            ImVec2((float)pos.x + start * size.x, yy + LineHeight),
            0xff04ff04, name.c_str());
    }

    ImGui::SetItemAllowOverlap();
    if (ImGui::InvisibleButton(
        ("evt_timeline_" + to_string(evtIndex)).c_str(), ImVec2(size.x, 27)))
        return true;
    return false;
}

ci::Json EventTrigger::toJson() const {
    auto j = ci::Json::object();
    j["type"]     = (int)mode;
    j["time"]     = startTime;
    j["duration"] = duration;
    j["name"]     = name;
    j["evt"]      = typeName();
    j["end"]      = endTime;
    j["target"]   = targetName;
    j["enabled"]  = enabled;
    j["triggerValue"] = triggerValue;
    return j;
}

void EventTrigger::fromJson(const ci::Json& j) {
    name     = j.value("name", "Trigger");
    duration = j.value("duration", 0.f);
    startTime = j.value("time", 0.f);
    mode     = (TriggerMode)j.value("type", 0);
    endTime  = j.value("end", 400.f);
    targetName = j.value("target", std::string());
    enabled  = j.value("enabled", true);
    triggerValue = j.value("triggerValue", 1.f);
}

// ============================================================================
// ColorTrigger
// ============================================================================

ci::ColorA ColorTrigger::evaluateColor(float time) const {
    ColorA res;
    if (mode == TriggerMode::Duration) {
        if (time >= startTime && time <= startTime + duration)
            gradient.getColorAt((time - startTime) / duration, &res);
        else
            gradient.getColorAt(0.f, &res);
    } else {
        gradient.getColorAt(0.f, &res);
    }
    return res;
}

bool ColorTrigger::drawUi(uint8_t idx, bool forceOpen) {
    bool updateGrad = false, rmv = false;

    {
        ImGui::ScopedId scpId(to_string(idx).c_str());
        if (drawUiContent(forceOpen))
            updateGrad = ImGui::GradientButton(&gradient);
        rmv = ImGui::Button("Remove");
    }

    {
        if (updateGrad)
            ImGui::OpenPopup(("Edit Gradient " + to_string(idx) + name).c_str());
        if (ImGui::BeginPopup(("Edit Gradient " + to_string(idx) + name).c_str())) {
            static GradientMark* draggingMark = nullptr;
            static GradientMark* selectedMark = nullptr;
            ImGui::GradientEditor(&gradient, draggingMark, selectedMark);
            if (ImGui::Button("Close"))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    return rmv;
}

ci::Json ColorTrigger::toJson() const {
    auto j = EventTrigger::toJson();
    j["evt"] = "color";
    j["gradient"] = gradient.toJson();
    return j;
}

void ColorTrigger::fromJson(const ci::Json& j) {
    EventTrigger::fromJson(j);
    if (j.contains("gradient"))
        gradient.fromJson(j["gradient"]);
}

// ============================================================================
// CurveTrigger
// ============================================================================

float CurveTrigger::evaluateValue(float time) {
    if (mode == TriggerMode::Duration) {
        if (time >= startTime && time <= startTime + duration) {
            float ratio = (time - startTime) / duration;
            return valueStart + (valueEnd - valueStart) * curve.sample(ratio).y;
        }
    }
    return evaluate(time);
}

bool CurveTrigger::drawUi(uint8_t idx, bool forceOpen) {
    ImGui::ScopedId scpId(to_string(idx).c_str());
    if (drawUiContent(forceOpen)) {
        curve.drawUi("Pulse Curve");
        ImGui::DragFloat("Value Start", &valueStart);
        ImGui::DragFloat("Value End", &valueEnd);
    }
    return ImGui::Button("Remove");
}

ci::Json CurveTrigger::toJson() const {
    auto j = EventTrigger::toJson();
    j["evt"] = "curve";
    j["curve"] = curve.toJson();
    j["valueStart"] = valueStart;
    j["valueEnd"] = valueEnd;
    return j;
}

void CurveTrigger::fromJson(const ci::Json& j) {
    EventTrigger::fromJson(j);
    if (j.contains("curve"))
        curve.fromJson(j["curve"]);
    valueStart = j.value("valueStart", 0.f);
    valueEnd   = j.value("valueEnd", 1.f);
}

// ============================================================================
// PathTrigger
// ============================================================================

PathTrigger::PathTrigger() {
    name = "PathTrigger";
    path.setPathData(std::vector<ci::vec3>{
        ci::vec3(.0f), ci::vec3(.0f, .0f, 1.f),
        ci::vec3(.0f, .0f, 2.f), ci::vec3(.0f, .0f, 3.f)
    });
}

ci::vec3 PathTrigger::evaluatePosition(float time) const {
    if (mode == TriggerMode::Duration) {
        if (time >= startTime && time <= startTime + duration) {
            float r = (time - startTime) / duration;
            auto pos = path.getPosition(r > 1.f ? glm::fract(r) : r);
            return ci::vec3(pos.x, pos.y, pos.z);
        }
    }
    return ci::vec3(0.f);
}

bool PathTrigger::drawUi(uint8_t idx, bool forceOpen) {
    ImGui::ScopedId scpId(to_string(idx).c_str());
    if (drawUiContent(forceOpen)) {
#if NT_ENABLE_TIMELINE_EDITOR
        path.drawUi("Path");
#endif
    }
    return ImGui::Button("Remove");
}

ci::Json PathTrigger::toJson() const {
    auto j = EventTrigger::toJson();
    j["evt"] = "path";
    j["path"] = path.toJson();
    return j;
}

void PathTrigger::fromJson(const ci::Json& j) {
    EventTrigger::fromJson(j);
    if (j.contains("path"))
        path = loadPath<3>(j["path"]);
}

#if NT_ENABLE_TIMELINE_EDITOR
void PathTrigger::drawEditor() {
    path.drawDebugPoints(0.f, Color::hex(0xff0000));
}

void PathTrigger::drawDebugPoints(float ratio) {
    path.drawDebugPoints(ratio, Color::hex(0xff0000));
}
#endif

// ============================================================================
// Factory
// ============================================================================

std::shared_ptr<EventTrigger> createEvent(const std::string& type) {
    if (type == "color") return std::make_shared<ColorTrigger>();
    if (type == "curve") return std::make_shared<CurveTrigger>();
    if (type == "path")  return std::make_shared<PathTrigger>();
    return std::make_shared<EventTrigger>();
}

} // namespace newtype::timeline
