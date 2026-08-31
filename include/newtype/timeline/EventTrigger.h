#pragma once

#include "cinder/Json.h"
#include "cinder/Vector.h"
#include "newtype/timeline/EasingCurve.h"
#include "newtype/timeline/Gradient.h"
#include "newtype/timeline/PathHelper.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace newtype::timeline {

enum class TriggerMode { Pulse, Repeat, Duration };

class EventTrigger {
public:
    std::string name;
    float startTime = 0.f;
    float duration  = 0.f;
    float endTime   = 400.f;
    TriggerMode mode = TriggerMode::Pulse;
    bool enabled = true;

    std::string targetName;
    float triggerValue = 1.f;
    std::function<void(float progress)> onUpdate;

    virtual ~EventTrigger() = default;

    virtual float evaluate(float time);
    void apply(float time);
    virtual void reset();
    void offsetTime(float offset);

    // Serialization
    virtual ci::Json toJson() const;
    virtual void fromJson(const ci::Json& j);
    virtual std::string typeName() const { return "base"; }

    // UI
    virtual bool drawUi(uint8_t idx, bool forceOpen = false);
    bool drawTimeline(ImDrawList* list, float yy, float length,
                      ImVec2 pos, ImVec2 size, int evtIndex = 0);

    float getDuration()  const { return duration; }
    float getStartTime() const { return startTime; }

protected:
    bool _triggered = false;
    int  _triggerIdx = -1;
    bool drawUiContent(bool forceOpen = false);
};

class ColorTrigger : public EventTrigger {
public:
    Gradient gradient;

    ci::ColorA evaluateColor(float time) const;

    std::string typeName() const override { return "color"; }
    bool drawUi(uint8_t idx, bool forceOpen = false) override;
    ci::Json toJson() const override;
    void fromJson(const ci::Json& j) override;
};

class CurveTrigger : public EventTrigger {
public:
    EasingCurve curve;
    float valueStart = 0.f;
    float valueEnd   = 1.f;

    float evaluateValue(float time);

    std::string typeName() const override { return "curve"; }
    bool drawUi(uint8_t idx, bool forceOpen = false) override;
    ci::Json toJson() const override;
    void fromJson(const ci::Json& j) override;
};

class PathTrigger : public EventTrigger {
public:
    PathHelper<3> path;

    PathTrigger();

    ci::vec3 evaluatePosition(float time) const;

    std::string typeName() const override { return "path"; }
    bool drawUi(uint8_t idx, bool forceOpen = false) override;
    ci::Json toJson() const override;
    void fromJson(const ci::Json& j) override;

#if NT_ENABLE_TIMELINE_EDITOR
    void drawEditor();
    void drawDebugPoints(float ratio);
#endif
};

// Factory for deserialization
std::shared_ptr<EventTrigger> createEvent(const std::string& type);

} // namespace newtype::timeline
