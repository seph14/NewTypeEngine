#pragma once

#include "newtype/timeline/EventTrigger.h"
#include "cinder/Json.h"
#include "cinder/Filesystem.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace newtype::timeline {

class Timeline {
public:
    // Playback
    void play();
    void pause();
    void stop();
    void setTime(float t);
    float currentTime() const { return _currentTime; }
    bool isPlaying() const { return _playing; }

    // Per-frame: evaluate all active events, fire callbacks
    void evaluate(float dt);

    // Event management
    void addEvent(std::shared_ptr<EventTrigger> evt);
    void removeEvent(const std::string& name);
    std::shared_ptr<EventTrigger> findEvent(const std::string& name);
    const std::vector<std::shared_ptr<EventTrigger>>& events() const { return _events; }

    // Serialization
    ci::Json toJson() const;
    void fromJson(const ci::Json& j);
    void load(const ci::fs::path& path);
    void save(const ci::fs::path& path);
    void reset();

    // UI
    bool drawUi();
    void drawTimeline(float time, float winHeight = 320.f, float maxWidth = 900.f);
    void drawEditor();

    float duration() const { return _duration; }

private:
    std::vector<std::shared_ptr<EventTrigger>> _events;
    float _currentTime = 0.f;
    float _duration = 0.f;
    bool _playing = false;
    bool _looping = true;
    bool _updateEnabled = true;
    bool _showTimeline = false;
    int _selectedEventIdx = -1;
    bool _scrollToSelected = false;

    void sortEvents();
    void recalcDuration();
};

} // namespace newtype::timeline
