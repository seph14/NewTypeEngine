#include "newtype/timeline/Timeline.h"
#include "cinder/CinderImGui.h"
#include "cinder/Utilities.h"
#include "cinder/app/App.h"
#include "cinder/Log.h"

using namespace ci;
using namespace std;

namespace newtype::timeline {

void Timeline::play()   { _playing = true; }
void Timeline::pause()  { _playing = false; }
void Timeline::stop()   { _playing = false; _currentTime = 0.f; reset(); }
void Timeline::setTime(float t) { _currentTime = t; }

void Timeline::evaluate(float dt) {
    if (!_playing || !_updateEnabled) return;

    _currentTime += dt;
    if (_looping && _duration > 0.f && _currentTime > _duration) {
        _currentTime = glm::mod(_currentTime, _duration);
        reset();
    }

    for (auto& evt : _events)
        evt->apply(_currentTime);
}

void Timeline::addEvent(std::shared_ptr<EventTrigger> evt) {
    _events.push_back(std::move(evt));
    recalcDuration();
}

void Timeline::removeEvent(const std::string& name) {
    _events.erase(
        std::remove_if(_events.begin(), _events.end(),
            [&](const auto& e) { return e->name == name; }),
        _events.end());
    recalcDuration();
}

std::shared_ptr<EventTrigger> Timeline::findEvent(const std::string& name) {
    for (auto& evt : _events)
        if (evt->name == name) return evt;
    return nullptr;
}

void Timeline::sortEvents() {
    std::sort(_events.begin(), _events.end(),
        [](const auto& a, const auto& b) {
            return a->startTime < b->startTime;
        });
}

void Timeline::recalcDuration() {
    _duration = 0.5f;
    for (auto& evt : _events)
        _duration = glm::max(_duration, evt->duration + evt->startTime);
}

void Timeline::reset() {
    for (auto& evt : _events)
        evt->reset();
}

ci::Json Timeline::toJson() const {
    auto arr = ci::Json::array();
    for (auto& evt : _events)
        arr.push_back(evt->toJson());
    return arr;
}

void Timeline::fromJson(const ci::Json& j) {
    _events.clear();
    for (size_t i = 0; i < j.size(); ++i) {
        auto& item = j[i];
        string type = item.value("evt", string("base"));
        auto evt = createEvent(type);
        evt->fromJson(item);
        _events.push_back(std::move(evt));
    }
    recalcDuration();
}

void Timeline::load(const fs::path& path) {
    if (!fs::exists(path)) return;
    try {
        auto j = ci::Json::parse(loadString(loadFile(path)));
        fromJson(j);
    } catch (const ci::Exception& ex) {
        CI_LOG_EXCEPTION("Timeline load error: ", ex);
    }
}

void Timeline::save(const fs::path& path) {
    auto j = toJson();
    std::ofstream out(path);
    out << j.dump(4) << std::endl;
}

bool Timeline::drawUi() {
    bool requestWire = false;

    if (ImGui::CollapsingHeader("Timeline")) {
        ImGui::ScopedId scpId("timeline");

        // Transport controls
        if (_playing) {
            if (ImGui::Button("Pause")) pause();
        } else {
            if (ImGui::Button("Play")) play();
        }

        ImGui::SameLine();
        if (ImGui::Button("Stop")) stop();
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &_looping);
        ImGui::Checkbox("Update", &_updateEnabled);
        ImGui::Checkbox("Show Timeline", &_showTimeline);
        ImGui::Checkbox("3D Path Gizmos", &_showGizmos);
        if (ImGui::Button("Require Update")) {
            requestWire = true;
            recalcDuration();
        }

        ImGui::DragFloat("Time", &_currentTime, 0.1f, 0.f, _duration);
        ImGui::Text("Duration: %.1f", _duration);

        ImGui::Dummy(ImVec2(0, 5));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 5));

        // Add event
        {
            static int evtType = 0;
            static vector<string> evtTypes = { "Base", "Color", "Curve", "Path" };
            ImGui::Combo("Event Type", &evtType, evtTypes);
            if (ImGui::Button("Add Event")) {
                switch (evtType) {
                case 0: addEvent(std::make_shared<EventTrigger>()); break;
                case 1: addEvent(std::make_shared<ColorTrigger>()); break;
                case 2: addEvent(std::make_shared<CurveTrigger>()); break;
                case 3: addEvent(std::make_shared<PathTrigger>()); break;
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Sort")) sortEvents();
        ImGui::SameLine();
        if (ImGui::Button("Save")) save(app::getAssetPath("") /"timeline.json");
        ImGui::SameLine();
        if (ImGui::Button("Load")) load(app::getAssetPath("") / "timeline.json");
    }

    return requestWire;
}

void Timeline::drawTimeline(float time, float winHeight, float winMaxWidth) {
    if (!_showTimeline) return;

    float tlength = glm::max(30.f, _duration);

    // Calculate window widths: 45% for Events, 55% for Timeline
    float availWidth = winMaxWidth > 0.f ? winMaxWidth : 700.f;
    float w0 = 5.f / 11.f * availWidth;  // 45% - Events window
    float w1 = 6.f / 11.f * availWidth;  // 55% - Timeline strip window
    
    {
        //==========================================================================
        // Left window: Events list (45%)
        //==========================================================================
        //ImGui::SetNextWindowSize(ImVec2(w0, winHeight), ImGuiCond_FirstUseEver);
        //ImGui::SetNextWindowPos(ImVec2(5, ImGui::GetIO().DisplaySize.y - winHeight - 5), ImGuiCond_FirstUseEver);
        ImGui::ScopedWindow scpEvents("Events", true);
        ImGui::SetWindowSize(ImVec2(w0, winHeight));
        ImGui::SetWindowPos(ImVec2(5, ImGui::GetIO().DisplaySize.y - winHeight - 5));

        // Transport controls (moved from drawUi)
        ImGui::Text("Timeline");
        
        // Event list
        uint8_t idx = 0;
        int rmvIdx = -1;
        vector<ci::Json> toAdd;
        for (auto& evt : _events) {
            bool isSelected = (int)idx == _selectedEventIdx;
            if (_scrollToSelected)
                ImGui::SetNextItemOpen(isSelected);
            if (evt->drawUi(idx++, isSelected && _scrollToSelected))
                rmvIdx = idx - 1;
            if (isSelected && _scrollToSelected) {
                ImGui::SetScrollHereY(0.3f);
                _scrollToSelected = false;
            }
            {
                ImGui::ScopedId scpCopy(idx);
                if (ImGui::Button("Copy"))
                    toAdd.push_back(evt->toJson());
            }
            ImGui::Dummy(ImVec2(0, 5));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0, 5));
        }

        if (rmvIdx >= 0)
            _events.erase(_events.begin() + rmvIdx);
        for (auto& item : toAdd) {
            string type = item.value("evt", string("base"));
            auto evt = createEvent(type);
            evt->fromJson(item);
            _events.push_back(std::move(evt));
        }
        if (rmvIdx >= 0 || !toAdd.empty())
            recalcDuration();
    }
    
    //==========================================================================
    // Right window: Timeline strip (55%)
    //==========================================================================
    {
        //ImGui::SetNextWindowSize(ImVec2(w1, winHeight), ImGuiCond_FirstUseEver);
        //ImGui::SetNextWindowPos(ImVec2(w0 + 10, ImGui::GetIO().DisplaySize.y - winHeight - 5), ImGuiCond_FirstUseEver);
        ImGui::ScopedWindow scpTimeline("Timeline", true);
        ImGui::SetWindowSize(ImVec2(w1, winHeight));
        ImGui::SetWindowPos(ImVec2(w0 + 10, ImGui::GetIO().DisplaySize.y - winHeight - 5));

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();

        float yy = canvas_pos.y;
        int evtIdx = 0;
        for (auto& evt : _events) {
            if (evt->drawTimeline(draw_list, yy, tlength + 5.f, canvas_pos, canvas_size, evtIdx)) {
                _selectedEventIdx = evtIdx;
                _scrollToSelected = true;
            }
            yy += 32;
            evtIdx++;
        }

        // Playhead
        float center = time / (tlength + 5.f);
        draw_list->AddRectFilled(
            ImVec2((float)canvas_pos.x + glm::max(0.f, center * canvas_size.x - 2.5f), 0.f),
            ImVec2((float)canvas_pos.x + glm::min(canvas_size.x, center * canvas_size.x + 2.5f), yy),
            0xffffffff);
    }
}

void Timeline::drawEditor() {
#if NT_ENABLE_TIMELINE_EDITOR
    // Playhead ratio so the 3D marker tracks playback (a node must be
    // selected in the path editor for the marker to show).
    const float ratio = _duration > 1e-3f
        ? glm::clamp(_currentTime / _duration, 0.f, 1.f) : 0.f;
    for (auto& evt : _events) {
        if (auto* pathEvt = dynamic_cast<PathTrigger*>(evt.get()))
            pathEvt->drawDebugPoints(ratio);
    }
#endif
}

} // namespace newtype::timeline
