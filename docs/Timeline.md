# Timeline Examples

`newtype::timeline::Timeline` is a CPU-side keyframe system ported from the old Sj framework. It evaluates per-frame, fires callbacks that mutate transforms / materials / arbitrary state, and ships with an ImGui transport + strip editor. Stateful playback lives in `Timeline`; pure per-trigger evaluation lives in each `EventTrigger` subclass.

Editor features (3D path viewport, ImGui curve widget) are gated by `NT_ENABLE_TIMELINE_EDITOR` in `Config.h` (default `1`).

---

## Quick Start

```cpp
#include "newtype/timeline/Timeline.h"

// In your app class:
std::unique_ptr<timeline::Timeline> mTimeline;
std::unordered_map<std::string, scene::ShapeId> _shapeNameToId;

// In setup(), after buildScene():
mTimeline = std::make_unique<timeline::Timeline>();
auto tlPath = app::getAssetPath("timeline.json");
if (fs::exists(tlPath)) {
    mTimeline->load(tlPath);
    wireTimelineCallbacks();
}

// In update(), before Pipeline::beginFrame:
if (mTimeline)
    mTimeline->evaluate(mDt);

// In draw(), after the main ImGui window:
if (mTimeline) {
    if (mTimeline->drawUi())
        wireTimelineCallbacks();   // re-wire if user added/removed events
    mTimeline->drawTimeline(mTimeline->currentTime(), 500.f, 1280.f);
}
```

`evaluate(dt)` advances `_currentTime` by `dt` (skipped when paused or `_updateEnabled=false`), loops at the computed `_duration`, and calls `evt->apply(time)` for every event — which in turn invokes `evt->onUpdate(progress)` if `progress > 0`.

---

## Trigger Types

| Class | What it produces | Typical use |
|-------|------------------|-------------|
| `EventTrigger` (base) | `progress` float in `[0,1]` | Generic pulse — visibility, scene mode switches |
| `CurveTrigger` | `evaluateValue(time)` mapped through an `EasingCurve` (cubic bezier) + `valueStart`/`valueEnd` | Float params: scale, intensity, roughness |
| `ColorTrigger` | `evaluateColor(time)` from a `Gradient` (multi-mark RGBA) | Material albedo / emission color |
| `PathTrigger` | `evaluatePosition(time)` from a `ci::BSpline<3>` | Shape transform translation along a curve |

All four are serializable — `typeName()` returns `"base"`/`"curve"`/`"color"`/`"path"`, which is what the `"evt"` field in `timeline.json` stores. The factory `createEvent(type)` rehydrates the right subclass.

### Trigger Modes

`TriggerMode` controls how `evaluate()` maps `time → progress`:

| Mode | Behavior |
|------|----------|
| `Pulse` | `progress = triggerValue` once `time >= startTime`, then latched (`_triggered=true`) |
| `Repeat` | Pulses every `duration` seconds between `startTime` and `endTime` |
| `Duration` | `progress = triggerValue` while `startTime <= time <= startTime + duration` |

---

## Wiring Callbacks

The integration layer (your app) is responsible for binding each event's `onUpdate` to a pipeline mutation. `targetName` is the string → `ShapeId` bridge: populate `_shapeNameToId` while building the scene, then look up at wire time.

```cpp
void NewTypeEngine::wireTimelineCallbacks() {
    if (!mTimeline) return;

    for (auto& evt : mTimeline->events()) {
        auto shapeIt = _shapeNameToId.find(evt->targetName);
        auto shapeId = (shapeIt != _shapeNameToId.end())
            ? std::optional<scene::ShapeId>(shapeIt->second)
            : std::nullopt;

        // === Path: drive shape translation ===
        if (auto* pathEvt = dynamic_cast<timeline::PathTrigger*>(evt.get())) {
            if (shapeId) {
                evt->onUpdate = [this, id = *shapeId, pathEvt](float) {
                    auto pos = pathEvt->evaluatePosition(mTimeline->currentTime());
                    auto mat = luisa::make_float4x4(
                        luisa::translation(luisa::make_float3(pos.x, pos.y, pos.z)));
                    mPipeline->setShapeTransform(id, mat);
                };
            }
        }
        // === Color: drive material albedo or emission ===
        else if (auto* colorEvt = dynamic_cast<timeline::ColorTrigger*>(evt.get())) {
            if (shapeId) {
                evt->onUpdate = [this, id = *shapeId, colorEvt](float) {
                    auto color = colorEvt->evaluateColor(mTimeline->currentTime());
                    uint matIdx = mPipeline->getShape(id)->material_layers() & 0xFFu;
                    auto matData = mPipeline->material()->getMaterial(matIdx).data;
                    matData.albedo = luisa::make_float4(color.r, color.g, color.b, color.a);
                    mPipeline->material()->updateMaterialData(matIdx, matData);
                };
            }
        }
        // === Curve: drive a float param ===
        else if (auto* curveEvt = dynamic_cast<timeline::CurveTrigger*>(evt.get())) {
            if (shapeId) {
                evt->onUpdate = [this, id = *shapeId, curveEvt](float) {
                    float val = curveEvt->evaluateValue(mTimeline->currentTime());
                    // E.g., visibility toggle at threshold
                    mPipeline->setShapeVisibility(id, val > 0.5f);
                };
            }
        }
        // === Base: no shape target — handle by event name ===
        else if (auto* baseEvt = dynamic_cast<timeline::EventTrigger*>(evt.get())) {
            if (evt->name == "visible") {
                evt->onUpdate = [this, baseEvt](float prog) {
                    // custom global state change
                };
            }
        }
    }
}
```

### Pattern: Driving a Non-Shape Parameter

For events that target app state instead of a shape (e.g., `_shapeNameToId` lookup fails), branch on `evt->name` and read the value directly. The demo uses this for things like card-blend curves and global visibility toggles — see `NewTypeEngine.cpp::wireTimelineCallbacks()` for the full set.

### Pattern: Re-Wire on UI Edits

`Timeline::drawUi()` returns `true` when the user clicks "Require Update". Treat that as a signal to re-wire callbacks (e.g., new events were added or targets renamed):

```cpp
if (mTimeline) {
    if (mTimeline->drawUi())
        wireTimelineCallbacks();
}
```

---

## Constructing a Timeline in Code

```cpp
using namespace newtype::timeline;

auto tl = std::make_unique<Timeline>();

// 1) Curve event: drive emissive intensity on "mainLight" from 0 → 5 over 2s
auto curveEvt = std::make_shared<CurveTrigger>();
curveEvt->name        = "lightIntensity";
curveEvt->targetName  = "mainLight";
curveEvt->mode        = TriggerMode::Duration;
curveEvt->startTime   = 1.0f;
curveEvt->duration    = 2.0f;
curveEvt->valueStart  = 0.0f;
curveEvt->valueEnd    = 5.0f;
// EasingCurve defaults to a smooth ease-in-out; tweak via its ImGui widget
tl->addEvent(curveEvt);

// 2) Color event: fade "floor" albedo from white to red
auto colorEvt = std::make_shared<ColorTrigger>();
colorEvt->name       = "albedo";
colorEvt->targetName = "floor";
colorEvt->startTime  = 0.f;
colorEvt->duration   = 3.f;
colorEvt->mode       = TriggerMode::Duration;
colorEvt->gradient.addMark(0.0f, ImColor(1.f, 1.f, 1.f, 1.f));
colorEvt->gradient.addMark(1.0f, ImColor(1.f, 0.f, 0.f, 1.f));
colorEvt->gradient.refreshCache();
tl->addEvent(colorEvt);

// 3) Path event: orbit "orbitingSphere" along a 4-point spline
std::vector<ci::vec3> waypoints = {
    { 0.f, 1.f,  3.f }, {  3.f, 1.f, 0.f },
    { 0.f, 1.f, -3.f }, { -3.f, 1.f, 0.f }
};
auto pathEvt = std::make_shared<PathTrigger>();
pathEvt->name       = "orbit";
pathEvt->targetName = "orbitingSphere";
pathEvt->startTime  = 0.f;
pathEvt->duration   = 6.f;
pathEvt->mode       = TriggerMode::Duration;
pathEvt->path.setPathData(waypoints);
tl->addEvent(pathEvt);

tl->play();
mTimeline = std::move(tl);
wireTimelineCallbacks();
```

---

## Serialization Format

`Timeline::save(path)` writes `toJson().dump(4)`. Each event is an object:

```json
[
  {
    "evt": "curve",
    "name": "lightIntensity",
    "target": "mainLight",
    "mode": 2,
    "startTime": 1.0,
    "duration": 2.0,
    "valueStart": 0.0,
    "valueEnd": 5.0,
    "curve": { "points": [0.42, 0.0, 0.58, 1.0], "type": 0.0 }
  },
  {
    "evt": "color",
    "name": "albedo",
    "target": "floor",
    "mode": 2,
    "startTime": 0.0,
    "duration": 3.0,
    "gradient": {
      "marks": [
        { "color": [1.0, 1.0, 1.0, 1.0], "position": 0.0 },
        { "color": [1.0, 0.0, 0.0, 1.0], "position": 1.0 }
      ]
    }
  },
  {
    "evt": "path",
    "name": "orbit",
    "target": "orbitingSphere",
    "mode": 2,
    "startTime": 0.0,
    "duration": 6.0,
    "path": {
      "dim": 3,
      "data": [[0,1,3], [3,1,0], [0,1,-3], [-3,1,0]]
    }
  }
]
```

`mode` is the integer index into `TriggerMode` (0=Pulse, 1=Repeat, 2=Duration).

---

## UI Integration

`Timeline` exposes three draw entry points:

| Method | Where it goes | What it draws |
|--------|---------------|---------------|
| `drawUi()` | Inside your main ImGui window | Transport controls (Play/Pause/Stop/Loop), event list with per-event inspector, Add/Sort/Save/Load buttons |
| `drawTimeline(time, winHeight, maxWidth)` | Creates its own `Events` (45%) + `Timeline` (55%) windows at the bottom of the screen | Per-event strip with playhead |
| `drawEditor()` | Inside your 3D viewport render pass | Path editor for any `PathTrigger` (gated by `NT_ENABLE_TIMELINE_EDITOR`) |

`drawTimeline` positions its own windows at the bottom-left and bottom-middle of the screen. Pass the same dimensions each frame so the layout is stable:

```cpp
mTimeline->drawTimeline(mTimeline->currentTime(), /*winHeight=*/500.f, /*maxWidth=*/1280.f);
```

The strip uses `ImDrawList` directly — clicking an event in the strip selects it in the events list (and vice versa via `_scrollToSelected`).

### Path Editor

`drawEditor()` iterates events, dispatches to any `PathTrigger`'s `drawEditor()`, which renders the BSpline control points in the 3D viewport and supports click-drag editing. Only compiled in when `NT_ENABLE_TIMELINE_EDITOR=1`.

---

## Per-Frame Lifecycle

```
                 setup()
                   │
     ┌─────────────▼──────────────────┐
     │  buildScene() — populate       │
     │  _shapeNameToId map            │
     │  load("timeline.json")         │
     │  wireTimelineCallbacks()       │
     └─────────────┬──────────────────┘
                   │
              update() per frame
                   │
     ┌─────────────▼──────────────────┐
     │ Timeline::evaluate(dt)         │
     │  • advance _currentTime        │
     │  • loop on _duration           │
     │  • for each evt: evt->apply()  │
     │    • calls onUpdate(progress)  │
     │      if progress > 0           │
     │      → mutates transforms /    │
     │        material data / etc.    │
     └─────────────┬──────────────────┘
                   │
              Pipeline::beginFrame()
                   │ (GPU upload sees the mutations)
```

Always call `evaluate()` **before** `Pipeline::beginFrame()` so the GPU upload picks up the mutations from this frame's callbacks.

---

## API Reference

### Playback

| Method | Description |
|--------|-------------|
| `play()` / `pause()` | Toggle `_playing`. Paused timelines still evaluate `apply()` for events whose window includes the current time — they just don't advance `_currentTime`. |
| `stop()` | Pause + reset `_currentTime` to 0 + `reset()` all events (clears `_triggered` latches). |
| `setTime(t)` | Scrub playhead. Does not auto-reset latches; call `stop()` first if you want a clean replay. |
| `currentTime()` | Read-only accessor. |
| `isPlaying()` | True iff `play()` was last called. |

### Event Management

| Method | Description |
|--------|-------------|
| `addEvent(evt)` | Append + recompute `_duration`. |
| `removeEvent(name)` | Erase by name + recompute `_duration`. |
| `findEvent(name)` | Linear scan; returns `nullptr` if missing. |
| `events()` | Const reference to the underlying vector — use for callback wiring. |
| `sortEvents()` | Sort by `startTime` (display only; `evaluate` does not require sorting). |

### Serialization

| Method | Description |
|--------|-------------|
| `toJson()` / `fromJson(j)` | `ci::Json::array()` of events. |
| `load(path)` | Parse JSON file via Cinder's `loadFile`. No-op if missing. |
| `save(path)` | Dump with 4-space indent. |
| `reset()` | Clear `_triggered` on every event (does not remove them). |

### Easing Primitives

| Class | Description |
|-------|-------------|
| `EasingCurve` | Cubic bezier (4 control points stored as `float[4]` + `type`). `sample(ratio)` returns `vec2`. Used by `CurveTrigger` and `PathHelper` for temporal easing. |
| `Gradient` | List of `GradientMark {ColorA, position}` with a 256-entry cache. `getColorAt(pos, out)`. ImGui `GradientButton` / `GradientEditor` widgets included. |
| `PathHelper<D>` | Wraps `ci::BSpline<D,float>`. `getPosition(ratio)`, `getTransform(ratio, includeRotation=true)`, `getRotation(ratio)`. Explicit instantiation for `D=3`. |

---

## Common Pitfalls

- **Forgetting to re-wire after UI edits**: `drawUi()` returns `true` when the user clicks "Require Update". If you skip the re-wire, newly added events have empty `onUpdate` callbacks and silently do nothing.
- **Calling `evaluate()` after `beginFrame()`**: callbacks mutate `material_layers` / transforms, but GPU upload already happened. Material changes won't appear until the next frame — fine for slow curves, jarring for `Pulse` events.
- **`Pulse` events latch**: once `_triggered=true`, the event will not refire on the same playthrough. Call `stop()` (which calls `reset()`) — not `setTime(0)` — to replay from the start.
- **`_duration` is computed, not authored**: `recalcDuration()` takes the max `startTime + duration` across events, with a 0.5s floor. Editing event times in the UI automatically extends the loop length.
- **Editor builds**: `PathTrigger::drawEditor()` and `PathHelper::drawUi()` are compiled out when `NT_ENABLE_TIMELINE_EDITOR=0`. The trigger still works at runtime; you just lose the in-viewport path-editing UI.
