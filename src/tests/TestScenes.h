#pragma once
//==============================================================================
// TestScenes — self-contained sample scenes for the NewTypeEngine app.
//
// Each scene is a small class that plugs materials, shapes and lights into
// the Pipeline before Pipeline::buildScene() runs. The app picks one at
// startup via the `--scene <name>` command-line argument:
//
//   NewTypeEngine.exe --scene material   material sphere grid (default)
//   NewTypeEngine.exe --scene cornell    Cornell box with animated occluder
//   NewTypeEngine.exe --scene room       full room interior (glass/metal/fabric)
//
// To add a scene: subclass TestScene, implement build() (+ optional
// update()/drawUi()), register it in createScene() below, and add the .cpp
// to the vc2022 project.
//==============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "newtype/NewType.h"  // nt alias for newtype + core engine headers

// NOTE: defined as newtype::test (not "namespace nt::test") — nt is a
// namespace *alias* (NewType.h) and can't be reopened directly.
namespace newtype::test {

class TestScene {
public:
    virtual ~TestScene() = default;

    /// Name used by `--scene <name>` on the command line.
    virtual const char* name() const = 0;

    /// Add materials, shapes and lights. Runs before Pipeline::buildScene().
    virtual void build(core::Pipeline& pipeline) = 0;

    /// Per-frame hook — runs inside the app's tick, before Pipeline::update().
    /// AnimatedTransform members mutated here are polled by the pipeline.
    virtual void update(float time, float dt) { (void)time; (void)dt; }

    /// ImGui hook — runs inside the "Engine" window.
    virtual void drawUi() {}

    /// Publish a material index for the optional experimental blocks in
    /// NewTypeEngine.cpp (NT_ALLOW_RASTER_FEATURES / NT_ENABLE_PROCEDURAL),
    /// which paint with scene materials by name.
    void publish_material(const std::string& name, uint32_t index) { _materials[name] = index; }

    /// Look up a published material index (0 if not published).
    [[nodiscard]] uint32_t material(const std::string& name) const {
        auto it = _materials.find(name);
        return it != _materials.end() ? it->second : 0u;
    }

    /// External bindless slot used as the albedo of the scene's "custom"
    /// material (video player / splash sim). Negative = load the texture file.
    void set_external_albedo_slot(int slot) { _externalAlbedoSlot = slot; }
    [[nodiscard]] int external_albedo_slot() const { return _externalAlbedoSlot; }

private:
    std::unordered_map<std::string, uint32_t> _materials;
    int _externalAlbedoSlot = -1;
};

using ScenePtr = std::unique_ptr<TestScene>;

/// Default scene when no --scene argument is given.
inline constexpr const char* kDefaultScene = "logo";

/// Factory for the built-in sample scenes: "cornell" | "material" | "room" | "logo".
/// Returns nullptr for unknown names.
ScenePtr createScene(const std::string& name);

} // namespace newtype::test
