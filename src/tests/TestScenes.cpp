//==============================================================================
// TestScenes.cpp — scene registry for the built-in sample scenes
//==============================================================================

#include "TestScenes.h"

namespace newtype::test {

// Defined at the bottom of each scene .cpp
ScenePtr createCornellBoxScene();
ScenePtr createMaterialTestScene();
ScenePtr createRoomScene();
ScenePtr createLogoScene();

ScenePtr createScene(const std::string& name) {
    if (name == "cornell")  return createCornellBoxScene();
    if (name == "material") return createMaterialTestScene();
    if (name == "room")     return createRoomScene();
    if (name == "logo")     return createLogoScene();
    return nullptr;
}

} // namespace newtype::test
