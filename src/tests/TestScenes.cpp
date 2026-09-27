//==============================================================================
// TestScenes.cpp — scene registry for the built-in sample scenes
//==============================================================================

#include "TestScenes.h"

namespace newtype::test {

// Defined at the bottom of each scene .cpp
ScenePtr createCornellBoxScene();
ScenePtr createMaterialTestScene();
ScenePtr createRoomScene();
ScenePtr createTransformTreeScene();
ScenePtr createAliasTableScene();
ScenePtr createHitBugScene();
ScenePtr createFbxScene();
ScenePtr createVatScene();
ScenePtr createTetCageScene();

ScenePtr createScene(const std::string& name) {
    if (name == "cornell")  return createCornellBoxScene();
    if (name == "material") return createMaterialTestScene();
    if (name == "room")     return createRoomScene();
    if (name == "xform")    return createTransformTreeScene();
    if (name == "alias")    return createAliasTableScene();
    if (name == "hit")    return createHitBugScene();
    if (name == "fbx")     return createFbxScene();
    if (name == "vat")     return createVatScene();
    if (name == "tetcage") return createTetCageScene();
    return nullptr;
}

} // namespace newtype::test
