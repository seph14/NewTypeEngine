//==============================================================================
// FbxScene — renders an FBXImporter conversion (src/tests/FbxScene.cpp)
//
// Loads `assets/fbx/model.json` produced by the external FBXImporter tool
// (D:\Projects\FBXImporter):
//
//   FBXImporter.exe --input model.fbx --out <engine>/assets/fbx
//
// (rename the resulting model.json accordingly, or name the FBX "model.fbx").
// Both conversion modes work: merge-by-material (baked transforms) and
// --hierarchy (the node transform chain is rebuilt and owned here).
// The first FBX camera record is applied via applyCamera() unless the
// artist's assets/scene.json overrides it.
//==============================================================================

#include "cinder/Log.h"
#include "cinder/app/App.h"

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/scene/ModelLoader.h"
#include "newtype/util/Camera.h"

namespace {

namespace fs = std::filesystem;

class FbxScene : public newtype::test::TestScene {
public:
    const char* name() const override { return "fbx"; }

    void build(newtype::core::Pipeline& pipeline) override {
        _modelJson = ci::app::getAssetPath("fbx/model.json");
        if (!fs::exists(_modelJson)) {
            CI_LOG_E("--scene fbx: no model at '" << _modelJson.string()
                << "'. Convert one with FBXImporter.exe --input <file>.fbx "
                "--out <engine>/assets/fbx");
            return;
        }
        newtype::scene::ModelLoader::load(pipeline, _modelJson,
            /*camera=*/nullptr, &_nodeTransforms);
    }

    void applyCamera(newtype::util::Camera& camera) override {
        if (fs::exists(_modelJson))
            newtype::scene::ModelLoader::applyFirstCamera(_modelJson, camera);
    }

private:
    fs::path _modelJson;
    std::vector<newtype::scene::StaticTransPtr> _nodeTransforms;   // owns hierarchy nodes
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createFbxScene() { return std::make_unique<FbxScene>(); }
} // namespace newtype::test
