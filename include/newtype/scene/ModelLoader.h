#pragma once

//==============================================================================
// ModelLoader — loads newtype.model JSON scenes into a Pipeline.
//
// The JSON (+ .msh/.obj geometries + engine-convention textures) is produced
// by the external FBXImporter tool (D:\Projects\FBXImporter, sibling of this
// repo); the engine itself stays free of any FBX dependency. See that tool's
// OutputWriter for the schema:
//
//   { "format": "newtype.model", "version": 1,
//     "materials":  [ { "name", "data": { engine MaterialData schema } } ],
//     "geometries": [ { "name", "file", "format", "material", counts } ],
//     "nodes":      [ { "name", "parent", TRS, "geometries": [...] } ],  // optional
//     "cameras":    [ { "name", "eye", "orient", "fov", "near", "far" } ],
//     "lights":     [ { "name", "type", "position", "direction", ... } ] }
//
// Textures bind by folder naming convention ({material}_albedo.png etc.,
// created by the tool next to the JSON) via createMaterialFromFolder.
//==============================================================================

#include <filesystem>
#include <string>
#include <vector>

#include "newtype/scene/Transform.h"   // StaticTransPtr

namespace newtype::util { class Camera; }
namespace newtype::core { class Pipeline; }

namespace newtype::scene {

class ModelLoader {
public:
    struct Result {
        bool ok = false;
        std::string error;
        uint32_t shapeCount = 0;
        uint32_t materialCount = 0;
        uint32_t nodeCount = 0;        // hierarchy mode only
        uint32_t cameraCount = 0;
        uint32_t lightCount = 0;
    };

    /// Load a model JSON into the pipeline. When the file keeps the FBX
    /// hierarchy, the node StaticTransforms are returned in `outTransforms`
    /// — the caller must own them for the pipeline's lifetime (transforms
    /// are polled every frame and links are non-owning).
    /// When `camera` is non-null, the first camera record is applied to it.
    static Result load(core::Pipeline& pipeline, const std::filesystem::path& jsonPath,
                       util::Camera* camera = nullptr,
                       std::vector<StaticTransPtr>* outTransforms = nullptr);

    /// Apply the first camera record of a model JSON to `camera`.
    static bool applyFirstCamera(const std::filesystem::path& jsonPath, util::Camera& camera);
};

} // namespace newtype::scene
