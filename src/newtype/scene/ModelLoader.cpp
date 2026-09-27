//==============================================================================
// ModelLoader — newtype.model JSON -> Pipeline (see ModelLoader.h for schema)
//==============================================================================

#include "cinder/DataSource.h"
#include "cinder/Json.h"
#include "cinder/Log.h"
#include "cinder/ObjLoader.h"
#include "cinder/TriMesh.h"

#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/render/Material.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/ModelLoader.h"
#include "newtype/util/Camera.h"
#include "newtype/util/TypeConv.h"

namespace newtype::scene {

namespace {

// Keep in sync with FBXImporter's OutputWriter::sanitizeName — texture files
// are prefixed with the sanitized material name.
std::string sanitizeMaterialName(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (std::isalnum((unsigned char)c) || c == '_' || c == '-') out += c;
        else out += '_';
    }
    return out.empty() ? std::string("unnamed") : out;
}

ci::mat4 nodeLocalMatrix(const ci::Json& nj) {
    auto vec3of = [&](const char* key, ci::vec3 fallback) {
        if (nj.contains(key) && nj[key].is_array() && nj[key].size() >= 3)
            return ci::vec3(nj[key][0].get<float>(), nj[key][1].get<float>(), nj[key][2].get<float>());
        return fallback;
    };
    ci::vec3 t = vec3of("translation", ci::vec3(0.f));
    ci::vec3 scl = vec3of("scale", ci::vec3(1.f));
    ci::quat r;   // identity (xyzw)
    if (nj.contains("rotation") && nj["rotation"].is_array() && nj["rotation"].size() >= 4)
        r = ci::quat(nj["rotation"][3].get<float>(), nj["rotation"][0].get<float>(),
                     nj["rotation"][1].get<float>(), nj["rotation"][2].get<float>());
    return ci::translate(t) * ci::toMat4(r) * ci::scale(scl);
}

} // namespace

ModelLoader::Result ModelLoader::load(core::Pipeline& pipeline,
                                      const std::filesystem::path& jsonPath,
                                      util::Camera* camera,
                                      std::vector<StaticTransPtr>* outTransforms) {
    namespace fs = std::filesystem;
    Result result;

    ci::Json root;
    try {
        root = ci::loadJson(jsonPath);
    } catch (const std::exception& e) {
        result.error = std::string("Failed to read model JSON: ") + e.what();
        CI_LOG_E(result.error);
        return result;
    }
    const fs::path baseDir = jsonPath.parent_path();
    auto& device = core::Renderer::device();
    auto& stream = core::Renderer::stream();
    auto matPool = pipeline.material();

    //--------------------------------------------------------------------
    // Materials (+ convention-scanned textures next to the JSON)
    //--------------------------------------------------------------------
    std::vector<uint> materialIndices;   // JSON order -> pool index
    if (root.contains("materials") && root["materials"].is_array()) {
        for (const auto& mj : root["materials"]) {
            std::string name = mj.value("name", std::string("material"));
            render::MaterialData data;
            if (mj.contains("data")) render::materialDataFromJson(mj["data"], data);
            uint idx = matPool->createMaterialFromFolder(name, baseDir, data);
            materialIndices.push_back(idx);
            CI_LOG_I("ModelLoader material '" << name << "' -> pool index " << idx);
        }
    }
    result.materialCount = (uint32_t)materialIndices.size();

    //--------------------------------------------------------------------
    // Node transforms (hierarchy mode) — created here, owned by the caller
    // (non-owning parent links are polled by the pipeline every frame).
    //--------------------------------------------------------------------
    std::vector<StaticTransPtr> nodes;
    std::vector<StaticTransform*> nodePtrs;
    std::vector<const ci::Json*> nodeJsons;
    if (root.contains("nodes") && root["nodes"].is_array()) {
        for (const auto& nj : root["nodes"]) {
            auto t = StaticTransform::create(tolc(nodeLocalMatrix(nj)));
            nodePtrs.push_back(t.get());
            nodes.push_back(std::move(t));
            nodeJsons.push_back(&nj);
        }
        for (size_t i = 0; i < nodePtrs.size(); i++) {
            int parent = nodeJsons[i]->value("parent", -1);
            if (parent >= 0 && parent < (int)nodePtrs.size())
                nodes[i]->set_parent(nodePtrs[parent]);
        }
        if (!outTransforms && !nodes.empty()) {
            result.error = "hierarchy models require outTransforms (node ownership)";
            CI_LOG_E("ModelLoader: " << result.error);
            return result;
        }
    }
    result.nodeCount = (uint32_t)nodePtrs.size();

    //--------------------------------------------------------------------
    // Geometries
    //--------------------------------------------------------------------
    if (root.contains("geometries") && root["geometries"].is_array()) {
        for (const auto& gj : root["geometries"]) {
            std::string file = gj.value("file", std::string());
            if (file.empty()) continue;
            fs::path geoPath = baseDir / file;
            CI_LOG_I("ModelLoader: loading geometry '" << file << "'");

            ci::TriMesh mesh;
            try {
                if (gj.value("format", std::string("trimesh")) == "trimesh")
                    mesh.read(ci::loadFile(geoPath));
                else
                    mesh = ci::TriMesh(ci::ObjLoader(ci::loadFile(geoPath)));
            } catch (const std::exception& e) {
                CI_LOG_E("ModelLoader failed to load geometry '" << geoPath.string()
                    << "': " << e.what());
                continue;
            }
            CI_LOG_I("ModelLoader: read ok (" << mesh.getNumVertices() << " verts, "
                << mesh.getIndices().size() << " indices)");

            // material lookup by name
            std::string matName = gj.value("material", std::string());
            uint matIdx = matPool->hasMaterial(matName)
                ? matPool->getIndex(matName)
                : (materialIndices.empty() ? 0u : materialIndices.front());

            auto shape = MeshShape::create(device, matIdx);
            if (!shape->load_from(mesh)) {
                CI_LOG_E("ModelLoader: mesh '" << gj.value("name", file) << "' has no indexed geometry");
                continue;
            }
            CI_LOG_I("ModelLoader: load_from ok");
            shape->build(stream);
            CI_LOG_I("ModelLoader: build ok");

            // attach to node when present (hierarchy mode)
            StaticTransform* xform = nullptr;
            if (!nodePtrs.empty()) {
                // find the node that lists this geometry index
                size_t geoIdx = result.shapeCount;
                for (size_t n = 0; n < nodeJsons.size(); n++) {
                    if (nodeJsons[n]->contains("geometries")
                        && nodeJsons[n]->at("geometries").is_array()) {
                        for (const auto& gi : nodeJsons[n]->at("geometries"))
                            if (gi.get<int>() == (int)geoIdx)
                                xform = nodePtrs[n];
                    }
                }
            }
            if (xform)
                pipeline.addShape(std::move(shape), xform);
            else
                pipeline.addShape(std::move(shape));   // owning overload: identity
            result.shapeCount++;
        }
    }

    //--------------------------------------------------------------------
    // Cameras / lights
    //--------------------------------------------------------------------
    if (root.contains("cameras") && root["cameras"].is_array()
        && !root["cameras"].empty()) {
        result.cameraCount = (uint32_t)root["cameras"].size();
        if (camera) {
            try {
                camera->load(root["cameras"][0]);
                CI_LOG_I("ModelLoader applied camera '"
                    << root["cameras"][0].value("name", std::string()) << "'");
            } catch (const std::exception& e) {
                CI_LOG_W("ModelLoader: camera apply failed: " << e.what());
            }
        }
    }
    if (root.contains("lights") && root["lights"].is_array()) {
        result.lightCount = (uint32_t)root["lights"].size();
        // Analytic lights are not yet part of the engine's light sampler
        // (lights are emissive geometry); surface them in the log for now.
        for (const auto& lj : root["lights"])
            CI_LOG_I("ModelLoader light '" << lj.value("name", std::string("?"))
                << "' type=" << lj.value("type", std::string("?"))
                << " (analytic lights not yet supported - record only)");
    }

    CI_LOG_I("ModelLoader: " << result.shapeCount << " shapes, "
        << result.materialCount << " materials, " << result.nodeCount
        << " nodes, " << result.cameraCount << " cameras, "
        << result.lightCount << " lights from " << jsonPath.string());
    // hand node ownership to the caller last — raw pointers stay valid since
    // the vectors above are only destroyed at scope exit
    if (outTransforms)
        for (auto& n : nodes) outTransforms->push_back(std::move(n));
    result.ok = result.shapeCount > 0 || result.materialCount > 0;
    return result;
}

bool ModelLoader::applyFirstCamera(const std::filesystem::path& jsonPath, util::Camera& camera) {
    try {
        ci::Json root = ci::loadJson(jsonPath);
        if (!root.contains("cameras") || !root["cameras"].is_array()
            || root["cameras"].empty())
            return false;
        camera.load(root["cameras"][0]);
        return true;
    } catch (const std::exception& e) {
        CI_LOG_W("ModelLoader::applyFirstCamera failed: " << e.what());
        return false;
    }
}

} // namespace newtype::scene
