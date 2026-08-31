//==============================================================================
// MaterialTest — 15-sphere material grid (src/tests/MaterialTest.cpp)
//
// Five rows of three spheres over a ground plane under a warm area light:
//   row 0  diffuse      row 1  conductor     row 2  subsurface
//   row 3  dielectric   row 4  fabric (sheen)
// Column 1 adds a clearcoat layer, column 2 a sheen layer (4-layer material
// system, layer weights via MaterialData::meta).
//
// ImGui "Move Spheres" slides the rows back and forth — useful for eyeballing
// temporal accumulation / ReSTIR reuse under motion.
//==============================================================================

#include "cinder/app/App.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include "cinder/CinderImGui.h"

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/Material.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {

class MaterialTestScene : public nt::test::TestScene {
public:
    const char* name() const override { return "material"; }

    void build(core::Pipeline& pipeline) override {
        auto& device  = core::Renderer::device();
        auto& stream  = core::Renderer::stream();
        auto matPool  = pipeline.material();

        {
            // warm area light panel above the grid
            auto litMatIdx = pipeline.addMaterial("lit",
                render::make_emissive(8.f * tolc(Color::hex(0xFFD5B8))));
            TriMesh whiteLightMesh = TriMesh(geom::Plane().normal(vec3(0.f, -1.f, 0.f)).size(vec2(50.f, 50.f)));
            auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

            auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, 25.f, 0.f))));
            pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
        }

        {
            scene::StaticTransform identity;
            auto groundMatIdx = pipeline.addMaterial("ground", render::make_diffuse());
            auto groundMesh = scene::MeshShape::create(device, groundMatIdx);
            groundMesh->load_from(TriMesh(geom::Plane().size(vec2(25.f, 25.f))));
            groundMesh->build(stream);
            pipeline.addShape(std::move(groundMesh), &identity);
        }

        render::MaterialTextures textures;
        textures.albedo = render::TextureConverter::loadFile(
            app::getAssetPath("textures/testsphere/albedo.jpg"), device, &matPool->stream(), true);
        textures.normal = render::TextureConverter::loadFile(
            app::getAssetPath("textures/testsphere/normal.png"), device, &matPool->stream());
        render::TextureCompressionSettings compression;
        compression.enableCompression = true;

        // One base material per row; every row shares the test-sphere textures.
        auto diffuseMat = render::make_diffuse();
        auto diffuseMatIdx = matPool->createMaterial("base_diffuse", diffuseMat, std::move(textures), compression);
        auto matdata = matPool->getMaterial(diffuseMatIdx).data;

        auto conductorMat = render::make_conductor();
        conductorMat.albedoTexIdx = matdata.albedoTexIdx;
        conductorMat.normalTexIdx = matdata.normalTexIdx;
        auto conductorMatIdx = matPool->createMaterial("base_conductor", conductorMat);

        auto subsurfaceMat = render::make_subsurface();
        subsurfaceMat.albedoTexIdx = matdata.albedoTexIdx;
        subsurfaceMat.normalTexIdx = matdata.normalTexIdx;
        auto subsurfaceMatIdx = matPool->createMaterial("base_subsurface", subsurfaceMat);

        auto glassMat = render::make_dielectric();
        glassMat.albedoTexIdx = matdata.albedoTexIdx;
        glassMat.normalTexIdx = matdata.normalTexIdx;
        auto glassMatIdx = matPool->createMaterial("base_glass", glassMat);

        auto fabricMat = render::make_fabric();
        fabricMat.albedoTexIdx = matdata.albedoTexIdx;
        fabricMat.normalTexIdx = matdata.normalTexIdx;
        auto fabricMatIdx = matPool->createMaterial("base_fabric", fabricMat);

        auto clearcoatIdx = pipeline.addMaterial("clearcoat", render::make_clearcoat());
        auto sheenIdx = pipeline.addMaterial("sheen", render::make_sheen());

        // 5 rows x 3 spheres; columns: plain / +clearcoat / +sheen
        struct Row { const char* name; uint matIdx; };
        const Row rows[] = {
            { "diffuse",    diffuseMatIdx    },
            { "conductor",  conductorMatIdx  },
            { "subsurface", subsurfaceMatIdx },
            { "glass",      glassMatIdx      },
            { "fabric",     fabricMatIdx     },
        };

        TriMesh sphere(ObjLoader(app::loadAsset("models/matsphere.obj")) >> geom::Scale(vec3(.015f)));
        int rowIdx = 0;
        for (const auto& row : rows) {
            float z = 3.f * rowIdx++ - 6.f;
            for (int i = 0; i < 3; i++) {
                auto sphTrans = scene::AnimatedTransform::create(tolc(glm::translate(vec3(3.f * i - 4.5f, 0, z))));
                auto testMesh = scene::MeshShape::create(device, row.matIdx);

                if (i == 1) testMesh->set_layer(1, clearcoatIdx);
                if (i == 2) testMesh->set_layer(1, sheenIdx);

                testMesh->load_from(sphere);
                testMesh->build(stream);
                testMesh->unloadCPUData();
                pipeline.addShape(std::move(testMesh), sphTrans.get());
                _transforms.push_back(std::move(sphTrans));
            }
        }
    }

    void update(float time, float) override {
        if (!_moveObjects) return;
        uint idx = 0;
        float ratio = glm::fract(time / 8.f);
        for (int row = 0; row < 5; row++)
            for (int i = 0; i < 3; i++)
                _transforms[idx++]->set_position(vec3(3.f * i - 4.5f, 0, 12.f * ratio + 3.f * row - 6.f));
    }

    void drawUi() override {
        ImGui::Checkbox("Move Spheres", &_moveObjects);
    }

private:
    bool _moveObjects = false;
    std::vector<nt::scene::AnimTransPtr> _transforms;  // kept alive: polled per frame
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createMaterialTestScene() { return std::make_unique<MaterialTestScene>(); }
} // namespace newtype::test
