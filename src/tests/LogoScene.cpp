//==============================================================================
// Logo Scene 

#include "cinder/app/App.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/Material.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/MetalData.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {

    class LogoScene : public nt::test::TestScene {
    public:
        const char* name() const override { return "logo"; }

        void build(core::Pipeline& pipeline) override {
            auto& device = core::Renderer::device();
            auto& stream = core::Renderer::stream();
            auto matPool = pipeline.material();

            //======================================================================
            // Materials
            //======================================================================
            auto leftMatIdx = pipeline.addMaterial("left",
                render::make_diffuse(luisa::make_float3(1.f, 1.f, 1.f)));

            auto rightMatIdx = pipeline.addMaterial("right",
                render::make_diffuse(luisa::make_float3(0.8f, 0.1f, 0.1f)));

            auto textMatIdx = pipeline.addMaterial("text",
                render::make_conductor());

            auto coatMatIdx = pipeline.addMaterial("clearcoat",
                render::make_clearcoat());

            auto sheenMatIdx = pipeline.addMaterial("sheen",
                render::make_sheen());

            // Material indices used by the optional experimental blocks in
            // NewTypeEngine.cpp (raster features / procedural geometry).
            publish_material("left", leftMatIdx);
            publish_material("right", rightMatIdx);
            publish_material("text", textMatIdx);
            publish_material("coat", coatMatIdx);
            publish_material("sheen", sheenMatIdx);

            //======================================================================
            // logo mesh
            //======================================================================
            scene::StaticTransform cubeTransform;
            {
                auto leftMesh = scene::MeshShape::create(device, leftMatIdx);
                leftMesh->load_from(ObjLoader(app::loadAsset("models/logo_n.obj")));
                leftMesh->set_layer(1, coatMatIdx);
                leftMesh->build(stream);
                pipeline.addShape(std::move(leftMesh), &cubeTransform);

                auto rightMesh = scene::MeshShape::create(device, rightMatIdx);
                rightMesh->load_from(ObjLoader(app::loadAsset("models/logo_t.obj")));
                rightMesh->set_layer(1, sheenMatIdx);
                rightMesh->build(stream);
                pipeline.addShape(std::move(rightMesh), &cubeTransform);

                auto backMesh = scene::MeshShape::create(device, textMatIdx);
                backMesh->load_from(ObjLoader(app::loadAsset("models/logo_text.obj")));
                backMesh->build(stream);
                pipeline.addShape(std::move(backMesh), &cubeTransform);
            }

            {
                // warm area light panel above the grid
                auto litMatIdx = pipeline.addMaterial("lit",
                    render::make_emissive(8.f * tolc(Color::hex(0xFFD5B8))));
                TriMesh whiteLightMesh = TriMesh(geom::Plane().normal(vec3(0.f, -1.f, 0.f)).size(vec2(5.f, 5.f)));
                auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

                auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, 25.f, 0.f))));
                pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
            }
        }

        void update(float time, float) override {
        }
    };

} // anonymous namespace

namespace newtype::test {
    ScenePtr createLogoScene() { return std::make_unique<LogoScene>(); }
} // namespace newtype::test
