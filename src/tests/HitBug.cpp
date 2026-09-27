
#include "cinder/app/App.h"
#include "cinder/Log.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include <fstream>
#include <random>
#include <cmath>

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/Material.h"
#include "newtype/render/LightSampler.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {
    class HitBugScene : public nt::test::TestScene {
    public:
        const char* name() const override { return "hit"; }

        void build(core::Pipeline& pipeline) override {
            auto& device = core::Renderer::device();
            auto& stream = core::Renderer::stream();
            auto matPool = pipeline.material();

            /* {
                // warm area light panel above the grid
                auto litMatIdx = pipeline.addMaterial("lit2",
                    render::make_emissive(8.f * tolc(Color::hex(0xFFD5B8))));
                TriMesh whiteLightMesh = TriMesh(geom::Plane().normal(vec3(-.707f, -.707f, 0.f)).size(vec2(10.f, 50.f)));
                auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

                auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(10.f, 10.f, 0.f))));
                auto sideLit = pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
                pipeline.setShapeCameraVisibility(sideLit, false);
            }*/

            {
                scene::StaticTransform identity;
                auto groundMatIdx = pipeline.addMaterial("ground", render::make_diffuse());
                auto groundMesh = scene::MeshShape::create(device, groundMatIdx);
                groundMesh->load_from(TriMesh(geom::Plane().size(vec2(25.f, 25.f))));
                groundMesh->build(stream);
                pipeline.addShape(std::move(groundMesh), &identity);
            }

            // High-frequency checker backdrop (64x32 cells of 8 texels).
            constexpr int kCheckerW = 512, kCheckerH = 256, kCheckerCell = 8;
            ci::Surface8u checker(kCheckerW, kCheckerH, true);
            {
                auto iter = checker.getIter();
                while (iter.line()) {
                    while (iter.pixel()) {
                        bool cell = (((iter.x() / kCheckerCell) + (iter.y() / kCheckerCell)) & 1) != 0;
                        iter.r() = cell ? 235 : 20;
                        iter.g() = cell ? 232 : 20;
                        iter.b() = cell ? 225 : 20;
                        iter.a() = 255;
                    }
                }
            }
            render::MaterialTextures checkerTextures;
            render::TextureCompressionSettings compression;
            compression.enableCompression = true;
            checkerTextures.albedo = render::TextureConverter::createTexture(
                checker, device, &matPool->stream(), /*srgbToLinear=*/true);
            auto checkerMatIdx = matPool->createMaterial("checker_bg",
                render::make_diffuse(), std::move(checkerTextures), compression);

            auto backdropMesh = scene::MeshShape::create(device, checkerMatIdx);
            backdropMesh->load_from(TriMesh(geom::Plane().size(vec2(36.f, 18.f))));
            backdropMesh->build(stream);
            auto backdropTrans = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, 7.f, -16.f))));
            pipeline.addShape(std::move(backdropMesh), backdropTrans.get());
        }

    private:
        
    };

} // anonymous namespace

namespace newtype::test {
    ScenePtr createHitBugScene() { return std::make_unique<HitBugScene>(); }
} // namespace newtype::test
