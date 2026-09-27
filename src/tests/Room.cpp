//==============================================================================
// Room — full interior scene (src/tests/Room.cpp)
//
// A furnished room: glass panels and chrome/gold metals, carpet, chairs, a
// wood table with a textured surface, floor/walls, plus a dielectric sphere
// — all lit by a small high-intensity warm area light (hard shadows, color
// bleeding, caustic-adjacent glass paths).
//
// The materials preamble (checker/gray/red/green/emissive/custom/layer) is
// shared with the Cornell box scene; raster-feature and procedural test
// blocks in NewTypeEngine.cpp pick these up by name via publish_material().
//==============================================================================

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
#include "newtype/render/TextureConverter.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {

class RoomScene : public nt::test::TestScene {
public:
    const char* name() const override { return "room"; }

    void build(core::Pipeline& pipeline) override {
        auto& device = core::Renderer::device();
        auto& stream = core::Renderer::stream();
        auto matPool = pipeline.material();

        //======================================================================
        // Materials
        //======================================================================
        auto whitemat = render::make_conductor();

        // Checkerboard — procedural albedo tiling (see CheckerboardResolver
        // in NewTypeEngine.cpp for the CPU-registered callable variant).
        render::MaterialData checkerMat = render::make_conductor();
        checkerMat.alphacut = 0.f;     // trigger G-Buffer alpha check path
#if RT_RUNTIME
        checkerMat.type = 14u;      // type 13 (first custom)
#endif
        uint checkerMatIdx = pipeline.addMaterial("boxc", checkerMat);

        auto grayMatIdx = pipeline.addMaterial("gray",
            render::make_diffuse(luisa::make_float3(1.f, 1.f, 1.f)));

        auto redMatIdx = pipeline.addMaterial("red",
            render::make_diffuse(luisa::make_float3(0.8f, 0.1f, 0.1f)));

        auto greenMatIdx = pipeline.addMaterial("green",
            render::make_diffuse(luisa::make_float3(0.1f, 0.8f, 0.1f)));

        auto cyanMatIdx = pipeline.addMaterial("cyan",
            render::make_emissive(luisa::make_float3(10.0f, 10.0f, 10.0f)));

        auto orangeMatIdx = pipeline.addMaterial("orange",
            render::make_emissive(luisa::make_float3(400.0f, 28.0f, 8.0f)));

        render::MaterialTextures textures;
        textures.albedo = render::TextureConverter::loadAsset("textures/streetview.png", device, &matPool->stream());// , true, true);
        render::TextureCompressionSettings compression;
        compression.enableCompression = true;  // Master toggle

        uint whiteMatIdx;
        if (external_albedo_slot() >= 0) {
            // video player / splash sim output as albedo
            whitemat.albedoTexIdx = static_cast<uint>(external_albedo_slot());
            whiteMatIdx = matPool->createMaterial("custom", whitemat);
        }
        else {
            whiteMatIdx = matPool->createMaterial("custom", whitemat, std::move(textures), compression);
        }

        auto layermat = render::make_clearcoat();
        layermat.meta = .5f;
        auto layerMatIdx = pipeline.addMaterial("layerc", layermat);

        // Material indices used by the optional experimental blocks in
        // NewTypeEngine.cpp (raster features / procedural geometry).
        publish_material("checker", checkerMatIdx);
        publish_material("red", redMatIdx);
        publish_material("green", greenMatIdx);
        publish_material("white", whiteMatIdx);

        //======================================================================
        // Room geometry
        //======================================================================
        scene::StaticTransform cubeTransform;
        {
            auto glassMatIdx = pipeline.addMaterial("glass",
                render::make_dielectric(luisa::make_float3(1.f, 1.f, 1.f))
            );
            auto glassMesh = scene::MeshShape::create(device, glassMatIdx);
            glassMesh->load_from(ObjLoader(app::loadAsset("models/room_glass.obj")));
            glassMesh->build(stream);
            pipeline.addShape(std::move(glassMesh), &cubeTransform);

            auto metalMatIdx = pipeline.addMaterial("metal",
                render::make_conductor(luisa::make_float3(237.f, 207.f, 44.f) / 255.f, .1f)
            );
            auto metalMesh = scene::MeshShape::create(device, metalMatIdx);
            metalMesh->load_from(ObjLoader(app::loadAsset("models/room_metal.obj")));
            metalMesh->build(stream);
            pipeline.addShape(std::move(metalMesh), &cubeTransform);

            auto fabricMatIdx = pipeline.addMaterial("fabric",
                render::make_plastic(luisa::make_float3(.7f, 1.f, .6f))
            );
            auto fabricMesh = scene::MeshShape::create(device, fabricMatIdx);
            fabricMesh->load_from(ObjLoader(app::loadAsset("models/room_carpet.obj")));
            fabricMesh->build(stream);
            pipeline.addShape(std::move(fabricMesh), &cubeTransform);

            auto woodMatIdx = pipeline.addMaterial("wood",
                render::make_diffuse(luisa::make_float3(.2f, 1.f, .3f))
            );
            auto chairMesh = scene::MeshShape::create(device, woodMatIdx);
            chairMesh->load_from(ObjLoader(app::loadAsset("models/room_chairs.obj")));
            chairMesh->build(stream);
            pipeline.addShape(std::move(chairMesh), &cubeTransform);

            auto cubbleMatIdx = pipeline.addMaterial("cubble",
                render::make_diffuse(luisa::make_float3(.2f, .2f, .9f))
            );
            auto cubbleMesh = scene::MeshShape::create(device, cubbleMatIdx);
            cubbleMesh->load_from(ObjLoader(app::loadAsset("models/room_cubble.obj")));
            cubbleMesh->build(stream);
            pipeline.addShape(std::move(cubbleMesh), &cubeTransform);

            auto woodMesh = scene::MeshShape::create(device, whiteMatIdx);
            woodMesh->load_from(ObjLoader(app::loadAsset("models/room_wood.obj")));
            woodMesh->build(stream);
            pipeline.addShape(std::move(woodMesh), &cubeTransform);

            auto floorMesh = scene::MeshShape::create(device, redMatIdx);
            floorMesh->load_from(ObjLoader(app::loadAsset("models/room_floor.obj")));
            floorMesh->build(stream);
            pipeline.addShape(std::move(floorMesh), &cubeTransform);

            auto wallMesh = scene::MeshShape::create(device, greenMatIdx);
            wallMesh->load_from(ObjLoader(app::loadAsset("models/room_wall.obj")));
            wallMesh->build(stream);
            pipeline.addShape(std::move(wallMesh), &cubeTransform);
        }

        {
            // contents — dielectric sphere inside the room
            auto sphMatIdx = pipeline.addMaterial("sphere",
                render::make_dielectric(luisa::make_float3(1.f, 1.f, 1.f))
            );
            auto sphMesh = scene::MeshShape::create(device, sphMatIdx);
            sphMesh->load_from(ObjLoader(app::loadAsset("models/sphere.obj")));
            sphMesh->build(stream);
            pipeline.addShape(std::move(sphMesh), &cubeTransform);
        }

        //======================================================================
        // Warm area light
        //======================================================================
        {
            TriMesh cyanLightMesh(ObjLoader(app::loadAsset("models/room_light.obj")));
            auto cyanLight = scene::make_light(device, cyanLightMesh, orangeMatIdx);

            auto cyanLightTransform = scene::StaticTransform::create(tolc(ci::mat4(1.f)));
            pipeline.addLightShape(std::move(cyanLight), cyanLightTransform.get());
        }
    }
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createRoomScene() { return std::make_unique<RoomScene>(); }
} // namespace newtype::test
