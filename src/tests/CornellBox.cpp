//==============================================================================
// CornellBox — classic Cornell box with animated contents
// (src/tests/CornellBox.cpp)
//
// Red/green walls and a gray back panel lit by a warm ceiling area light.
// Contents: a clearcoat-layered checker cube, a gold sphere, a glass cube and
// an orbiting gold occluder — the animated objects exercise ReSTIR temporal
// reuse, motion vectors and glass refraction under motion.
//
// The materials preamble (checker/gray/red/green/emissive/custom/layer) is
// shared with the room scene; raster-feature and procedural test blocks in
// NewTypeEngine.cpp pick these up by name via publish_material().
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
#include "newtype/render/MetalData.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

// 1 = box walls + light only (pure Cornell box)
#define RASTER_TEST 0

namespace {

class CornellBoxScene : public nt::test::TestScene {
public:
    const char* name() const override { return "cornell"; }

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

        _litMatIdx = pipeline.addMaterial("cyan",
            render::make_emissive(luisa::make_float3(10.0f, 10.0f, 10.0f)));

        auto orangeMatIdx = pipeline.addMaterial("orange",
            render::make_emissive(luisa::make_float3(400.0f, 28.0f, 8.0f)));

        render::MaterialTextures textures;
        textures.albedo = render::TextureConverter::loadAsset("textures/streetview.png", device);// , & matPool->stream());// , true, true);
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
        // Cornell box walls
        //======================================================================
        scene::StaticTransform cubeTransform;
        {
            auto leftMesh = scene::MeshShape::create(device, redMatIdx);
            leftMesh->load_from(ObjLoader(app::loadAsset("models/left.obj")));
            leftMesh->build(stream);
            pipeline.addShape(std::move(leftMesh), &cubeTransform);

            auto rightMesh = scene::MeshShape::create(device, greenMatIdx);
            rightMesh->load_from(ObjLoader(app::loadAsset("models/right.obj")));
            rightMesh->build(stream);
            pipeline.addShape(std::move(rightMesh), &cubeTransform);

            auto backMesh = scene::MeshShape::create(device, grayMatIdx);
            backMesh->load_from(ObjLoader(app::loadAsset("models/box.obj")));
            backMesh->build(stream);
            pipeline.addShape(std::move(backMesh), &cubeTransform);
        }

#if !RASTER_TEST
        //======================================================================
        // Contents
        //======================================================================
        auto sphMatIdx = pipeline.addMaterial("sphere_conductor",
            render::make_conductor_metal(render::MetalPreset::Gold)
        );

#if RT_RUNTIME
        // Glass-blend verification (docs/glass_blend_plan.md): the center
        // sphere uses the DLL glass_blend_resolver (5th registration -> tag
        // 18; runtime_shaders/CustomMaterialShader). bsdf_type_override = 3
        // routes it through the PSR glass branch; the resolver's world-x band
        // blends glass (left) <-> amber diffuse (right) with a static-IGN
        // dithered midband. Expect: refraction through the left half, lit
        // diffuse on the right, stable dither midband when the camera is
        // still, shadows blocked behind the diffuse half and attenuated
        // behind the glass half.
        render::MaterialData blendMat = render::make_dielectric(
            luisa::make_float3(1.f), 1.5f, 0.f);
        blendMat.type               = 18u;  // DLL glass_blend_resolver
        blendMat.bsdf_type_override = 3.f;  // classify as Dielectric for PSR
        blendMat.roughness          = 0.f;
        auto sphBlendIdx = pipeline.addMaterial("sphere_blend", blendMat);
#endif

        {
            scene::StaticTransform identity;
            auto sphTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(.0f, 0.f, .0f))));
            auto cubeMesh = scene::MeshShape::create(device, whiteMatIdx);
            cubeMesh->set_layer(1, layerMatIdx);
            cubeMesh->load_from(ObjLoader(app::loadAsset("models/obj1.obj")));
            cubeMesh->build(stream);
            pipeline.addShape(std::move(cubeMesh), &identity);

#if RT_RUNTIME
            auto sphMesh = scene::MeshShape::create(device, sphBlendIdx);
#else
            auto sphMesh = scene::MeshShape::create(device, sphMatIdx);
#endif
            sphMesh->load_from(ObjLoader(app::loadAsset("models/sphere.obj")));
            sphMesh->build(stream);
            pipeline.addShape(std::move(sphMesh), sphTransform.get());
        }

        {
            auto sphGlassIdx = pipeline.addMaterial("sphere_glass",
                render::make_dielectric(luisa::make_float3(1.f, 1.f, 1.f))
            );

            auto cubeMesh = scene::MeshShape::create(device, sphGlassIdx);
            cubeMesh->load_from(geom::Cube().size(vec3(.15f)));
            cubeMesh->build(stream);
            _glassTrans = nt::scene::AnimatedTransform::create(tolc(glm::translate(vec3(-.5f, -.4f, 0.f))));
            pipeline.addShape(std::move(cubeMesh), _glassTrans.get());
        }

        {
            auto cubeMesh = scene::MeshShape::create(device, sphMatIdx);
            cubeMesh->load_from(geom::Cube().size(vec3(.35f)));
            cubeMesh->build(stream);
            _occluderTrans = nt::scene::AnimatedTransform::create(tolc(glm::translate(vec3(.0f, 1.f, 1.f))));
            pipeline.addShape(std::move(cubeMesh), _occluderTrans.get());
        }
#endif

        //======================================================================
        // Ceiling area light
        //======================================================================
        {
            TriMesh whiteLightMesh = ObjLoader(app::loadAsset("models/arealight.obj"));
            auto whiteLight = scene::make_light(device, whiteLightMesh, _litMatIdx);

            auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::scale(glm::vec3(1.5f, 1.f, 1.5f))));
            _litID = pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
            //pipeline.setShapeCameraVisibility(_litID, false);
        }
    }

    void update(float time, float, core::Pipeline& pipeline) override {
        if (_occluderTrans)
            _occluderTrans->set_position(vec3(glm::sin(time), 1.f, glm::cos(time)));

        //auto mat = render::make_emissive();
        //mat.emission = 4.f * (.5f + glm::cos(time)) * make_float3(1.f,.5f,.25f);
        //pipeline.material()->updateMaterialData(_litMatIdx, mat);
    }

private:
    // Animated transforms are polled by the pipeline every frame — they must
    // stay alive for the lifetime of the scene.
    nt::scene::AnimTransPtr _glassTrans, _occluderTrans;
    nt::scene::ShapeId _litID;
    uint _litMatIdx;
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createCornellBoxScene() { return std::make_unique<CornellBoxScene>(); }
} // namespace newtype::test
