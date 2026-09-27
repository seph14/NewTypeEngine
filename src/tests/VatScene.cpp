//==============================================================================
// VatScene — VATMesh (TLAS DeformableMesh path) playback test
// (src/tests/VatScene.cpp)
//
// Renders `assets/models/northwall/NorthWall.vat` — a VAT v1 packed file
// (9 topologies, 700 frames, exported by VATExporter) — through
// scene::VATMesh, which owns one DeformableMesh per topology and swaps TLAS
// visibility as the baked animation crosses topology boundaries.
//
// Contents: a ground plane, the VAT at identity transform, and one local area
// light in front of the wall. The asset collapses toward -x; later frames
// dip below y=0 (source-animation pivot), so some debris intersects the
// ground — reposition _groundTrans if that bothers the test.
//==============================================================================

#include "cinder/Log.h"
#include "cinder/app/App.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/scene/VATMesh.h"
#include "newtype/render/Material.h"
#include "newtype/util/Camera.h"
#include "newtype/util/TypeConv.h"

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;

namespace {

namespace fs = std::filesystem;

class VatScene : public nt::test::TestScene {
public:
    const char* name() const override { return "vat"; }

    void build(core::Pipeline& pipeline) override {
        auto& device = core::Renderer::device();
        auto& stream = core::Renderer::stream();

        auto vatFolder = app::getAssetPath("models/northwall");
        if (!fs::exists(vatFolder / "NorthWall.vat")) {
            CI_LOG_E("--scene vat: NorthWall.vat not found under " << vatFolder.string());
            return;
        }

        //======================================================================
        // Materials
        //======================================================================
        auto wallMatIdx = pipeline.addMaterial("vat_wall",
            render::make_conductor());// luisa::make_float3(0.65f, 0.62f, 0.58f)));
        auto groundMatIdx = pipeline.addMaterial("vat_ground",
            render::make_diffuse(luisa::make_float3(0.4f)));
        _litMatIdx = pipeline.addMaterial("vat_light",
            render::make_emissive(luisa::make_float3(120.f)));

        //======================================================================
        // Ground (top surface at y=0, under the wall's base)
        //======================================================================
        {
            auto groundMesh = scene::MeshShape::create(device, groundMatIdx);
            groundMesh->load_from(geom::Cube().size(vec3(160.f, .1f, 160.f)));
            groundMesh->build(stream);
            _groundTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(-8.f, -.05f, 0.f))));
            _groundId = pipeline.addShape(std::move(groundMesh), _groundTrans.get());
        }

        //======================================================================
        // VAT — loads the packed <base>.vat via load_folder's V1 probe.
        // double-sided must be set before build(); update() only advances
        // frames while playing.
        //======================================================================
        _vat = scene::VATMesh::create(device);
        if (!_vat->load_folder(vatFolder, "NorthWall")) {
            CI_LOG_E("--scene vat: failed to load NorthWall.vat");
            _vat.reset();
            return;
        }
        //_vat->set_double_sided(true);
        _vat->set_play_mode(scene::VATPlayMode::Loop);
        _vat->build(pipeline, stream, wallMatIdx);
        _vat->set_speed(1.f);
        _vat->set_playing(true);

        //======================================================================
        // Local area light in front of the wall's +x face
        //======================================================================
        {
            TriMesh lightMesh = ObjLoader(app::loadAsset("models/arealight.obj"));
            auto light = scene::make_light(device, lightMesh, _litMatIdx);
            _lightTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(6.f, 5.f, 0.f)) * glm::scale(vec3(2.5f))));
            _lightId = pipeline.addLightShape(std::move(light), _lightTrans.get());
        }
    }

    // The hook contract guarantees this runs before Pipeline::update(), which
    // VATMesh::update() requires (it dispatches deformation + BLAS rebuild).
    void update(float, float dt, core::Pipeline& pipeline) override {
        if (_vat) _vat->update(pipeline, dt);
    }

    void drawUi() override {
        if (_vat) _vat->drawUi();   // playback speed / frame scrub controls
    }

    void applyCamera(newtype::util::Camera& camera) override {
        camera.ciCam().lookAt(vec3(14.f, 4.f, 10.f), vec3(-8.f, .5f, 0.f));
    }

private:
    scene::VATMeshPtr        _vat;
    scene::StaticTransPtr    _groundTrans, _lightTrans;   // polled by the pipeline — keep alive
    scene::ShapeId           _groundId, _lightId;
    uint                     _litMatIdx;
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createVatScene() { return std::make_unique<VatScene>(); }
} // namespace newtype::test
