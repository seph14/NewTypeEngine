//==============================================================================
// MaterialTest — 15-sphere material grid (src/tests/MaterialTest.cpp)
//
// Five rows of three spheres over a ground plane under a warm area light:
//   row 0  diffuse      row 1  conductor     row 2  subsurface
//   row 3  dielectric   row 4  fabric (sheen)
// Column 1 adds a clearcoat layer, column 2 a sheen layer (4-layer material
// system, layer weights via MaterialData::meta).
//
// Static validation rows around the grid: Belcour-Barla iridescence
// (z = +9, docs/iridescence.md), rough glass + checker backdrop
// (z = -9, docs/sharc_rough_glass_plan.md), dispersion (z = -13,
// docs/dispersion.md), and the Plan E coat-IOR / coat-normal / modifier
// composition row (z = +12, docs/wavelength_and_material_gap_report.md §6).
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
#include "newtype/render/MetalData.h"
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
            auto lightTop = pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
            pipeline.setShapeCameraVisibility(lightTop, false);
        }
        
        {
            // warm area light panel above the grid
            auto litMatIdx = pipeline.addMaterial("lit2",
                render::make_emissive(8.f * tolc(Color::hex(0xFFD5B8))));
            TriMesh whiteLightMesh = TriMesh(geom::Plane().normal(vec3(-.707f, -.707f, 0.f)).size(vec2(10.f, 50.f)));
            auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

            auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(10.f, 10.f, 0.f))));
            auto sideLit = pipeline.addLightShape(std::move(whiteLight), whiteLightTransform.get());
            pipeline.setShapeCameraVisibility(sideLit, false);
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
        textures.albedo = render::TextureConverter::loadAsset("textures/testsphere/albedo.jpg", device, &matPool->stream(), true, true);
        textures.normal = render::TextureConverter::loadAsset("textures/testsphere/normal.png", device, &matPool->stream());
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

        //----------------------------------------------------------------------
        // Belcour-Barla iridescence validation row (docs/iridescence.md):
        //   1. gold, no film          — control
        //   2. gold, 400nm film       — vs control: warm-shifted interference
        //   3. silver, 400nm film     — vs #2: DIFFERENT colors on the same film
        //                                (substrate-aware R23, the bug fix)
        //   4. copper, 400nm film     — third substrate
        //   5. gold, 800nm film       — thickness sensitivity (denser hue cycles)
        //   6. silver + radial-gradient thickness map [100..700nm] — soap-bubble
        // All static, in front of the grid (z = +9).
        //----------------------------------------------------------------------
        {
            auto iridMat = [](render::MetalPreset preset, float thicknessNm) {
                auto m = render::make_conductor_metal(preset, 0.15f);
                m.iridescence            = 1.f;
                m.iridescence_ior        = 1.3f;
                m.iridescence_thickness  = thicknessNm;
                return m;
            };

            struct IridRow { const char* name; render::MetalPreset preset; float thicknessNm; };
            const IridRow iridRow[] = {
                { "irid_gold_ctrl", render::MetalPreset::Gold,   0.f   },
                { "irid_gold_400",  render::MetalPreset::Gold,   400.f },
                { "irid_silver_400",render::MetalPreset::Silver, 400.f },
                { "irid_copper_400",render::MetalPreset::Copper, 400.f },
                { "irid_gold_800",  render::MetalPreset::Gold,   800.f },
            };
            for (const auto& r : iridRow) {
                uint matIdx = pipeline.addMaterial(r.name, iridMat(r.preset, r.thicknessNm));
                auto iridMesh = scene::MeshShape::create(device, matIdx);
                iridMesh->load_from(sphere);
                iridMesh->build(stream);
                iridMesh->unloadCPUData();
                auto iridTrans = scene::StaticTransform::create(
                    tolc(glm::translate(vec3(2.4f * (&r - iridRow) - 6.f, 0.f, 9.f))));
                pipeline.addShape(std::move(iridMesh), iridTrans.get());
            }

            // Soap bubble: radial thickness gradient in the G channel (data, not
            // color — srgbToLinear=false keeps the ramp linear), full factor in R.
            constexpr int kIridTexSize = 256;
            ci::Surface8u iridTex(kIridTexSize, kIridTexSize, true);
            {
                auto iter = iridTex.getIter();
                while (iter.line()) {
                    while (iter.pixel()) {
                        float dx = (iter.x() + 0.5f) / kIridTexSize * 2.f - 1.f;
                        float dy = (iter.y() + 0.5f) / kIridTexSize * 2.f - 1.f;
                        float rad = glm::clamp(glm::length(vec2(dx, dy)), 0.f, 1.f);
                        iter.r() = 255;
                        iter.g() = static_cast<uint8_t>(rad * 255.f + 0.5f);
                        iter.b() = 255;
                        iter.a() = 255;
                    }
                }
            }
            render::MaterialTextures iridTextures;
            iridTextures.iridescence = render::TextureConverter::createTexture(
                iridTex, device, &matPool->stream(), /*srgbToLinear=*/false);
            auto bubbleMat = iridMat(render::MetalPreset::Silver, 100.f);
            bubbleMat.iridescence_thickness_max = 700.f;
            auto bubbleIdx = matPool->createMaterial(
                "irid_bubble", bubbleMat, std::move(iridTextures), compression);

            auto bubbleMesh = scene::MeshShape::create(device, bubbleIdx);
            bubbleMesh->load_from(sphere);
            bubbleMesh->build(stream);
            bubbleMesh->unloadCPUData();
            auto bubbleTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(2.4f * 5 - 6.f, 0.f, 9.f))));
            pipeline.addShape(std::move(bubbleMesh), bubbleTrans.get());
        }

        //----------------------------------------------------------------------
        // Plan E validation row (docs/wavelength_and_material_gap_report.md §6
        // Plan E): coat IOR parameter, coat normal map, and modifier-layer
        // composition — static, in front of the iridescence row (z = +12):
        //   1. diffuse + coat IOR 1.5 — control (Disney default)
        //   2. diffuse + coat IOR 2.4 — vs #1: visibly stronger coat Fresnel
        //      (R0 0.04 → 0.17)
        //   3. gold + coat, no map   — control for #4
        //   4. gold + coat with bumpy coat-normal map — vs #3: orange-peel
        //      sparkle (the coat reflects about its own normal)
        //   5. gold + Anisotropy layer  — vs #3: stretched highlight
        //   6. gold + Iridescence layer — vs #3: 400nm interference tint
        // #5/#6 were silently dropped before Plan E composed modifier layers
        // into the base (resolve_surface_layered §2b).
        //----------------------------------------------------------------------
        {
            // Bumpy tangent-space coat normal: sine-sum height field, slopes
            // encoded in RG (perturb_normal reconstructs Z; B channel unused).
            constexpr int kCoatTexSize = 256;
            ci::Surface8u coatNrm(kCoatTexSize, kCoatTexSize, true);
            {
                auto iter = coatNrm.getIter();
                while (iter.line()) {
                    while (iter.pixel()) {
                        float u = (iter.x() + 0.5f) / kCoatTexSize;
                        float v = (iter.y() + 0.5f) / kCoatTexSize;
                        float e = 1.f / kCoatTexSize;
                        auto h = [&](float uu, float vv) {
                            return 0.5f * sin(uu * 6.2831853f * 9.f + 1.7f)
                                 + 0.3f * sin(vv * 6.2831853f * 13.f)
                                 + 0.2f * sin((uu + vv) * 6.2831853f * 5.f);
                        };
                        float dhdu = (h(u + e, v) - h(u - e, v)) / (2.f * e);
                        float dhdv = (h(u, v + e) - h(u, v - e)) / (2.f * e);
                        constexpr float kSlope = 4.f;   // bump strength
                        auto n = glm::normalize(vec3(-dhdu * kSlope, -dhdv * kSlope, 1.f));
                        iter.r() = static_cast<uint8_t>(n.x * 127.5f + 128.f);
                        iter.g() = static_cast<uint8_t>(n.y * 127.5f + 128.f);
                        iter.b() = 255;
                        iter.a() = 255;
                    }
                }
            }
            render::MaterialTextures coatTextures;
            coatTextures.normal = render::TextureConverter::createTexture(
                coatNrm, device, &matPool->stream(), /*srgbToLinear=*/false);
            uint bumpyCoatIdx = matPool->createMaterial(
                "coat_bumpy_normal", render::make_clearcoat(1.f, 0.4f),
                std::move(coatTextures), compression);

            uint goldIdx = pipeline.addMaterial("plan_e_gold",
                render::make_conductor_metal(render::MetalPreset::Gold, 0.25f));

            struct PlanE { const char* name; uint baseIdx; uint layerIdx; };
            const PlanE planERow[] = {
                { "coat_ior_ctrl",   diffuseMatIdx, pipeline.addMaterial("coat_ior_15_layer",  render::make_clearcoat(1.f, 0.8f, -1, 1.5f)) },
                { "coat_ior_24",     diffuseMatIdx, pipeline.addMaterial("coat_ior_24_layer",  render::make_clearcoat(1.f, 0.8f, -1, 2.4f)) },
                { "coat_nrm_ctrl",   goldIdx,       pipeline.addMaterial("coat_nrm_ctrl_layer", render::make_clearcoat(1.f, 0.4f)) },
                { "coat_nrm_bumpy",  goldIdx,       bumpyCoatIdx },
                { "mod_aniso",       goldIdx,       pipeline.addMaterial("mod_aniso_layer",   render::make_anisotropy(1.f, 0.f)) },
                { "mod_irid",        goldIdx,       pipeline.addMaterial("mod_irid_layer",    render::make_iridescence(1.f, 1.3f, 400.f)) },
            };
            for (const auto& r : planERow) {
                auto mesh = scene::MeshShape::create(device, r.baseIdx);
                mesh->set_layer(1, r.layerIdx);
                mesh->load_from(sphere);
                mesh->build(stream);
                mesh->unloadCPUData();
                auto trans = scene::StaticTransform::create(
                    tolc(glm::translate(vec3(2.4f * (&r - planERow) - 6.f, 0.f, 12.f))));
                pipeline.addShape(std::move(mesh), trans.get());
            }
        }

        //----------------------------------------------------------------------
        // Phase-3 rough-glass validation row (docs/sharc_rough_glass_plan.md
        // §7): rough dielectric spheres (0.05 / 0.2 / 0.5) plus one rough thin
        // dielectric, in front of a high-frequency checker backdrop — the
        // canonical frosted-glass A/B target. Expect: 0.05 ~ smooth (gate
        // rejects, denoised fallback), 0.2 mild, 0.5 clearly frosted; blur
        // grows with background distance (distance-adaptive SHARC footprint
        // gate). NOTE: with the default SHARC sceneScale (10) the backdrop's
        // voxels are coarse (~2.5 units at this camera distance) — raise
        // SceneScale in the SHARC panel (~50–100) to bring the 0.2/0.5 lobes
        // over the gate for the blur to engage.
        //----------------------------------------------------------------------
        {
            scene::StaticTransform identity;

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
            checkerTextures.albedo = render::TextureConverter::createTexture(
                checker, device, &matPool->stream(), /*srgbToLinear=*/true);
            //checkerTextures.albedo = render::TextureConverter::loadAsset("textures/testsphere/albedo.jpg", device, &matPool->stream(), true, true);
            auto checkerMatIdx = matPool->createMaterial("checker_bg",
                render::make_diffuse(), std::move(checkerTextures), compression);

            auto backdropMesh = scene::MeshShape::create(device, checkerMatIdx);
            auto mesh = TriMesh(geom::Plane().size(vec2(36.f, 18.f)));
            backdropMesh->load_from(mesh);// TriMesh(geom::Cube().size(vec3(12.f))));
            backdropMesh->build(stream);
            auto backdropTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(0.f, 7.f, -16.f))));
            pipeline.addShape(std::move(backdropMesh), backdropTrans.get());

            // Rough-dielectric row (static — the moving-rows test above stays
            // a smooth-glass/PSR exercise).
            struct RoughGlass { float roughness; bool thin; };
            const RoughGlass roughRow[] = {
                { 0.05f, false }, { 0.2f, false }, { 0.5f, false }, { 0.35f, true },
            };
            for (int i = 0; i < 4; i++) {
                luisa::float3 white{1.f};
                auto mat = roughRow[i].thin
                    ? render::make_thin_dielectric(white, 1.5f, roughRow[i].roughness)
                    : render::make_dielectric(white, 1.5f, roughRow[i].roughness);
                uint matIdx = pipeline.addMaterial(
                    roughRow[i].thin ? "rough_thin_glass" : "rough_glass", mat);
                auto roughMesh = scene::MeshShape::create(device, matIdx);
                roughMesh->load_from(sphere);
                roughMesh->build(stream);
                roughMesh->unloadCPUData();
                auto roughTrans = scene::StaticTransform::create(
                    tolc(glm::translate(vec3(3.6f * i - 5.4f, 0.f, -9.f))));
                pipeline.addShape(std::move(roughMesh), roughTrans.get());
            }

            //----------------------------------------------------------------------
            // Dispersion validation row (docs/dispersion.md): smooth dispersive
            // glass spheres (Abbe V = off / 26 / 16 / 8) directly in front of
            // the checker backdrop. Post-Phase-3 (2026-09-06) expectation:
            // V=0 = plain glass control; the dispersive spheres show spectral
            // fringes via the tint-pass replay (exact per-channel estimator —
            // stronger/cleaner than the old PSR channel-mean, which speckled),
            // speckle-free and temporally stable at every V (the speckle repro
            // was V=5 at roughness 0 and 0.19). Debug Tag "Dispersion Rings"
            // shows the per-channel first-interface TIR bands. Rough +
            // dispersive glass fringes via the gather taps (stronger with
            // lower V).
            // V is physical (KHR spec closed form — n_F−n_C = (n_d−1)/V), so
            // 26/16/8 sit in gem/flint territory; crown glass is ~50-65.
            //----------------------------------------------------------------------
            {
                struct DispGlass { float abbe; };
                const DispGlass dispRow[] = { { 0.f }, { 26.f }, { 16.f }, { 8.f } };
                for (int i = 0; i < 4; i++) {
                    auto mat = render::make_dielectric(
                        luisa::float3{1.f}, 1.6f, 0.f, -1, -1, 0.f, dispRow[i].abbe);
                    uint matIdx = pipeline.addMaterial("disp_glass", mat);
                    auto dispMesh = scene::MeshShape::create(device, matIdx);
                    dispMesh->load_from(sphere);
                    dispMesh->build(stream);
                    dispMesh->unloadCPUData();
                    auto dispTrans = scene::StaticTransform::create(
                        tolc(glm::translate(vec3(3.0f * i - 4.5f, 0.f, -13.f))));
                    pipeline.addShape(std::move(dispMesh), dispTrans.get());
                }
            }

            //----------------------------------------------------------------------
            // Glass-blend validation (docs/glass_blend_plan.md): the DLL
            // glass_blend_resolver (tag 18) blends glass <-> amber diffuse by
            // world x with a dithered band across [-0.45, -0.35]. Middle
            // sphere sits centered on the band (x=-0.4): left sphere pure
            // glass, middle dithered midband, right pure diffuse. Expect a
            // STATIC dither pattern (camera still) with refraction left of
            // the band and lit diffuse right of it.
            //----------------------------------------------------------------------
#if RT_RUNTIME
            {
                render::MaterialData blendMat = render::make_dielectric(
                    luisa::float3{1.f}, 1.5f, 0.f);
                blendMat.type               = 18u;  // DLL glass_blend_resolver
                blendMat.bsdf_type_override = 3.f;
                blendMat.roughness          = 0.f;
                uint blendIdx = pipeline.addMaterial("glass_blend", blendMat);
                for (int i = 0; i < 3; i++) {
                    auto blendMesh = scene::MeshShape::create(device, blendIdx);
                    blendMesh->load_from(sphere);
                    blendMesh->build(stream);
                    blendMesh->unloadCPUData();
                    auto blendTrans = scene::StaticTransform::create(
                        tolc(glm::translate(vec3(1.5f * i - 1.9f, 0.f, -11.f))));
                    pipeline.addShape(std::move(blendMesh), blendTrans.get());
                }
            }

            //----------------------------------------------------------------------
            // Per-instance variation (track B2,
            // docs/vertex-packing-instancing-plan.md §4): ONE prototype, ONE
            // material (type 19 = DLL per_instance_variation resolver), 7
            // instances. The resolver hash-tints each instance by its TLAS
            // row via s.instance_index, and the slot-0 user rows authored
            // with Pipeline::setInstanceUserData modulate tint strength /
            // roughness bias / emissive boost per instance. Expect a row of
            // distinctly-tinted spheres with ramping tint strength; instance
            // 3 (emissive boost 2.0) glows; instances 5-6 (no authored row)
            // fall back to the resolver's default 25% tint.
            //----------------------------------------------------------------------
            {
                render::MaterialData varMat = render::make_diffuse(luisa::float3{0.8f}, 0.35f);
                varMat.type = 19u;  // DLL per_instance_variation resolver
                uint varIdx = pipeline.addMaterial("per_instance_variation", varMat);

                auto protoMesh = scene::MeshShape::create(device, varIdx);
                protoMesh->load_from(sphere);
                protoMesh->build(stream);
                auto proto = pipeline.addPrototype(std::move(protoMesh));

                for (int i = 0; i < 7; i++) {
                    auto inst = scene::StaticTransform::create(tolc(
                        glm::translate(vec3(1.1f * i - 3.3f, -1.6f, -11.f)) *
                        glm::scale(vec3(0.35f))));
                    auto id = pipeline.addPrototypeInstance(proto, inst.get());
                    if (i < 5) {
                        // Authored row: tint strength / roughness bias / emissive boost.
                        pipeline.setInstanceUserData(id, 0u, luisa::float4{
                            0.25f + 0.15f * i,    // tint strength ramps up
                            0.05f * i,            // roughness bias
                            i == 3 ? 2.0f : 0.f,  // one glowing instance
                            0.f});
                    }
                }
            }
#endif
        }
    }

    void update(float time, float, core::Pipeline& pipeline) override {
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
