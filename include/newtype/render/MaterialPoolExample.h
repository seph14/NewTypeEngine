/**
 * @file MaterialPoolExample.h
 * @brief Example usage of the MaterialPool system
 *
 * This file demonstrates how to use the MaterialPool, TextureConverter,
 * and MaterialTextureLoader classes in your NewTypeEngine application.
 */

#pragma once

#include "newtype/render/MaterialPool.h"
#include "newtype/render/TextureConverter.h"
#include "cinder/app/App.h"

namespace newtype::example {

using namespace luisa;
using namespace luisa::compute;
using namespace newtype::render;

/**
 * @example Creating materials in setup()
 *
 * This example shows how to set up materials in your Cinder app's setup() method.
 */
inline void materialPoolExample(Device& device) {

    //==========================================================================
    // 1. Create the MaterialPool
    //==========================================================================

    MaterialPool pool(device);

    //==========================================================================
    // 2. Create simple color materials (no textures)
    //==========================================================================

    // White diffuse material
    uint whiteIdx = pool.createMaterial(
        "white",
        make_float3(0.9f, 0.9f, 0.9f),  // albedo
        0.9f,    // roughness
        0.0f,    // metallic
        0        // type: diffuse
    );

    // Red plastic material
    uint redPlasticIdx = pool.createMaterial(
        "red_plastic",
        make_float3(0.8f, 0.1f, 0.1f),
        0.5f,
        0.0f,
        3  // type: plastic
    );

    // Gold metallic material
    uint goldIdx = pool.createMaterial(
        "gold",
        make_float3(1.0f, 0.765f, 0.336f),
        0.3f,
        1.0f,
        4  // type: metallic
    );

    // Mirror material
    uint mirrorIdx = pool.createMaterial(
        "mirror",
        make_float3(0.95f, 0.95f, 0.95f),
        0.0f,
        0.0f,
        1  // type: mirror
    );

    // Glass material
    uint glassIdx = pool.createMaterial(
        "glass",
        make_float3(0.95f, 0.95f, 0.95f),
        0.0f,
        0.0f,
        2  // type: glass
    );

    //==========================================================================
    // 3. Load materials from texture folders
    //==========================================================================

    // Load all textures from a folder using naming convention
    // Expects files like: wood_albedo.png, wood_normal.png, etc.
    uint woodIdx = pool.createMaterialFromFolder(
        "wood",
        "assets/textures/wood"  // folder path
    );

    // Load textured metal with custom base properties
    MaterialData rustedIronData{
        .albedo = make_float3(0.8f, 0.8f, 0.8f),
        .roughness = 0.6f,
        .metallic = 1.0f,
        .type = 4  // metallic
    };
    uint rustedIronIdx = pool.createMaterialFromFolder(
        "rusted_iron",
        "assets/textures/rusted_iron",
        rustedIronData
    );

    //==========================================================================
    // 4. Clone existing materials
    //==========================================================================

    // Clone gold with different roughness
    MaterialData shinyGold{
        .roughness = 0.1f,  // Shinier than original gold
        // Other properties inherited from goldIdx
    };
    uint shinyGoldIdx = pool.cloneMaterial("shiny_gold", goldIdx, &shinyGold);

    //==========================================================================
    // 5. Update materials at runtime
    //==========================================================================

    // Update material properties
    MaterialData newWhiteData{
        .albedo = make_float3(0.5f, 0.5f, 0.5f),  // Darker gray
        .roughness = 0.95f,
    };
    pool.updateMaterialData(whiteIdx, newWhiteData);

    //==========================================================================
    // 6. Synchronize with GPU
    //==========================================================================

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    pool.update(stream);  // Upload dirty materials to GPU

    //==========================================================================
    // 7. Use materials with mesh instances
    //==========================================================================

    // Later, when adding mesh instances:
    // DynamicScene scene(device, resolution);
    // scene.addInstance("floor", floorMesh, whiteIdx);
    // scene.addInstance("sphere", sphereMesh, goldIdx);
    // scene.addInstance("cube", cubeMesh, woodIdx);

    //==========================================================================
    // 8. Access materials in shaders
    //==========================================================================

    // In your path tracer shader:
    //
    // Kernel2D path_tracer = [&](
    //     ImageFloat image,
    //     AccelVar accel,
    //     BufferVar<MaterialData> materials,  // Material pool buffer
    //     BindlessArrayVar textures,           // Texture bindless array
    //     ...) {
    //
    //     // After ray hit:
    //     Var<MaterialData> material = materials.read(instance.materialIndex);
    //
    //     // Sample albedo with optional texture
    //     Float3 albedo = material->get_albedo(uv, textures);
    //
    //     // Get roughness
    //     Float roughness = material->get_roughness(uv, textures);
    //
    //     // Check if emissive
    //     $if(material->is_emissive()) {
    //         radiance += material->get_emission(uv, textures);
    //     };
    // };
}

/**
 * @example Material usage in NewTypeEngine
 *
 * Integration example showing how to use MaterialPool with the rest
 * of the NewTypeEngine system.
 */
inline void newTypeEngineIntegrationExample() {

    /*
    // In NewTypeEngine.cpp:

    class NewTypeEngine : public App {
    private:
        std::unique_ptr<render::DynamicScene> mScene;

        void setup() override {
            // Create renderer
            renderer = core::Renderer::create(
                getWindowWidth(), getWindowHeight());

            // Create dynamic scene (includes MaterialPool)
            mScene = std::make_unique<render::DynamicScene>(
                core::Renderer::device(),
                uint2(getWindowWidth(), getWindowHeight())
            );

            auto& materialPool = mScene->materials();

            // Create materials
            uint floorMatIdx = materialPool.createMaterial(
                "floor",
                make_float3(0.725f, 0.710f, 0.680f),
                0.9f
            );

            uint metalMatIdx = materialPool.createMaterialFromFolder(
                "rusted_iron",
                "assets/textures/rusted_iron"
            );

            // Load meshes
            auto sphereMesh = util::Mesh::create();
            sphereMesh->pack(ci::TriMesh(ci::geom::Sphere().radius(0.5f)));

            auto floorMesh = util::Mesh::create();
            floorMesh->pack(ci::TriMesh(ci::geom::Cube().size(10.f, 0.1f, 10.f)));

            // Add instances with material indices
            mSphereID = mScene->addInstance("sphere", sphereMesh, metalMatIdx);
            mScene->addInstance("floor", floorMesh, floorMatIdx);

            // Build scene
            mScene->build();
        }

        void update() override {
            // Animate objects
            auto* sphere = mScene->getInstance(mSphereID);
            if (sphere) {
                static float angle = 0.0f;
                angle += 0.01f;
                sphere->setPosition(make_float3(sin(angle) * 2.0f, 0.0f, cos(angle) * 2.0f));
                mScene->markDirty();
            }

            // Update scene (includes material pool)
            mScene->update(core::Renderer::stream());
        }

        void draw() override {
            auto& frame = renderer->beginFrame();

            // Render with material pool
            auto& shader = getShader();
            core::Renderer::stream()
                << shader(frame.render_target,
                          mScene->accel(),
                          mScene->materials(),        // MaterialData buffer
                          mScene->materialTextures(),  // Texture bindless array
                          mCamera->cam())
                .dispatch(renderer->width(), renderer->height())
                << synchronize();

            renderer->endFrame();
            gl::draw(renderer->currentTexture(), getWindowBounds());
        }

        void changeMaterialAtRuntime() {
            // Example: Change sphere material when key pressed
            auto& pool = mScene->materials();

            // Create new material or get existing
            uint goldIdx = pool.createMaterial(
                "gold",
                make_float3(1.0f, 0.765f, 0.336f),
                0.3f,
                1.0f,
                4
            );

            // Update sphere to use gold material
            auto* sphere = mScene->getInstance(mSphereID);
            if (sphere) {
                sphere->setMaterialIndex(goldIdx);
            }

            // Sync to GPU
            pool.update(core::Renderer::stream());
        }
    };
    */
}

/**
 * @example Texture folder structure
 *
 * Expected folder structure for createMaterialFromFolder():
 *
 * assets/
 *   textures/
 *     wood/
 *       wood_albedo.png        (or wood_diffuse.png, wood_color.png)
 *       wood_normal.png        (or wood_nrm.png, wood_n.png)
 *       wood_roughness.png     (or wood_rough.png, wood_r.png)
 *       wood_metallic.png      (or wood_metal.png, wood_m.png)
 *       wood_emissive.png      (or wood_emission.png, wood_e.png)
 *       wood_ao.png            (or wood_ambient_occlusion.png)
 *
 *     rusted_iron/
 *       rusted_iron_albedo.png
 *       rusted_iron_normal.png
 *       rusted_iron_roughness.png
 *       rusted_iron_metallic.png
 */

} // namespace newtype::example
