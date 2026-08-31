/**
 * @brief SimpleTestShader Implementation
 *
 * Simple ray tracing test shader to verify geometry foundations.
 * Now includes LightSampler integration for direct lighting with proper triangle sampling and shadow rays.
 *
 * When SIMPLETEST_EXPORTS is defined (building as DLL):
 * - Exports createSimpleTest() and destroySimpleTest() functions
 *
 * When RT_RUNTIME is NOT defined (static linking):
 * - Provides static compile() function
 *
 * EDIT THIS CODE FOR HOT-RELOAD!
 */

#include "SimpleTestShader.h"
#include "newtype/util/Camera.h"
#include "newtype/render/LightSampler.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/scene/Geometry.h"
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/expr.h>
#include <luisa/dsl/stmt.h>
#include <luisa/dsl/struct.h>

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// SimpleTestShader Kernel Implementation
//==============================================================================

static luisa::compute::Resource* compileSimpleTest(luisa::compute::Device& device) {
    using namespace newtype::render;

    //==========================================================================
    // DSL Helper Callables (defined locally within compile scope)
    //==========================================================================

    // LCG random number generator
    Callable<uint(uint)> lcg = [&](UInt seed) noexcept {
        const UInt a = 1664525u;
        const UInt c = 1013904223u;
        return ite(seed == 0u, 1u, seed * a + c);
    };

    // Sample uniform triangle (barycentric)
    Callable<float3(float2)> sample_uniform_triangle = [&](Float2 u) noexcept {
        Float su = sqrt(u.x);
        Float2 uv = make_float2(1.0f - su, u.y * su);
        return make_float3(uv.x, uv.y, 1.0f - uv.x - uv.y);
    };

    // Sample point on triangle light
    Callable<float3(float3, float3, float3, float2)> sample_triangle_point =
        [&](Float3 p0, Float3 p1, Float3 p2, Float2 u) noexcept {
            Float3 bary = sample_uniform_triangle(u);
            return bary.x * p0 + bary.y * p1 + bary.z * p2;
        };

    //==========================================================================
    // Main Kernel
    //==========================================================================

    Kernel2D simpletest_kernel = [&](
        ImageFloat image,                              // 0: output
        ImageUInt seed_image,                          // 1: seeds
        ImageFloat accum_buffer,                       // 2: accumulation buffer (running sum)
        ImageUInt sample_count_buffer,                 // 3: sample count per pixel
        UInt frame_count,                              // 4: current frame index
        UInt reset_accum,                              // 5: reset accumulation flag
        AccelVar accel,                                // 6: TLAS
        Var<newtype::util::CameraData> camera,         // 7: camera
        BufferVar<uint4> instance_buffer,              // 8: instance data
        BufferVar<MaterialData> material_buffer,       // 9: materials
        BufferVar<LightSampler::TriangleLight> triangle_lights,  // 10: triangle lights
        BufferVar<LightSampler::TriangleVertexData> triangle_vertices,  // 11: triangle vertices
        BufferVar<AliasEntry> alias_table,             // 12: alias table
        UInt emissive_count,                           // 13: number of emissive triangles
        Float total_power_inv                          // 14: inverse total power
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord      = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();

        // Get and update seed
        UInt seed = seed_image.read(coord).x;
        seed = lcg(seed);

        // Generate camera ray
        Float2 uv        = (make_float2(coord) + 0.5f) / make_float2(resolution);
        Float2 ndc       = uv * 2.0f - 1.0f;
        Float  aspect    = resolution.x / resolution.y;
        Float2 pixel_ndc = ndc * make_float2(aspect, 1.0f);

        Var<Ray> ray = camera->generate_ray(pixel_ndc);

        // Trace ray
        Var<TriangleHit> hit = accel.intersect(ray, {});

        // Default color (sky gradient)
        Float3 color = def(lerp(
            make_float3(0.02f, 0.02f, 0.03f),
            make_float3(0.1f, 0.15f, 0.2f),
            (ndc.y + 1.0f) * 0.5f));

        $if (!hit->miss()) {
            // Get instance and primitive info
            UInt inst_id = hit.inst;
            UInt prim_id = hit.prim;

            // Compute hit position (from ray and hit distance)
            Float hit_t = hit.committed_ray_t;  // Direct member access
            Float3 hit_pos = ray->origin() + ray->direction() * hit_t;

            // Get instance properties
            UInt4 instance_props = instance_buffer.read(inst_id);
            UInt material_layers = instance_props.y;  // Packed 4 × 8-bit material indices
            UInt material_id = material_layers & 0xFFu;  // Layer 0 = base material

            // Read material
            Var<MaterialData> material = material_buffer.read(material_id);

            // Simple geometric normal (assume up normal for floor)
            // TODO: Interpolate vertex normals for proper shading
            Float3 ng = normalize(make_float3(0.0f, 1.0f, 0.0f));
            ng = ite(inst_id == 1u, normalize(make_float3(0.0f, 0.0f, 1.0f)), ng);  // Cube front

            // Base color from material
            Float3 base_color = material.albedo.xyz();

            // Initialize direct lighting
            Float3 direct_light = make_float3(0.0f);

            // Sample ONE light per frame (temporal accumulation will reduce variance over time)
            $if (emissive_count > 0u) {
                // Get random numbers for light sampling
                Float u_select = cast<float>(seed) * (1.0f / 4294967296.0f);
                seed = lcg(seed);
                Float u_tri_x = fract(cast<float>(seed) * (1.0f / 4294967296.0f));
                seed = lcg(seed);
                Float u_tri_y = fract(cast<float>(seed) * (1.0f / 4294967296.0f));
                seed = lcg(seed);

                // Sample from alias table (power-weighted light selection)
                Float n = cast<float>(emissive_count);
                UInt idx = cast<uint>(u_select * n);
                idx = min(idx, emissive_count - 1u);

                Var<AliasEntry> entry = alias_table.read(idx);
                UInt light_idx = ite(u_select - cast<float>(idx) < entry.pdf,
                    idx, entry.alias_index);
                light_idx = min(light_idx, emissive_count - 1u);

                // Read triangle light data (emission, area)
                Var<LightSampler::TriangleLight> tri_light = triangle_lights.read(light_idx);

                // Read triangle vertex positions (world space)
                Var<LightSampler::TriangleVertexData> verts = triangle_vertices.read(light_idx);

                // Sample point on triangle using barycentric coordinates
                Float2 u_tri = make_float2(u_tri_x, u_tri_y);
                Float3 light_point = sample_triangle_point(verts.v0, verts.v1, verts.v2, u_tri);

                // Compute lighting direction
                Float3 light_dir = light_point - hit_pos;
                Float light_dist = length(light_dir);
                light_dir = normalize(light_dir);

                // Compute geometric term (cosine at surface)
                Float cos_theta = max(0.0f, dot(ng, light_dir));

                // Trace shadow ray
                //Float3 shadow_origin = hit_pos + ng * 0.0001f;  // Offset to avoid self-intersection
                Var<Ray> shadow_ray = make_ray(hit_pos, light_dir, 0.001f, light_dist - 0.0001f);
                Var<TriangleHit> shadow_hit = accel.intersect(shadow_ray, {});

                // If no shadow hit, light is visible (use bitwise & for boolean logic)
                $if (shadow_hit->miss() & cos_theta > 0.0f) {
                    // Compute lighting with proper BRDF (Lambertian)
                    Float3 light_emission = tri_light->emission();

                    // Light PDF = pdf / area (precomputed selection probability)
                    Float light_pdf = tri_light.pdf / tri_light.area;

                    // Lambertian BRDF = albedo / pi
                    Float3 brdf = base_color * 0.3183099f;  // 1/pi

                    // MIS-weighted contribution (simplified - just direct sampling)
                    // L = (Le * BRDF * cos) / pdf
                    Float3 contrib = light_emission * brdf * cos_theta / light_pdf;

                    direct_light = direct_light + .05f * contrib;
                };
            };

            // Self-emission (light sources glow)
            Float3 emission = material.emission;

            // Combine: ambient + direct + emission
            color = base_color * 0.01f + direct_light +emission;
        };

        // Temporal accumulation
        UInt current_count = sample_count_buffer.read(coord).x;
        Float4 current_accum = accum_buffer.read(coord);

        // Check if we should reset (first frame or camera moved)
        UInt should_reset = ite(reset_accum != 0u | current_count == 0u, 1u, 0u);

        Float4 new_accum = ite(should_reset != 0u,
            make_float4(color, 1.0f),  // Start fresh
            current_accum + make_float4(color, 1.0f)  // Add to accumulation
        );

        UInt new_count = ite(should_reset == 1u, 1u, current_count + 1u);

        // Write back to buffers
        accum_buffer.write(coord, new_accum);
        sample_count_buffer.write(coord, make_uint4(new_count));

        // Output averaged color
        Float3 final_color = new_accum.xyz() / 3200.f / cast<float>(new_count);
        image.write(coord, make_float4(final_color, 1.0f));

        // Write back seed
        seed_image.write(coord, make_uint4(seed));
    };

    // Heap-allocate so the shader is destroyed when destroySimpleTest calls
    // delete, BEFORE FreeLibrary. A function-local static would have its
    // destructor run during DLL_PROCESS_DETACH, where calling
    // device()->destroy_shader() via the vtable is unsafe.
    auto compiled = device.compile(simpletest_kernel);
    return new decltype(compiled)(std::move(compiled));
}

//==============================================================================
// DLL Export Functions
//==============================================================================
#ifndef RT_RUNTIME
extern "C" {
SIMPLETEST_API luisa::compute::Resource* createSimpleTest(
    luisa::compute::Device& device) {

    return compileSimpleTest(device);
}

SIMPLETEST_API void destroySimpleTest(
    luisa::compute::Resource* shader) {
    // Virtual ~Resource() runs ~ShaderBase → device->destroy_shader(handle),
    // releasing the GPU shader handle. Must be invoked BEFORE FreeLibrary so
    // the destructor runs in normal app context, not during DLL_PROCESS_DETACH.
    delete shader;
}

} // extern "C"
#endif // SIMPLETEST_EXPORTS
//==============================================================================
// Static Mode Implementation
//==============================================================================

#if defined(RT_RUNTIME) || defined(RT_RUNTIME_DLL)

// Runtime mode: compile() is not used
luisa::unique_ptr<luisa::compute::Resource> SimpleTestShader::compile(Device& device) {
    throw std::runtime_error("SimpleTestShader::compile() should not be called in runtime mode. Use loadShader(..., true) instead.");
}

#else

luisa::unique_ptr<luisa::compute::Resource> SimpleTestShader::compile(Device& device) {
    return luisa::unique_ptr<luisa::compute::Resource>(compileSimpleTest(device));
}

#endif
