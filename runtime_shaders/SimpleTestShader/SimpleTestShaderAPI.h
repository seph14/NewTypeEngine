#pragma once

/**
 * @brief SimpleTestShader DLL Export Interface
 *
 * This header defines the API for the SimpleTestShader DLL.
 * It is shared between the DLL (export) and the application (import).
 */

#ifdef SIMPLETEST_EXPORTS
// When building the DLL, export symbols
#define SIMPLETEST_API __declspec(dllexport)
#elif !defined(RT_RUNTIME)
// In static mode (not runtime), don't use dll attributes
#define SIMPLETEST_API
#else
// When using the DLL, import symbols
#define SIMPLETEST_API __declspec(dllimport)
#endif

#include <luisa/luisa-compute.h>
#include "newtype/render/LightSampler.h"
#include "newtype/render/MaterialPool.h"

// Forward declarations
namespace newtype::util {
    struct CameraData;
}

/**
 * @brief SimpleTestShader shader type
 *
 * Defines the signature of the SimpleTestShader kernel.
 * Now includes LightSampler for direct lighting and MaterialPool for materials.
 * Also includes temporal accumulation buffers.
 */
using SimpleTestShaderType = luisa::compute::Shader<2,
    luisa::compute::Image<float>,                           // 0: output image
    luisa::compute::Image<luisa::uint>,                    // 1: seed image
    luisa::compute::Image<float>,                           // 2: accumulation buffer
    luisa::compute::Image<luisa::uint>,                    // 3: sample count buffer
    luisa::uint,                                           // 4: frame count
    luisa::uint,                                           // 5: reset accumulation flag
    luisa::compute::Accel,                                  // 6: TLAS
    newtype::util::CameraData,                              // 7: camera
    luisa::compute::Buffer<luisa::uint4>,                   // 8: instance buffer
    luisa::compute::Buffer<newtype::render::MaterialData>,  // 9: material buffer
    luisa::compute::Buffer<newtype::render::LightSampler::TriangleLight>,  // 10: triangle lights
    luisa::compute::Buffer<newtype::render::LightSampler::TriangleVertexData>,  // 11: triangle vertices
    luisa::compute::Buffer<newtype::render::AliasEntry>,  // 12: alias table
    luisa::uint,                                           // 13: emissive triangle count
    float                                                  // 14: total power (inverse)
>;

extern "C" {
/**
 * @brief Factory function to create the SimpleTestShader shader
 *
 * This function is exported from the DLL and called by the application
 * to create a new instance of the SimpleTestShader shader.
 *
 * @param device LuisaCompute device for compilation
 * @return Pointer to the compiled shader as Resource* (Shader<dim,...> inherits from Resource)
 */
SIMPLETEST_API luisa::compute::Resource* createSimpleTest(
    luisa::compute::Device& device);

/**
 * @brief Destroy a SimpleTestShader shader instance
 *
 * Deletes the heap-allocated shader object. The virtual ~Resource()
 * destructor runs ~ShaderBase, which calls device->destroy_shader(handle)
 * to release the GPU shader handle. Must be called BEFORE FreeLibrary so
 * the destructor runs in normal app context, not during DLL_PROCESS_DETACH.
 *
 * @param shader Pointer to the shader Resource to destroy (may be null)
 */
SIMPLETEST_API void destroySimpleTest(
    luisa::compute::Resource* shader);

} // extern "C"
