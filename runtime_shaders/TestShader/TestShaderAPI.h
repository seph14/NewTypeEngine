#pragma once

/**
 * @brief TestShader DLL Export Interface
 *
 * This header defines the API for the TestShader DLL.
 * It is shared between the DLL (export) and the application (import).
 */

#ifdef TESTSHADER_EXPORTS
// When building the DLL, export symbols
#define TESTSHADER_API __declspec(dllexport)
#elif !defined(RT_RUNTIME)
// In static mode (not runtime), don't use dll attributes
#define TESTSHADER_API
#else
// When using the DLL, import symbols
#define TESTSHADER_API __declspec(dllimport)
#endif

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>


/**
 * @brief TestShader shader type
 *
 * Defines the signature of the TestShader kernel.
 */
using TestShaderType = luisa::compute::Shader2D<
    luisa::compute::Image<float>,
    luisa::compute::Image<float>,
    uint
>;

extern "C" {
/**
 * @brief Factory function to create the TestShader shader
 *
 * This function is exported from the DLL and called by the application
 * to create a new instance of the TestShader shader.
 *
 * @param device LuisaCompute device for compilation
 * @return Pointer to the compiled shader as Resource* (Shader2D<...> inherits from Resource)
 */
TESTSHADER_API luisa::compute::Resource* createTest(
    luisa::compute::Device& device);

/**
 * @brief Destroy a TestShader shader instance
 *
 * Deletes the heap-allocated shader object. The virtual ~Resource()
 * destructor runs ~ShaderBase, which calls device->destroy_shader(handle)
 * to release the GPU shader handle. Must be called BEFORE FreeLibrary so
 * the destructor runs in normal app context, not during DLL_PROCESS_DETACH.
 *
 * @param shader Pointer to the shader Resource to destroy (may be null)
 */
TESTSHADER_API void destroyTest(
    luisa::compute::Resource* shader);

} // extern "C"