/**
 * @brief TestShader Implementation
 *
 * When TESTSHADER_EXPORTS is defined (building as DLL):
 * - Exports createTestShader() and destroyTestShader() functions
 *
 * When RT_RUNTIME is NOT defined (static linking):
 * - Provides static compile() function
 *
 * EDIT THIS CODE FOR HOT-RELOAD!
 */

#include "TestShader.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// TestShader Kernel Implementation
//==============================================================================

static luisa::compute::Resource* compileTestShader(luisa::compute::Device& device) {
    // TODO: Implement your shader kernel here
    //
    // Example kernel structure:
    // Kernel2D testshader_kernel = [&](/* params */) noexcept {
    //     set_block_size(16u, 16u, 1u);
    //     UInt2 coord = dispatch_id().xy();
    //
    //     // Your shader code here
    // };

    // Placeholder kernel - replace with your implementation
    Kernel2D testshader_kernel = [&](
        ImageFloat image, ImageFloat src, UInt uint
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2  coord = dispatch_id().xy();
        Float2 resolution = make_float2(dispatch_size().xy());
        Float2 uv = make_float2(coord) / resolution;

        auto px = src.read(coord);
        image.write(coord, make_float4(px.xyz(), 1.f));
    };

    // Heap-allocate so the shader is destroyed when destroyTest calls delete,
    // BEFORE FreeLibrary. A function-local static would have its destructor
    // run during DLL_PROCESS_DETACH, where calling device()->destroy_shader()
    // via the vtable is unsafe.
    auto compiled = device.compile(testshader_kernel);
    return new decltype(compiled)(std::move(compiled));
}

//==============================================================================
// DLL Export Functions
//==============================================================================
#ifndef RT_RUNTIME
extern "C" {

TESTSHADER_API luisa::compute::Resource* createTest(
    luisa::compute::Device& device) {

    return compileTestShader(device);
}

TESTSHADER_API void destroyTest(
    luisa::compute::Resource* shader) {
    // Virtual ~Resource() runs ~ShaderBase → device->destroy_shader(handle),
    // releasing the GPU shader handle. Must be invoked BEFORE FreeLibrary so
    // the destructor runs in normal app context, not during DLL_PROCESS_DETACH.
    delete shader;
}

} // extern "C"
#endif
//==============================================================================
// Static Mode Implementation
//==============================================================================

#if defined(RT_RUNTIME) || defined(RT_RUNTIME_DLL)
// Runtime mode: compile() is not used, but we need a stub implementation
// to allow instantiation of TestShader for metadata access
luisa::unique_ptr<luisa::compute::Resource> TestShader::compile(Device& device) {
    // In runtime mode, shaders are loaded from DLL, not compiled directly
    throw std::runtime_error("TestShader::compile() should not be called in runtime mode. Use loadShader(..., true) instead.");
}

#else

luisa::unique_ptr<luisa::compute::Resource> TestShader::compile(Device& device) {
    // createTest now returns Resource*
    auto* resource = static_cast<luisa::compute::Resource*>(createTest(device));
    return luisa::unique_ptr<luisa::compute::Resource>(resource);
}

#endif