#pragma once

/**
 * @brief TestShader - Interface header
 *
 * In Runtime mode (RT_RUNTIME defined):
 * - Shaders are loaded from external DLL
 * - Interface only, implementation is in the DLL
 *
 * In Static mode (RT_RUNTIME not defined):
 * - Shaders are statically linked
 * - Implementation is included directly
 */

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "TestShaderAPI.h"
#include "newtype/core/IShaderGenerator.h"


/**
 * @brief None
 *
 * In Debug_Runtime mode (RT_RUNTIME defined), changes to the shader code
 * will trigger automatic recompilation and hot-reload via DLL.
 *
 * In Debug/Release mode, this is statically linked with zero overhead.
 *
 * Usage:
 * @code
 *   // In static mode
 *   TestShader generator;
 *   ShaderManager::instance().loadShader(generator);
 *
 *   // In runtime mode - DLL is loaded automatically
 *   auto& shader = ShaderManager::instance().getShader<TestShaderType>("TestShader");
 * @endcode
 */
class TestShader : public newtype::core::IShaderGenerator<2> {
public:
    // Shader type alias - matches the DLL export
    using ShaderType = TestShaderType;

    TestShader() = default;
    ~TestShader() override = default;

    // IShaderGenerator interface
    [[nodiscard]] std::string getName() const override { return "Test"; }
    [[nodiscard]] std::string getTypeName() const override { return "TestShader"; }

    [[nodiscard]] luisa::unique_ptr<luisa::compute::Resource> compile(
        luisa::compute::Device& device) override;

private:
    // In runtime mode, these are implemented in the DLL
    // In static mode, they are implemented in TestShader.cpp
    static TestShaderType* create(luisa::compute::Device& device);
    static void destroy(TestShaderType* shader);
};