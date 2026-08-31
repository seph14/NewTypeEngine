#pragma once

/**
 * @brief SimpleTestShader - Interface header
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
#include "SimpleTestShaderAPI.h"
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
 *   SimpleTestShader generator;
 *   ShaderManager::instance().loadShader(generator);
 *
 *   // In runtime mode - DLL is loaded automatically
 *   auto& shader = ShaderManager::instance().getShader<SimpleTestShaderType>("SimpleTestShader");
 * @endcode
 */
class SimpleTestShader : public newtype::core::IShaderGenerator<2u> {
public:
    // Shader type alias - matches the DLL export
    using ShaderType = SimpleTestShaderType;

    SimpleTestShader() = default;
    ~SimpleTestShader() override = default;

    // IShaderGenerator interface
    [[nodiscard]] std::string getName() const override { return "SimpleTest"; }
    [[nodiscard]] std::string getTypeName() const override { return "SimpleTestShader"; }
    
    [[nodiscard]] luisa::unique_ptr<luisa::compute::Resource> compile(
        luisa::compute::Device& device) override;

private:
    // In runtime mode, these are implemented in the DLL
    // In static mode, they are implemented in SimpleTestShader.cpp
    static SimpleTestShaderType* create(luisa::compute::Device& device);
    static void destroy(SimpleTestShaderType* shader);
};