#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <string>
#include "../util/MoveOnlyAny.h"

#ifdef RT_RUNTIME
#include "newtype/runtime/Virtual.h"
#endif

namespace newtype::core {

/**
 * @brief Abstract interface for runtime-compiled shader generators
 *
 * Shader generators implement this interface to provide hot-reloadable
 * DSL shaders. In Debug_Runtime mode, these are loaded from DLLs and
 * recompiled when source changes. In Debug/Release mode, they are
 * statically linked with zero overhead.
 *
 * Usage:
 * @code
 * // Define a shader generator
 * class PathTracerShader : public IShaderGenerator {
 *     RT_DECL  // Empty in Release, tracked in Debug_Runtime
 * public:
 *     using ShaderType = Shader2D<Image<float>, Image<uint>, Accel, CameraData, uint>;
 *
 *     rt_virtual ShaderType compile(Device& device) override;
 * };
 * @endcode
 */    
template<uint dim>
class IShaderGenerator {
public:
    virtual ~IShaderGenerator() = default;

    /**
     * @brief Get the unique name identifier for this shader
     */
    [[nodiscard]] virtual std::string getName() const = 0;

    /**
     * @brief Compile and return the shader
     *
     * This method is called during initial load and after hot-reload.
     * The shader type is defined by the implementing class.
     */
    [[nodiscard]] virtual luisa::unique_ptr<luisa::compute::Resource> compile(luisa::compute::Device& device) = 0;

    /**
     * @brief Get the type name of the shader for type checking
     */
    [[nodiscard]] virtual std::string getTypeName() const = 0;

protected:
#ifdef RT_RUNTIME
    // In runtime mode, enable dynamic dispatch
    IShaderGenerator() = default;
#else
    // In release mode, allow efficient construction
    IShaderGenerator() = default;
#endif
};

} // namespace newtype::core
