#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <cinder/Log.h>
#include <cinder/app/App.h>
#include "../util/MoveOnlyAny.h"
#include "newtype/core/IShaderGenerator.h"

// Include DLLHotReload header for runtime mode (needed for getShader template)
#ifdef RT_RUNTIME
#include "newtype/runtime/DLLHotReload.h"
#endif

#include <string>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <mutex>
#include <thread>
#include <atomic>
#include <filesystem>
#include <vector>
#include <memory>

// Type alias for DLL loader pointer
#ifdef RT_RUNTIME
namespace newtype::runtime {
    using DllRef = std::unique_ptr<DLLHotReload>;
}
#endif

namespace newtype::core {

// Forward declarations
template<uint dim>
class IShaderGenerator;

/**
 * @brief Callback type for shader reload events
 *
 * Called when a shader is successfully reloaded. The app can
 * use this to reset accumulation buffers or update state.
 */
using ReloadCallback = std::function<void(std::string_view)>;

/**
 * @brief Hot-reloadable DSL shader manager
 *
 * Supports dual-mode operation:
 * - **Debug_Runtime**: File watching enabled, detects shader changes
 * - **Debug/Release**: Static linking with zero overhead
 *
 * Features:
 * - Load shaders from IShaderGenerator instances
 * - File watching for shader source changes (Debug_Runtime only)
 * - Callable registry for helper functions
 * - Singleton pattern for app-wide access
 *
 * Usage:
 * @code
 *   // In setup() - unified API for both modes!
 *   auto& sm = ShaderManager::instance();
 *
 *   PathTracerShader generator;  // Implements IShaderGenerator
 *   sm.loadShader(generator, true);  // Enable hot-reload (Debug_Runtime only)
 *
 *   // In update() - enables hot reload detection in Debug_Runtime
 *   ShaderManager::instance().update();
 *
 *   // In draw()
 *   auto& shader = sm.getShader<PathTracerShader::ShaderType>("path_tracer");
 *   stream << shader(args...).dispatch(w, h);
 * @endcode
 */
class ShaderManager {
public:
    using RscHdl = luisa::unique_ptr<luisa::compute::Resource>;

    // Non-copyable, non-movable
    ShaderManager(const ShaderManager&) = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;
    ShaderManager(ShaderManager&&) = delete;
    ShaderManager& operator=(ShaderManager&&) = delete;

    /**
     * @brief Get the singleton instance
     */
    static ShaderManager& instance();

    /**
     * @brief Load a shader from an IShaderGenerator
     *
     * In Debug_Runtime mode with hotReload=true:
     *   - Loads the shader as a DLL for hot-reloading
     *   - File watching enabled for source changes
     *
     * In Debug/Release mode OR with hotReload=false:
     *   - Compiles the shader directly from the generator
     *   - No hot-reload capability (zero overhead)
     *
     * @param generator Shader generator instance
     * @param hotReload Enable DLL hot-reload (ignored in Debug/Release, only applies to Debug_Runtime)
     * @return true if loaded successfully
     */

#ifdef RT_RUNTIME
    template<uint dim>
    bool loadShader(IShaderGenerator<dim>& generator, bool hotReload = true) {
        return loadShaderImpl(generator, hotReload);
    }
#else
    template<uint dim>
    bool loadShader(IShaderGenerator<dim>& generator, bool hotReload = false) {
        return loadShaderImpl(generator, hotReload);
    }
#endif
    /**
     * @brief Load a DLL shader for hot-reload (runtime mode only)
     *
     * This overload allows loading a DLL shader without requiring an IShaderGenerator instance.
     * Use this when you want to load a pre-compiled DLL shader.
     *
     * @param name Unique identifier for the shader (used to construct DLL name: name + "Shader.dll")
     * @param sourcePath Path to the source file to watch for changes
     * @return true if loaded successfully
     */
    //bool loadShader(std::string_view name, std::string_view sourcePath);

#ifdef RT_RUNTIME
    /**
     * @brief Set the reload callback for the DLL hot-reload manager
     *
     * @param callback Function to call after successful reload
     */
    void setReloadCallback(ReloadCallback callback);
#endif

    void launch();

#ifdef RT_RUNTIME
    /**
     * @brief Process any pending shader DLL reloads (main thread only).
     *
     * Called from the render loop. For each DLL whose source changed (flag
     * set by the watch thread's checkForChanges), this performs the actual
     * destroy + rebuild + compile on the calling thread. Doing the work here
     * — rather than on the watch thread — guarantees the render loop is
     * paused and no in-flight stream commands reference the shader we're
     * about to delete.
     *
     * @param stream Stream to synchronize before destroying the old shader
     */
    void processPendingReloads(luisa::compute::Stream& stream);
#endif

    /**
     * @brief Register a pre-compiled shader directly
     *
     * Use this when you have a pre-compiled shader and want to manage
     * recompilation yourself.
     *
     * @tparam ShaderT The shader type
     * @param name Unique identifier
     * @param shader Compiled shader kernel
     * @return true if registered successfully
     */
    template<uint dim, typename Def>
    bool registerShader(std::string_view name, Def&& def);

    /**
     * @brief Get a compiled shader by name
     *
     * Searches BOTH static shaders AND DLL-loaded shaders.
     * Returns the first match found.
     *
     * @tparam ShaderT The shader type
     * @param name Shader identifier
     * @return Reference to the compiled shader
     * @throws std::runtime_error if shader not found
     */
    template<uint dim, typename... Args, typename... CallArgs>
    [[nodiscard]] auto shader(std::string_view name, CallArgs &&...call_args) const noexcept;

    /**
     * @brief Register a helper callable for use in shaders
     *
     * @tparam CallableT The callable type
     * @param name Unique identifier
     * @param callable The LuisaCompute Callable
     */
    template<typename CallableT>
    void registerCallable(std::string_view name, CallableT&& callable);

    /**
     * @brief Clear all cached shaders
     */
    void clear();

    /**
     * @brief Check if a shader is loaded
     */
    [[nodiscard]] bool hasShader(std::string_view name) const;

    /**
     * @brief Get list of all loaded shader names
     */
    [[nodiscard]] std::vector<std::string> shaderNames() const;

    /**
     * @brief Check if running in runtime mode
     */
    [[nodiscard]] static bool isRuntimeMode() noexcept {
#ifdef RT_RUNTIME
        return true;
#else
        return false;
#endif
    }

private:
    ShaderManager();
    ~ShaderManager();

    // Template implementation for loadShader (must be in header)
    template<uint dim>
    bool loadShaderImpl(IShaderGenerator<dim>& generator, bool hotReload) {
        if (!sDevice) {
            logError("ShaderManager: Device not set. Call from Renderer context.");
            return false;
        }

        auto name = generator.getName();

#ifdef RT_RUNTIME
        if(!hotReload)
#endif
        {
            // Static compilation path
            try {
                auto compiled   = generator.compile(*sDevice);
                {
                    std::lock_guard lock(_shadersMutex);

                    ShaderEntry entry;
                    entry.kernel    = std::move(compiled);
                    entry.valid     = true;
                    _shaders[name]  = std::move(entry);
                }

                CI_LOG_I("ShaderManager: Loaded shader '" << name << "' (static compilation)");
                return true;
            } catch (const std::exception& e) {
                logError("Failed to compile shader '" + name + "': " + e.what());
                return false;
            }
        }

#ifdef RT_RUNTIME
        if (_dllLoaders.find(name) != _dllLoaders.end()) {
            logError("Shader '" + name + "' already loaded as DLL");
            return false;
        }

        auto projPath = "../runtime_shaders/" + name + "Shader/" + name;
        // Create DLL loader for this shader
        _dllLoaders[name] = std::make_unique<runtime::DLLHotReload>(
            name,
            "build/Runtime/x64/Debug_Runtime/" + name + "Shader.dll",
            projPath + "Shader.vcxproj",
            projPath + "Shader.cpp"
        );

        if (!_dllLoaders[name]->load(*sDevice)) {
            logError("Failed to load DLL shader '" + name + "': " + _dllLoaders[name]->getLastError());
            _dllLoaders.erase(name);
            return false;
        }

        CI_LOG_I("ShaderManager: Loaded shader '" << name << "' as DLL (hot-reload enabled)");
        return true;
#endif
    }

    // Shader entry
    struct ShaderEntry {
        RscHdl kernel;      // Stores the compiled shader
        bool valid = false; // True if compilation succeeded
    };

    // Runtime-specific members
#ifdef RT_RUNTIME
    void stopWatchThread();
    void watchThreadFunc();
    std::thread _watchThread;
    std::atomic<bool> _running{false};

    // DLL hot-reload manager (for runtime shader DLLs)
    std::unordered_map<std::string, runtime::DllRef> _dllLoaders;
#endif

    // File watching utilities
    [[nodiscard]] static std::filesystem::file_time_type getLastModified(
        const std::filesystem::path& path);

    // Log error message
    static void logError(std::string_view msg);

    // Members
    mutable std::mutex _shadersMutex;
    std::unordered_map<std::string, ShaderEntry> _shaders;

    mutable std::mutex _callablesMutex;
    std::unordered_map<std::string, std::any> _callables;  // Stores Callable<Ts...>

    // LuisaCompute device (for shader compilation)
    static luisa::compute::Device* sDevice;

    // Set device reference
    static void setDevice(luisa::compute::Device* device) { sDevice = device; }

    // Allow Renderer to set device
    friend class Renderer;
};

//==============================================================================
// Template Implementations
//==============================================================================

template<uint dim, typename Def>
bool ShaderManager::registerShader(std::string_view name, Def&& def) {
    static_assert(dim == 1u || dim == 2u || dim == 3u);
    std::lock_guard lock(_shadersMutex);

    ShaderEntry entry;
    auto shader     = sDevice->compile<dim>(std::forward<Def>(def));
    auto resource   = luisa::make_unique<decltype(shader)>(std::move(shader));
    entry.kernel    = std::move(resource);
    entry.valid     = true;

    _shaders[std::string(name)] = std::move(entry);
    CI_LOG_I("ShaderManager: Registered shader '" << name << "' (static)");
    return true;
}

template<uint dim, typename... Args, typename... CallArgs>
auto ShaderManager::shader(std::string_view name, CallArgs &&...call_args) const noexcept {
    
    auto it = _shaders.find(std::string(name));
    if (it != _shaders.end()) {
        if (!it->second.valid) {
            logError("Shader '" + std::string(name) + "' is invalid (compilation failed)");
            throw std::runtime_error("Shader '" + std::string(name) + "' is invalid");
        }
        
        // if found, return shader
        auto shader = dynamic_cast<const luisa::compute::Shader<dim, Args...>*>(
            it->second.kernel.get());
        return (*shader)(std::forward<CallArgs>(call_args)...);
    }

#ifdef RT_RUNTIME
    // If not found in static shaders, check DLL shaders
    {
        auto dllIt = _dllLoaders.find(std::string(name));
        if (dllIt != _dllLoaders.end()) {
            if (dllIt->second->isLoaded()) {
                auto shader = dllIt->second->template getShader<dim, Args...>();
                if(shader) return (*shader)(std::forward<CallArgs>(call_args)...);
            }
        }
    }
#endif

    // Not found anywhere
    logError("Shader not found: " + std::string(name));
    throw std::runtime_error("Shader not found: " + std::string(name));
}

template<typename CallableT>
void ShaderManager::registerCallable(std::string_view name, CallableT&& callable) {
    std::lock_guard lock(_callablesMutex);
    _callables[std::string(name)] = std::forward<CallableT>(callable);
    CI_LOG_D("ShaderManager: Registered callable '" << name << "'");
}

} // namespace newtype::core
