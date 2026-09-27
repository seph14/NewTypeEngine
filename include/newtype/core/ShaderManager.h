#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <cinder/Log.h>
#include <cinder/app/App.h>
#include "../util/MoveOnlyAny.h"
#include "newtype/core/IShaderGenerator.h"
#include "newtype/core/EngineVersion.h"

// Include DLLHotReload header for the DLL shader machinery (compiled into
// every _DEBUG binary — the prebuilt Debug lib serves both plain-Debug and
// Debug_Runtime (RT_RUNTIME) consumers; see docs/Prebuilt Engine Library.md).
#ifdef _DEBUG
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
#ifdef _DEBUG
namespace newtype::runtime {
    using DllRef = std::unique_ptr<DLLHotReload>;
}
#endif

namespace newtype::core {

// Forward declarations
template<uint dim>
class IShaderGenerator;
class ShaderManager;

/**
 * @brief Typed handle to a ShaderManager-registered shader.
 *
 * Replaces per-dispatch string lookups: resolution (name lookup +
 * dynamic_cast) happens once on the first dispatch, and again only after a
 * registry mutation (registration, clear, DLL hot-reload) invalidates the
 * cached pointer — detected via a generation compare. Steady-state dispatch
 * cost is one integer compare plus the shader invocation itself.
 *
 * assign() only records the name; resolution is deferred to the first
 * dispatch, so the shader may be registered before or after assign().
 *
 * All handle use must happen on the main thread (the same thread as every
 * registry mutation; the RT_RUNTIME watch thread never touches the registry).
 */
template<uint dim, typename... Args>
class ShaderHandle {
public:
    ShaderHandle() = default;

    /// Configure the shader name; drops any cached resolution.
    void assign(std::string_view name) { _name = name; _shader = nullptr; }

    /// A name is configured (not necessarily resolvable yet).
    [[nodiscard]] bool is_set() const noexcept { return !_name.empty(); }
    /// Resolved and current (false until the first successful dispatch).
    [[nodiscard]] bool valid() const noexcept { return _shader != nullptr; }
    [[nodiscard]] const std::string &name() const noexcept { return _name; }

private:
    friend class ShaderManager;

    std::string _name;
    const luisa::compute::Shader<dim, Args...> *_shader = nullptr;
    uint32_t _generation = 0u;  // registry epoch _shader was resolved in
};

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
 * - **Debug/Debug_Runtime**: DLL shader machinery compiled in; loading a
 *   shader as a DLL (hotReload=true) enables file watching + hot reload
 * - **Release**: Static linking with zero overhead
 *
 * The machinery is keyed on _DEBUG (not RT_RUNTIME) so the prebuilt Debug
 * library serves both plain-Debug and Debug_Runtime (RT_RUNTIME) consumers
 * with an identical class layout. RT_RUNTIME only flips the loadShader()
 * hot-reload default and isRuntimeMode() at the consumer's side.
 *
 * Features:
 * - Load shaders from IShaderGenerator instances
 * - File watching for shader source changes (starts with the first DLL
 *   shader load; plain Debug apps without shader DLLs pay nothing)
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
 *   // Member: resolved once, re-resolved automatically after registry
 *   // mutations (registration, clear, DLL hot-reload).
 *   ShaderHandle<2, Buffer<float4>, uint> _pathTracer;
 *   _pathTracer.assign("path_tracer");
 *
 *   // In draw() - no string lookup on the hot path
 *   stream << sm.shader(_pathTracer, args...).dispatch(w, h);
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
     * With hotReload=true (the default when the consumer defines RT_RUNTIME,
     * i.e. the Debug_Runtime config):
     *   - Loads the shader as a DLL for hot-reloading
     *   - Starts the file-watch thread with the first DLL shader load
     *
     * With hotReload=false (the default in plain Debug and Release):
     *   - Compiles the shader directly from the generator
     *   - No hot-reload capability (zero overhead)
     *
     * @param generator Shader generator instance
     * @param hotReload Enable DLL hot-reload (mechanism exists in all _DEBUG
     *        builds; ignored in Release)
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

#ifdef _DEBUG
    /**
     * @brief Set the reload callback for the DLL hot-reload manager
     *
     * @param callback Function to call after successful reload
     */
    void setReloadCallback(ReloadCallback callback);
#endif

    void launch();

#ifdef _DEBUG
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
     * @brief Resolve a shader name into a typed handle (one-time cost)
     *
     * @tparam dim Shader dimension
     * @tparam Args Shader argument prototype types
     * @param name Shader identifier
     * @return Handle for use with shader(handle, args...); valid() is false
     *         if the name cannot be resolved yet
     */
    template<uint dim, typename... Args>
    [[nodiscard]] ShaderHandle<dim, Args...> resolve(std::string_view name) const noexcept;

    /**
     * @brief Dispatch a resolved shader handle
     *
     * Resolves the handle on first use and re-resolves after any registry
     * mutation (registration, clear, DLL hot-reload) or assign() — steady
     * state is a generation compare. Searches BOTH static and DLL-loaded
     * shaders.
     *
     * @throws std::runtime_error if the shader cannot be resolved
     */
    template<uint dim, typename... Args, typename... CallArgs>
    [[nodiscard]] auto shader(ShaderHandle<dim, Args...> &handle, CallArgs &&...call_args) const;

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
        newtype::detail::engineAbiCheck();
        if (!sDevice) {
            logError("ShaderManager: Device not set. Call from Renderer context.");
            return false;
        }

        auto name = generator.getName();

#ifdef _DEBUG
        if (!hotReload)
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
                    ++_generation;
                }

                CI_LOG_D("ShaderManager: Loaded shader '" << name << "' (static compilation)");
                return true;
            } catch (const std::exception& e) {
                logError("Failed to compile shader '" + name + "': " + e.what());
                return false;
            }
        }

#ifdef _DEBUG
        {
            std::lock_guard dllLock(_dllMutex);
            if (_dllLoaders.find(name) != _dllLoaders.end()) {
                logError("Shader '" + name + "' already loaded as DLL");
                return false;
            }
        }

        auto projPath = "../runtime_shaders/" + name + "Shader/" + name;
        // Create DLL loader for this shader
        auto loader = std::make_unique<runtime::DLLHotReload>(
            name,
            "build/Runtime/x64/Debug_Runtime/" + name + "Shader.dll",
            projPath + "Shader.vcxproj",
            projPath + "Shader.cpp"
        );

        if (!loader->load(*sDevice)) {
            logError("Failed to load DLL shader '" + name + "': " + loader->getLastError());
            return false;
        }

        // The watch thread starts with the first DLL shader load and only
        // polls loaded DLLs — apps that never load a shader DLL (plain Debug
        // default) never pay for it.
        launch();

        {
            std::lock_guard dllLock(_dllMutex);
            _dllLoaders[name] = std::move(loader);
        }

        CI_LOG_D("ShaderManager: Loaded shader '" << name << "' as DLL (hot-reload enabled)");
        return true;
#endif
    }

    // Shader entry
    struct ShaderEntry {
        RscHdl kernel;      // Stores the compiled shader
        bool valid = false; // True if compilation succeeded
    };

    // Runtime-specific members (compiled into every _DEBUG binary so the
    // prebuilt Debug lib serves Debug_Runtime consumers; inert unless a
    // shader DLL is actually loaded)
#ifdef _DEBUG
    void stopWatchThread();
    void watchThreadFunc();
    std::thread _watchThread;
    std::atomic<bool> _running{false};

    // DLL hot-reload manager (for runtime shader DLLs). Guarded by
    // _dllMutex: mutated on the main thread (load/clear), iterated by the
    // watch thread once the first DLL shader load has started it.
    mutable std::mutex _dllMutex;
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

    // Registry epoch: bumped on every mutation (register, load, clear, DLL
    // reload) to invalidate outstanding ShaderHandle pointers. Main thread
    // only — see ShaderHandle.
    uint32_t _generation = 1u;

    // Static map first, then DLL hot-reload shaders (RT_RUNTIME); null +
    // error log when unresolvable.
    template<uint dim, typename... Args>
    [[nodiscard]] const luisa::compute::Shader<dim, Args...> *
    _lookup_shader(std::string_view name) const noexcept;

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
    ++_generation;
    CI_LOG_D("ShaderManager: Registered shader '" << name << "' (static)");
    return true;
}

template<uint dim, typename... Args>
const luisa::compute::Shader<dim, Args...> *
ShaderManager::_lookup_shader(std::string_view name) const noexcept {
    {
        std::lock_guard lock(_shadersMutex);
        auto it = _shaders.find(std::string(name));
        if (it != _shaders.end()) {
            if (!it->second.valid) {
                logError("Shader '" + std::string(name) + "' is invalid (compilation failed)");
                return nullptr;
            }
            auto shader = dynamic_cast<const luisa::compute::Shader<dim, Args...> *>(
                it->second.kernel.get());
            if (shader == nullptr)
                logError("Shader '" + std::string(name) + "' has a mismatched prototype");
            return shader;
        }
    }
#ifdef _DEBUG
    {
        std::lock_guard dllLock(_dllMutex);
        auto dllIt = _dllLoaders.find(std::string(name));
        if (dllIt != _dllLoaders.end() && dllIt->second->isLoaded()) {
            return dllIt->second->template getShader<dim, Args...>();
        }
    }
#endif
    logError("Shader not found: " + std::string(name));
    return nullptr;
}

template<uint dim, typename... Args>
ShaderHandle<dim, Args...> ShaderManager::resolve(std::string_view name) const noexcept {
    ShaderHandle<dim, Args...> handle;
    handle._name = name;
    handle._shader = _lookup_shader<dim, Args...>(name);
    handle._generation = _generation;
    return handle;
}

template<uint dim, typename... Args, typename... CallArgs>
auto ShaderManager::shader(ShaderHandle<dim, Args...> &handle, CallArgs &&...call_args) const {
    // Header/lib ABI guard (prebuilt distro): fires once per process on the
    // first dispatch through this path; a no-op compare when macros match.
    newtype::detail::engineAbiCheck();
    if (handle._shader == nullptr || handle._generation != _generation) {
        handle._shader = _lookup_shader<dim, Args...>(handle._name);
        handle._generation = _generation;
        if (handle._shader == nullptr)
            throw std::runtime_error(
                "ShaderManager: cannot resolve shader '" + handle._name + "'");
    }
    return (*handle._shader)(std::forward<CallArgs>(call_args)...);
}

template<typename CallableT>
void ShaderManager::registerCallable(std::string_view name, CallableT&& callable) {
    std::lock_guard lock(_callablesMutex);
    _callables[std::string(name)] = std::forward<CallableT>(callable);
    CI_LOG_D("ShaderManager: Registered callable '" << name << "'");
}

} // namespace newtype::core
