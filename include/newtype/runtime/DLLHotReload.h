#pragma once

#include <luisa/luisa-compute.h>
#include <cinder/Log.h>
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

namespace newtype::runtime {

/**
 * @brief Callback type for shader reload events
 *
 * Called when a shader is successfully reloaded. The app can
 * use this to reset accumulation buffers or update state.
 */
using ReloadCallback = std::function<void(std::string_view)>;

/**
 * @brief Manages hot-reloading of shader DLLs
 *
 * This class handles:
 * - Loading/unloading DLLs at runtime
 * - Building DLLs using MSBuild when source changes
 * - Proper cleanup of Windows DLL resources
 * - Thread-safe operations
 *
 * Usage:
 * @code
 *   DLLHotReload loader("PathTracerShader");
 *   loader.load(device);
 *   auto* shader = loader.getShader<PathTracerShaderType>();
 *
 *   // In update loop:
 *   if (loader.checkAndReload(device)) {
 *       // Shader was reloaded, reset accumulation
 *       frame_index = 0;
 *   }
 * @endcode
 */
class DLLHotReload {
public:
    /**
     * @brief Construct a DLL hot-reload manager
     *
     * @param shaderName Name/identifier of the shader (for logging)
     * @param dllPath Path to the DLL file (relative or absolute)
     * @param projectPath Path to the .vcxproj for building
     * @param sourcePath Path to the .cpp source file to watch for changes
     */
    DLLHotReload(
        std::string shaderName,
        std::string dllPath,
        std::string projectPath,
        std::string sourcePath);

    ~DLLHotReload();

    // Non-copyable, non-movable
    DLLHotReload(const DLLHotReload&) = delete;
    DLLHotReload& operator=(const DLLHotReload&) = delete;
    DLLHotReload(DLLHotReload&&) = delete;
    DLLHotReload& operator=(DLLHotReload&&) = delete;

    /**
     * @brief Load the DLL and create the shader
     *
     * @param device LuisaCompute device for shader compilation
     * @return true if loaded successfully
     */
    bool load(luisa::compute::Device& device);

    /**
     * @brief Unload the DLL and cleanup resources
     */
    void unload();

    /**
     * @brief Poll the source file for changes (watch thread).
     *
     * Lightweight: stat the source file, compare to the last-seen mtime, and
     * set an internal pending-reload flag if it has advanced. The actual
     * destroy+rebuild+compile work happens later when the main thread calls
     * performReload(). This split exists so the heavy reload runs on the
     * render thread, where the shader is guaranteed not to be in use by an
     * in-flight stream command.
     *
     * @return true if a change was detected (flag set)
     */
    bool checkForChanges();

    /**
     * @brief Perform the actual reload if one is pending (main thread).
     *
     * Sequence: synchronize the stream → destroy old shader → rebuild DLL via
     * MSBuild → load new DLL → compile new shader. Stream sync is mandatory:
     * pending dispatch commands hold the shader pointer as a uint64 handle
     * (LuisaCompute LCCmdBuffer.cpp reinterpret_cast<ComputeShader*>), and
     * destroy_shader frees the object immediately with no deferred queue —
     * so we must drain in-flight commands before deleting.
     *
     * @param device LuisaCompute device
     * @param stream Stream to synchronize before destroying the old shader
     * @return true if a reload was performed
     */
    bool performReload(luisa::compute::Device& device, luisa::compute::Stream& stream);

    /**
     * @brief Get the loaded shader pointer
     *
     * @tparam ShaderT The shader type (e.g., PathTracerShaderType)
     * @return Pointer to the shader, or nullptr if not loaded
     */
    template<uint dim, typename... Args>
    [[nodiscard]] auto getShader() const {
        return dynamic_cast<const luisa::compute::Shader<dim, Args...>*>(_shaderPtr.get());
    }

    /**
     * @brief Check if the DLL is currently loaded
     */
    [[nodiscard]] bool isLoaded() const noexcept { return _module != nullptr; }
    [[nodiscard]] bool isValid() const noexcept { return _shaderPtr != nullptr; }

    /**
     * @brief Get the shader name
     */
    [[nodiscard]] const std::string& getShaderName() const noexcept { return _shaderName; }

    /**
     * @brief Set a callback to be invoked when the shader is reloaded
     *
     * @param callback Function to call after successful reload
     */
    void setReloadCallback(ReloadCallback callback) { _reloadCallback = std::move(callback); }

    /**
     * @brief Enable/disable automatic rebuilding
     *
     * When disabled, checkAndReload() will only detect changes
     * but won't trigger MSBuild.
     */
    void setAutoRebuild(bool enabled) noexcept { _autoRebuild = enabled; }

    /**
     * @brief Get the last error message
     */
    [[nodiscard]] const std::string& getLastError() const noexcept { return _lastError; }

private:
    // Build the DLL using MSBuild
    bool buildDLL();

    // Get the last write time of a file
    [[nodiscard]] std::filesystem::file_time_type getLastWriteTime(
        const std::filesystem::path& path) const;

    // Log an error message
    void logError(std::string_view msg);

    // Find MSBuild executable
    [[nodiscard]] static std::string findMSBuild();

    // Members
    std::string _shaderName;
    std::string _dllPath;
    std::string _projectPath;
    std::string _sourcePath;
    std::string _lastError;

    // DLL module handle
    HMODULE _module = nullptr;

    // Shader pointer (owned by the DLL/Device)
    luisa::unique_ptr<luisa::compute::Resource> _shaderPtr = nullptr;

    // Function pointers to DLL exports
    // Returns Resource* (Shader<dim, ...> inherits from Resource)
    using CreateFunc        = luisa::compute::Resource* (*)(luisa::compute::Device&);
    using DestroyFunc       = void (*)(luisa::compute::Resource*);
    CreateFunc _createFunc  = nullptr;
    DestroyFunc _destroyFunc = nullptr;

    // File tracking
    std::filesystem::file_time_type _lastDllModTime;
    std::unordered_map<std::string, std::filesystem::file_time_type> _sourceModTimes;
    bool _initialized = false;

    // Set by the watch thread (checkForChanges), consumed and cleared by the
    // main thread (performReload). Atomic so the two threads don't need to
    // coordinate via the mutex just to read this flag.
    std::atomic<bool> _pendingReload{false};

    // Settings
    bool _autoRebuild = true;
    ReloadCallback _reloadCallback;

    // Thread safety
    mutable std::mutex _mutex;

    // Generate function names from shader name
    // e.g., "PathTracer" -> "createPathTracer", "destroyPathTracer"
    [[nodiscard]] std::string getCreateFuncName() const;
    [[nodiscard]] std::string getDestroyFuncName() const;
    [[nodiscard]] std::string getProjectTargetName() const;
};

} // namespace newtype::runtime
