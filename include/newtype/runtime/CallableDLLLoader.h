#pragma once

#include <string>
#include <filesystem>
#include <functional>
#include <any>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif

namespace newtype::runtime {

/// Host-side function pointer types matching the DLL's CallableRegisterFn/CallableClearFn.
using HostRegisterFn = std::uint32_t(*)(const char* name, std::any resolve);
using HostClearFn   = void(*)();

/// ABI v2 host-side param registration (matches the DLL's ParamRegisterFn —
/// ResolverParamDesc is POD and re-declared here so the engine doesn't include
/// the DLL API header; keep fields in lockstep).
struct HostResolverParamDesc {
    const char* name;
    float       min_v;
    float       max_v;
    float       def_v;
};
using HostParamRegisterFn = std::uint32_t(*)(const char* callable,
                                             const HostResolverParamDesc* descs,
                                             std::uint32_t count);

/**
 * @brief Minimal DLL loader for custom material callable hot-reload.
 *
 * Loads DLLs that export `registerMaterialCallables` / `unregisterMaterialCallables`.
 * The DLL receives host-provided function pointers to register callables into
 * the main app's MaterialCallableTable (avoids shared singleton issues across
 * DLL boundaries).
 *
 * Usage (Debug_Runtime only):
 * @code
 *   CallableDLLLoader loader;
 *   loader.setHostFunctions(myRegisterFn, myClearFn);
 *   loader.load("path/to/CustomMaterialShader.dll", "path/to/source.cpp");
 *   // ... render loop ...
 *   if (loader.checkAndReload()) {
 *       pipeline.recompileAllShaders();
 *   }
 * @endcode
 */
class CallableDLLLoader {
public:
    CallableDLLLoader() = default;
    ~CallableDLLLoader();

    // Non-copyable
    CallableDLLLoader(const CallableDLLLoader&) = delete;
    CallableDLLLoader& operator=(const CallableDLLLoader&) = delete;

    /// Set the host-side registration functions (called from Pipeline).
    /// paramFn may be null: v2 DLLs then see a null ParamRegisterFn and must
    /// skip param registration (v1 behavior).
    void setHostFunctions(HostRegisterFn registerFn, HostClearFn clearFn,
                          HostParamRegisterFn paramFn = nullptr);

    /// Load the DLL and call registerMaterialCallables().
    /// @param dllPath    Path to the compiled DLL
    /// @param sourcePath Path to the source .cpp file to watch
    /// @return true if loaded and registered successfully
    bool load(const std::string& dllPath, const std::string& sourcePath);

    /// Unload the DLL (calls unregisterMaterialCallables first).
    void unload();

    /// Check if source changed, rebuild DLL, reload, re-register.
    /// Call from render loop. Returns true if DLL was rebuilt + reloaded.
    bool checkAndReload();

    /// Is the DLL currently loaded?
    [[nodiscard]] bool isLoaded() const noexcept { return _module != nullptr; }

    /// Get the last error message
    [[nodiscard]] const std::string& getLastError() const noexcept { return _lastError; }

private:
    // DLL-side function signatures (match CustomMaterialShaderAPI.h)
    using DLLRegisterFn    = void(*)(HostRegisterFn, HostClearFn);
    using DLLRegisterFn2   = void(*)(HostRegisterFn, HostParamRegisterFn, HostClearFn);
    using DLLUnregisterFn  = void(*)(HostClearFn);

    bool buildDLL();
    bool loadDLL();
    void unloadDLL();

    [[nodiscard]] static std::string findMSBuild();
    [[nodiscard]] static std::filesystem::file_time_type getLastWriteTime(
        const std::filesystem::path& path);
    void logError(std::string_view msg);

    HMODULE _module = nullptr;
    DLLRegisterFn   _dllRegisterFn   = nullptr;
    DLLRegisterFn2  _dllRegisterFn2  = nullptr;
    DLLUnregisterFn _dllUnregisterFn = nullptr;

    HostRegisterFn _hostRegisterFn = nullptr;
    HostClearFn    _hostClearFn    = nullptr;
    HostParamRegisterFn _hostParamFn = nullptr;

    std::string _dllPath;
    std::string _sourcePath;
    std::string _projectPath;
    std::string _lastError;

    std::filesystem::file_time_type _lastSourceTime;
    bool _initialized = false;
};

} // namespace newtype::runtime
