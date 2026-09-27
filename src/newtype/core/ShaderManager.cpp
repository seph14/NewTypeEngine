#include "newtype/core/ShaderManager.h"
#include "newtype/core/Renderer.h"
#include "cinder/Log.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <filesystem>

// Runtime-only includes
#ifdef _DEBUG
#include "newtype/runtime/DLLHotReload.h"
#endif

namespace newtype::core {

// Static member initialization
luisa::compute::Device* ShaderManager::sDevice = nullptr;

//==============================================================================
// ShaderManager Implementation
//==============================================================================

ShaderManager& ShaderManager::instance() {
    static ShaderManager instance;
    return instance;
}

ShaderManager::ShaderManager() {
}

ShaderManager::~ShaderManager() {
#ifdef _DEBUG
    stopWatchThread();
#endif
    CI_LOG_D("ShaderManager: Destroyed");
}

void ShaderManager::clear() {
    {
        std::lock_guard lock(_shadersMutex);
        _shaders.clear();
        ++_generation;
    }
#ifdef _DEBUG
    std::lock_guard dllLock(_dllMutex);
    _dllLoaders.clear();
#endif
    CI_LOG_D("ShaderManager: Cleared all shaders");
}

bool ShaderManager::hasShader(std::string_view name) const {
    {
        std::lock_guard lock(_shadersMutex);

        // Check static shaders
        auto sIt = _shaders.find(std::string(name));
        if (sIt != _shaders.end() && sIt->second.valid)
            return true;
    }

#ifdef _DEBUG
    {
        std::lock_guard dllLock(_dllMutex);

        // Check DLL shaders
        auto dIt = _dllLoaders.find(std::string(name));
        if (dIt != _dllLoaders.end() && dIt->second->isValid())
            return true;
    }
#endif

    return false;
}

std::vector<std::string> ShaderManager::shaderNames() const {
    std::vector<std::string> names;

    {
        std::lock_guard lock(_shadersMutex);
        names.reserve(_shaders.size());
        for (const auto& [name, _] : _shaders)
            names.push_back(name);
    }

#ifdef _DEBUG
    {
        std::lock_guard dllLock(_dllMutex);

        // Add DLL shader names
        names.reserve(names.size() + _dllLoaders.size());
        for (const auto& [name, _] : _dllLoaders)
            names.push_back(name);
    }
#endif

    return names;
}

void ShaderManager::launch() {
#ifdef _DEBUG
    std::lock_guard dllLock(_dllMutex);
    if (_running)
        return;
    _running     = true;
    _watchThread = std::thread(&ShaderManager::watchThreadFunc, this);
#endif
}

//==============================================================================
// Runtime-Only Implementation (_DEBUG — machinery for DLL shader hot reload)
//==============================================================================

#ifdef _DEBUG
void ShaderManager::setReloadCallback(ReloadCallback callback) {
    std::lock_guard dllLock(_dllMutex);
    for (auto& shaderPtr : _dllLoaders)
        shaderPtr.second->setReloadCallback(std::move(callback));
}

void ShaderManager::stopWatchThread() {
    _running = false;
    if (_watchThread.joinable())
        _watchThread.join();
    CI_LOG_D("ShaderManager: File watch thread stopped");
}

void ShaderManager::watchThreadFunc() {
    constexpr auto check_interval = std::chrono::milliseconds(500);

    while (_running) {
        // Watch thread ONLY polls files. The heavy reload (destroy + rebuild
        // + compile) runs on the main thread via processPendingReloads, so
        // that the shader is never destroyed while the render loop has
        // in-flight stream commands holding its pointer.
        if (sDevice) {
            std::lock_guard dllLock(_dllMutex);
            for (auto& shaderPtr : _dllLoaders) {
                if (shaderPtr.second->checkForChanges())
                    CI_LOG_D("ShaderManager: " <<
                        shaderPtr.second->getShaderName()
                        << " changed - will reload on main thread");
            }
        }

        std::this_thread::sleep_for(check_interval);
    }
}

void ShaderManager::processPendingReloads(luisa::compute::Stream& stream) {
    if (!sDevice) return;
    std::lock_guard dllLock(_dllMutex);
    if (_dllLoaders.empty()) return;
    bool reloaded = false;
    for (auto& shaderPtr : _dllLoaders) {
        if (shaderPtr.second->performReload(*sDevice, stream)) {
            reloaded = true;
            CI_LOG_I("ShaderManager: " <<
                shaderPtr.second->getShaderName() << " reloaded via DLL");
        }
    }
    // Reloaded shader objects were destroyed and rebuilt — invalidate every
    // outstanding ShaderHandle so its next dispatch re-resolves.
    if (reloaded) ++_generation;
}

#endif // _DEBUG

//==============================================================================
// Utility Functions
//==============================================================================

std::filesystem::file_time_type ShaderManager::getLastModified(
    const std::filesystem::path& path) {
    return std::filesystem::last_write_time(path);
}

void ShaderManager::logError(std::string_view msg) {
    CI_LOG_E("ShaderManager: " << msg);
    std::cerr << "[ShaderManager ERROR] " << msg << std::endl;
}

} // namespace newtype::core
