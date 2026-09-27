#include "newtype/runtime/DLLHotReload.h"
#include "cinder/app/App.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdio>

// Windows-specific: _popen/_pclose declarations
#ifdef _WIN32
extern "C" {
    FILE* _popen(const char* command, const char* mode);
    int _pclose(FILE* stream);
}
#endif

namespace newtype::runtime {

//==============================================================================
// DLLHotReload Implementation
//==============================================================================

DLLHotReload::DLLHotReload(
    std::string shaderName,
    std::string dllPath,
    std::string projectPath,
    std::string sourcePath)
    : _shaderName   (std::move(shaderName))
    , _dllPath      (std::move(dllPath))
    , _projectPath  (std::move(projectPath))
    , _sourcePath   (std::move(sourcePath)) {

    // Resolve relative paths to absolute
    // Paths are expected to be relative to the working directory (vc2022/x64/Debug_Runtime/)
    auto currentPath = std::filesystem::current_path();

    // Resolve DLL path
    {
        auto absPath = std::filesystem::weakly_canonical(currentPath / _dllPath);
        if (std::filesystem::exists(absPath)) 
            _dllPath = absPath.string();
    }

    // Resolve project path
    {
        auto absPath = std::filesystem::weakly_canonical(currentPath / _projectPath);
        if (std::filesystem::exists(absPath)) 
            _projectPath = absPath.string();
    }

    // Resolve source path
    {
        auto absPath = std::filesystem::weakly_canonical(currentPath / _sourcePath);
        if (std::filesystem::exists(absPath)) 
            _sourcePath = absPath.string();
    }

    // Initialize file times
    if (std::filesystem::exists(_dllPath)) 
        _lastDllModTime = getLastWriteTime(_dllPath);
    
    CI_LOG_D("DLLHotReload: Initialized for '" << _shaderName
             << "' (DLL: " << _dllPath << ", Source: " << _sourcePath << ")");
}

DLLHotReload::~DLLHotReload() {
    unload();
}

bool DLLHotReload::load(luisa::compute::Device& device) {
    std::lock_guard lock(_mutex);

    // Already loaded
    if (_module != nullptr) {
        CI_LOG_W("DLLHotReload: DLL already loaded for '" << _shaderName << "'");
        return true;
    }

    // Check if DLL exists
    if (!std::filesystem::exists(_dllPath)) {
        logError("DLL not found: " + _dllPath);
        return false;
    }

    // Load the DLL
    _module = LoadLibraryA(_dllPath.c_str());
    if (_module == nullptr) {
        DWORD error = GetLastError();
        std::ostringstream ss;
        ss << "Failed to load DLL '" << _dllPath << "' (error code: " << error << ")";
        logError(ss.str());
        return false;
    }

    // Get function pointers (dynamic based on shader name)
    auto createFuncName  = getCreateFuncName();
    auto destroyFuncName = getDestroyFuncName();

    _createFunc = reinterpret_cast<CreateFunc>(
        GetProcAddress(_module, createFuncName.c_str()));
    _destroyFunc = reinterpret_cast<DestroyFunc>(
        GetProcAddress(_module, destroyFuncName.c_str()));

    if (_createFunc == nullptr || _destroyFunc == nullptr) {
        logError("Failed to find exported functions in DLL");
        FreeLibrary(_module);
        _module      = nullptr;
        _createFunc  = nullptr;
        _destroyFunc = nullptr;
        return false;
    }

    // Create the shader (CreateFunc returns Resource*)
    _shaderPtr = luisa::unique_ptr<luisa::compute::Resource>(
        static_cast<luisa::compute::Resource*>(_createFunc(device)));
    if (_shaderPtr == nullptr) {
        logError("DLL create function returned nullptr");
        FreeLibrary(_module);
        _module      = nullptr;
        _createFunc  = nullptr;
        _destroyFunc = nullptr;
        return false;
    }

    // Update timestamp
    _lastDllModTime = getLastWriteTime(_dllPath);

    CI_LOG_I("DLLHotReload: Successfully loaded '" << _shaderName << "' from " << _dllPath);
    return true;
}

void DLLHotReload::unload() {
    std::lock_guard lock(_mutex);

    if (_module == nullptr) 
        return;

    // Destroy the shader
    if (_destroyFunc != nullptr && _shaderPtr != nullptr) {
        auto* p = _shaderPtr.release();   // give up ownership so the deleter won't double-free
        _destroyFunc(p);                   // DLL deletes p (calls ~ShaderBase → device->destroy_shader)
    }
    _shaderPtr   = nullptr;
    _createFunc  = nullptr;
    _destroyFunc = nullptr;

    // Unload the DLL
    FreeLibrary(_module);
    _module = nullptr;

    CI_LOG_D("DLLHotReload: Unloaded '" << _shaderName << "'");
}

bool DLLHotReload::checkForChanges() {
    std::lock_guard lock(_mutex);

    // Initialize source file mod times on first call
    if (!_initialized) {
        if (std::filesystem::exists(_sourcePath)) {
            _sourceModTimes[_sourcePath] = getLastWriteTime(_sourcePath);
            _initialized = true;
            CI_LOG_D("DLLHotReload: Watching source file: " << _sourcePath);
        }
    }
    if (!_initialized) return false;

    // Poll the source file mtime and set the pending-reload flag if it has
    // advanced. The actual reload happens later on the main thread via
    // performReload — never inside this watch-thread call.
    auto currentTime = getLastWriteTime(_sourcePath);
    auto it = _sourceModTimes.find(_sourcePath);
    if (it != _sourceModTimes.end() && currentTime > it->second) {
        it->second = currentTime;
        _pendingReload.store(true, std::memory_order_release);
        CI_LOG_D("DLLHotReload: Detected change in source file: " << _shaderName);
        return true;
    }
    return false;
}

bool DLLHotReload::performReload(luisa::compute::Device& device, luisa::compute::Stream& stream) {
    // Fast path: atomic check without taking the mutex.
    if (!_pendingReload.load(std::memory_order_acquire)) return false;

    std::lock_guard lock(_mutex);
    // Re-check under the mutex in case two callers race here.
    if (!_pendingReload.load(std::memory_order_relaxed)) return false;
    _pendingReload.store(false, std::memory_order_release);

    // CRITICAL: drain in-flight GPU commands before destroying the old shader.
    // Dispatch commands hold the shader pointer as a uint64 handle
    // (LuisaCompute LCCmdBuffer.cpp: reinterpret_cast<ComputeShader*>(cmd->handle())),
    // and destroy_shader deletes the object synchronously with no deferred
    // queue — any pending command that resolves the handle after we delete
    // would touch freed memory and stall or crash the device.
    stream << luisa::compute::synchronize();

    // CRITICAL: On Windows, you CANNOT overwrite a DLL while it's loaded!
    // The sequence must be:
    // 1. Unload the DLL (releases file lock)
    // 2. Build the new DLL (now MSBuild can overwrite)
    // 3. Load the new DLL

    // Step 1: Unload the DLL FIRST
    if (_module != nullptr) {
        CI_LOG_D("DLLHotReload: Unloading old DLL to allow rebuild...");

        // Destroy old shader
        if (_destroyFunc != nullptr && _shaderPtr != nullptr) {
            auto* p = _shaderPtr.release();   // give up ownership so the deleter won't double-free
            _destroyFunc(p);                   // DLL deletes p (calls ~ShaderBase → device->destroy_shader)
        }
        _shaderPtr  = nullptr;
        _createFunc = nullptr;
        _destroyFunc= nullptr;

        // Unload DLL - this releases the file lock
        FreeLibrary(_module);
        _module = nullptr;

        // Give Windows time to release the file lock
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }

    // Step 2: Build the new DLL (now the file is not locked)
    if (_autoRebuild) {
        CI_LOG_D("DLLHotReload: Building new DLL...");

        // Backup old DLL before build so we can restore on failure. COPY, not
        // rename: the original stays in place, keeping the linker's
        // incremental state valid — renaming forced a full link (full PDB
        // rewrite) every reload, which collides with locked/corrupt PDBs.
        auto backupPath = _dllPath + ".bak";
        bool hadBackup = false;
        if (std::filesystem::exists(_dllPath)) {
            std::error_code ec;
            std::filesystem::copy_file(_dllPath, backupPath,
                                       std::filesystem::copy_options::overwrite_existing, ec);
            if (!ec) {
                hadBackup = true;
                CI_LOG_D("DLLHotReload: Backed up DLL to " << backupPath);
            } else {
                CI_LOG_W("DLLHotReload: Failed to backup old DLL: " << ec.message());
            }
        }

        if (!buildDLL()) {
            logError("Failed to build DLL — restoring previous version");
            // Restore backup so we can reload the old version
            if (hadBackup && std::filesystem::exists(backupPath)) {
                std::error_code ec;
                std::filesystem::copy_file(backupPath, _dllPath,
                                           std::filesystem::copy_options::overwrite_existing, ec);
                if (ec) CI_LOG_W("DLLHotReload: Failed to restore backup: " << ec.message());
            }
            load(device);
            return false;
        }

        // Clean up backup on success
        if (hadBackup && std::filesystem::exists(backupPath)) {
            std::error_code ec;
            std::filesystem::remove(backupPath, ec);
        }

        // Wait for file system to settle after build
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Step 3: Load the new DLL
    CI_LOG_D("DLLHotReload: Loading new DLL...");

    if (!std::filesystem::exists(_dllPath)) {
        //logError("New DLL not found after build: " << _dllPath.c_str());
        return false;
    }

    _module = LoadLibraryA(_dllPath.c_str());
    if (_module == nullptr) {
        DWORD error = GetLastError();
        std::ostringstream ss;
        ss << "Failed to load new DLL (error code: " << error << ")";
        logError(ss.str());
        return false;
    }

    // Get new function pointers (dynamic based on shader name)
    auto createFuncName  = getCreateFuncName();
    auto destroyFuncName = getDestroyFuncName();

    _createFunc = reinterpret_cast<CreateFunc>(
        GetProcAddress(_module, createFuncName.c_str()));
    _destroyFunc = reinterpret_cast<DestroyFunc>(
        GetProcAddress(_module, destroyFuncName.c_str()));

    if (_createFunc == nullptr || _destroyFunc == nullptr) {
        logError("Failed to find exported functions in new DLL");
        FreeLibrary(_module);
        _module = nullptr;
        return false;
    }

    // Create new shader (CreateFunc returns Resource*)
    _shaderPtr = luisa::unique_ptr<luisa::compute::Resource>(
        static_cast<luisa::compute::Resource*>(_createFunc(device)));
    if (_shaderPtr == nullptr) {
        logError("New DLL create function returned nullptr");
        FreeLibrary(_module);
        _module = nullptr;
        return false;
    }

    // Update timestamp
    _lastDllModTime = getLastWriteTime(_dllPath);

    CI_LOG_I("DLLHotReload: Successfully reloaded '" << _shaderName << "'!");

    // Invoke callback if set
    if (_reloadCallback) {
        try {
            _reloadCallback(_shaderName);
        } catch (const std::exception& e) {
            CI_LOG_E("DLLHotReload: Reload callback threw exception: " << e.what());
        }
    }

    return true;
}

bool DLLHotReload::buildDLL() {
    CI_LOG_D("DLLHotReload: Building shader DLL...");

    // Find MSBuild
    std::string msbuildPath = findMSBuild();
    if (msbuildPath.empty()) {
        logError("MSBuild not found. Please add MSBuild to PATH or install Visual Studio.");
        return false;
    }

    // Log paths for debugging
    //CI_LOG_I("DLLHotReload: Project path: " << _projectPath);
    CI_LOG_D("DLLHotReload: DLL path: " << _dllPath);
    CI_LOG_D("DLLHotReload: Source path: " << _sourcePath);
    //CI_LOG_I("DLLHotReload: Current directory: " << std::filesystem::current_path().string());

    // The project path should already be absolute from constructor
    auto projectPath = std::filesystem::path(_projectPath);

    if (!std::filesystem::exists(projectPath)) {
        logError("Project file not found: " + projectPath.string());
        return false;
    }
    CI_LOG_D("DLLHotReload: Project path: " << projectPath.string());

    // Find the solution directory (vc2022 folder)
    // projectPath is at: <repo>/runtime_shaders/SimpleTestShader/SimpleTestShader.vcxproj
    // We need to go up 3 levels: vcxproj → shader folder → runtime_shaders → (project root) → vc2022
    auto solutionDir = projectPath.parent_path().parent_path().parent_path() / "vc2022";
    solutionDir = std::filesystem::weakly_canonical(solutionDir);
    CI_LOG_D("DLLHotReload: Solution directory: " << solutionDir.string());

    // Discover the .sln file dynamically (works across projects)
    std::filesystem::path slnPath;
    if (std::filesystem::exists(solutionDir) && std::filesystem::is_directory(solutionDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(solutionDir)) {
            if (entry.path().extension() == ".sln") {
                slnPath = entry.path();
                break;
            }
        }
    }
    if (slnPath.empty() || !std::filesystem::exists(slnPath))
        CI_LOG_W("DLLHotReload: Solution file not found in: " << solutionDir.string());
    

    // Build command line - build the SOLUTION file, not the project file
    // This ensures $(SolutionDir) and output directories resolve correctly.
    // INCREMENTAL build (no :Rebuild): the reload flow has already unloaded the
    // DLL by this point, so MSBuild can overwrite it; the source file just
    // changed, so the up-to-date check recompiles the single TU and relinks.
    // A full :Rebuild paid ~9-10s per hot reload for a one-file project.
    std::ostringstream cmdLine;
    cmdLine << "\"" << msbuildPath << "\""
             << " \"" << slnPath.string() << "\""
             << " /t:" << getProjectTargetName()
             << " /p:Configuration=Debug_Runtime"
             << " /p:Platform=x64"
             << " /v:minimal"
             << " /nologo";

    //CI_LOG_D("Command: " << cmdLine.str());

    auto start = std::chrono::steady_clock::now();

    // Use Windows CreateProcess for more reliable execution
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    SECURITY_ATTRIBUTES sa;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    ZeroMemory(&sa, sizeof(sa));

    // Set up pipes for stdout/stderr capture
    HANDLE hReadPipe, hWritePipe;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        logError("Failed to create pipe for MSBuild output");
        return false;
    }

    // Ensure the read handle is not inherited
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    // Redirect stdout and stderr to the pipe
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;
    si.wShowWindow = SW_HIDE;

    // Create the command in a mutable buffer
    std::string cmdLineStr = cmdLine.str();
    std::vector<char> cmdLineBuf(cmdLineStr.begin(), cmdLineStr.end());
    cmdLineBuf.push_back('\0');

    // Create process
    BOOL success = CreateProcessA(
        NULL,
        cmdLineBuf.data(),
        NULL,
        NULL,
        TRUE,  // Inherit handles
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &si,
        &pi
    );

    // Close write end of pipe immediately after process creation
    CloseHandle(hWritePipe);

    if (!success) {
        DWORD error = GetLastError();
        std::ostringstream ss;
        ss << "Failed to execute MSBuild (error code: " << error << ")";
        logError(ss.str());
        CloseHandle(hReadPipe);
        return false;
    }
    
    // Capture output
    std::string output;
    char buffer[1024];
    DWORD bytesRead;
    size_t totalRead = 0;

    while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        output += buffer;
        totalRead += bytesRead;
        // Limit output size
        if (totalRead > 50000) {
            output += "\n... (output truncated)";
            break;
        }
    }
    CloseHandle(hReadPipe);

    // Wait for process to complete
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Get exit code
    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    if (exitCode != 0) {
        CI_LOG_E("DLLHotReload: MSBuild failed (exit code: " << exitCode << ")");
        if (!output.empty()) 
            CI_LOG_E("Output:\n" << output);
        else 
            CI_LOG_E("No output captured");
        _lastError = "Build failed (exit code: " + std::to_string(exitCode) + ")";
        return false;
    }

    CI_LOG_D("DLLHotReload: Build completed in " << elapsed << "ms");
    /*if (!output.empty()) {
        CI_LOG_D("MSBuild output:\n" << output);
    }*/

    // Verify DLL was actually created/updated
    if (!std::filesystem::exists(_dllPath)) {
        logError("DLL not found after build: " + _dllPath);
        // List all DLLs in the output directory
        auto outputDir = std::filesystem::path(_dllPath).parent_path();
        if (std::filesystem::exists(outputDir)) {
            CI_LOG_D("DLLHotReload: Listing files in output directory: " << outputDir.string());
            for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
                CI_LOG_D("  - " << entry.path().filename().string()
                         << " (time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                entry.last_write_time().time_since_epoch()).count() << "ms)");
            }
        }
        return false;
    }

    return true;
}

std::filesystem::file_time_type DLLHotReload::getLastWriteTime(
    const std::filesystem::path& path) const {

    try {
        return std::filesystem::last_write_time(path);
    } catch (const std::exception& e) {
        CI_LOG_W("DLLHotReload: Failed to get write time for " << path.string()
                 << ": " << e.what());
        return std::filesystem::file_time_type{};
    }
}

void DLLHotReload::logError(std::string_view msg) {
    _lastError = msg;
    CI_LOG_E("DLLHotReload [" << _shaderName << "]: " << msg);
    std::cerr << "[DLLHotReload ERROR] " << _shaderName << ": " << msg << std::endl;
}

std::string DLLHotReload::findMSBuild() {
    // Try to find MSBuild in standard locations
    const std::vector<std::string> searchPaths = {
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\MSBuild\\Current\\Bin\\MSBuild.exe",
    };

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            CI_LOG_D("DLLHotReload: Found MSBuild at " << path);
            return path;
        }
    }

    // Try from PATH
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv) {
        std::istringstream pathStream(pathEnv);
        std::string dir;
        while (std::getline(pathStream, dir, ';')) {
            auto msbuild = std::filesystem::path(dir) / "MSBuild.exe";
            if (std::filesystem::exists(msbuild)) {
                CI_LOG_D("DLLHotReload: Found MSBuild in PATH at " << msbuild.string());
                return msbuild.string();
            }
        }
    }

    return "";
}

//==============================================================================
// Dynamic Function Name Generation
//==============================================================================

std::string DLLHotReload::getCreateFuncName() const {
    return "create" + _shaderName;
}

std::string DLLHotReload::getDestroyFuncName() const {
    return "destroy" + _shaderName;
}

std::string DLLHotReload::getProjectTargetName() const {
    // If shader name already ends with "Shader", return as-is
    // Otherwise append "Shader" for the project target name
    if (_shaderName.length() >= 7 &&
        _shaderName.substr(_shaderName.length() - 7) == "Shader") {
        return _shaderName;
    }
    return _shaderName + "Shader";
}

} // namespace newtype::runtime
