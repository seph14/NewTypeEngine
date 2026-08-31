#include "newtype/runtime/CallableDLLLoader.h"
#include "cinder/Log.h"
#include <chrono>
#include <thread>
#include <sstream>
#include <cstdio>

namespace newtype::runtime {

//==============================================================================
// CallableDLLLoader Implementation
//==============================================================================

CallableDLLLoader::~CallableDLLLoader() {
    unload();
}

void CallableDLLLoader::setHostFunctions(HostRegisterFn registerFn, HostClearFn clearFn) {
    _hostRegisterFn = registerFn;
    _hostClearFn    = clearFn;
}

bool CallableDLLLoader::load(const std::string& dllPath, const std::string& sourcePath) {
    if (!_hostRegisterFn || !_hostClearFn) {
        CI_LOG_E("CallableDLLLoader: Host functions not set — call setHostFunctions() first");
        return false;
    }

    // Resolve paths relative to current working directory
    auto cwd = std::filesystem::current_path();

    _dllPath    = std::filesystem::weakly_canonical(cwd / dllPath).string();
    _sourcePath = std::filesystem::weakly_canonical(cwd / sourcePath).string();

    // Derive project path from source path (same directory, same stem, .vcxproj)
    auto srcDir  = std::filesystem::path(_sourcePath).parent_path();
    auto stem    = std::filesystem::path(_sourcePath).stem().string();
    auto vcxproj = srcDir / (stem + ".vcxproj");
    _projectPath = std::filesystem::weakly_canonical(vcxproj).string();

    CI_LOG_I("CallableDLLLoader: Initialized (DLL: " << _dllPath
             << ", Source: " << _sourcePath << ")");

    return loadDLL();
}

void CallableDLLLoader::unload() {
    unloadDLL();
}

bool CallableDLLLoader::loadDLL() {
    if (!std::filesystem::exists(_dllPath)) {
        logError("DLL not found: " + _dllPath);
        return false;
    }

    _module = LoadLibraryA(_dllPath.c_str());
    if (!_module) {
        DWORD err = GetLastError();
        std::ostringstream ss;
        ss << "Failed to load DLL (error: " << err << ")";
        logError(ss.str());
        return false;
    }

    _dllRegisterFn = reinterpret_cast<DLLRegisterFn>(
        GetProcAddress(_module, "registerMaterialCallables"));
    _dllUnregisterFn = reinterpret_cast<DLLUnregisterFn>(
        GetProcAddress(_module, "unregisterMaterialCallables"));

    if (!_dllRegisterFn) {
        logError("registerMaterialCallables not found in DLL");
        FreeLibrary(_module);
        _module = nullptr;
        return false;
    }

    // Call register with host-provided function pointers
    _dllRegisterFn(_hostRegisterFn, _hostClearFn);
    CI_LOG_I("CallableDLLLoader: Registered custom material callables");
    return true;
}

void CallableDLLLoader::unloadDLL() {
    if (!_module) return;

    // Unregister before unloading
    if (_dllUnregisterFn && _hostClearFn) {
        _dllUnregisterFn(_hostClearFn);
        CI_LOG_I("CallableDLLLoader: Unregistered custom material callables");
    }

    _dllRegisterFn   = nullptr;
    _dllUnregisterFn = nullptr;

    FreeLibrary(_module);
    _module = nullptr;
}

bool CallableDLLLoader::checkAndReload() {
    // Initialize source file timestamp on first call
    if (!_initialized) {
        if (std::filesystem::exists(_sourcePath)) {
            _lastSourceTime = getLastWriteTime(_sourcePath);
            _initialized = true;
            // Auto-build if DLL doesn't exist yet
            if (!_module && !std::filesystem::exists(_dllPath)) {
                CI_LOG_I("CallableDLLLoader: DLL not found, building...");
                if (buildDLL()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    if (loadDLL()) {
                        CI_LOG_I("CallableDLLLoader: Successfully built and loaded callable DLL");
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Check for source change
    auto currentTime = getLastWriteTime(_sourcePath);
    if (currentTime <= _lastSourceTime) return false;

    _lastSourceTime = currentTime;
    CI_LOG_I("CallableDLLLoader: Detected source change, rebuilding...");

    // Unload old DLL (releases file lock)
    unloadDLL();

    // Give Windows time to release the file lock
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Backup old DLL before build so we can restore on failure
    auto backupPath = _dllPath + ".bak";
    bool hadBackup = false;
    if (std::filesystem::exists(_dllPath)) {
        std::error_code ec;
        std::filesystem::rename(_dllPath, backupPath, ec);
        if (!ec) {
            hadBackup = true;
            CI_LOG_I("CallableDLLLoader: Backed up DLL to " << backupPath);
        } else {
            CI_LOG_W("CallableDLLLoader: Failed to backup old DLL: " << ec.message());
        }
    }

    // Rebuild
    if (!buildDLL()) {
        logError("Failed to rebuild callable DLL — restoring previous version");
        if (hadBackup && std::filesystem::exists(backupPath)) {
            std::error_code ec;
            std::filesystem::rename(backupPath, _dllPath, ec);
            if (ec) CI_LOG_W("CallableDLLLoader: Failed to restore backup: " << ec.message());
        }
        loadDLL();
        return false;
    }

    // Wait for file system to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Clean up backup on success
    if (hadBackup && std::filesystem::exists(backupPath)) {
        std::error_code ec;
        std::filesystem::remove(backupPath, ec);
    }

    // Load new DLL
    if (!loadDLL()) {
        logError("Failed to load rebuilt callable DLL");
        return false;
    }

    CI_LOG_I("CallableDLLLoader: Successfully reloaded custom material callables");
    return true;
}

bool CallableDLLLoader::buildDLL() {
    std::string msbuild = findMSBuild();
    if (msbuild.empty()) {
        logError("MSBuild not found");
        return false;
    }

    auto projectDir = std::filesystem::path(_projectPath);
    auto solutionDir = std::filesystem::weakly_canonical(
        projectDir.parent_path().parent_path().parent_path() / "vc2022");

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

    if (slnPath.empty() || !std::filesystem::exists(slnPath)) {
        logError("Solution not found in: " + solutionDir.string());
        return false;
    }

    auto targetName = projectDir.stem().string();

    std::ostringstream cmd;
    cmd << "\"" << msbuild << "\""
        << " \"" << slnPath.string() << "\""
        << " /t:" << targetName << ":Rebuild"
        << " /p:Configuration=Debug_Runtime"
        << " /p:Platform=x64"
        << " /v:minimal"
        << " /nologo";

    auto start = std::chrono::steady_clock::now();

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        logError("Failed to create pipe");
        return false;
    }
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWritePipe;
    si.hStdError  = hWritePipe;
    si.wShowWindow = SW_HIDE;

    std::string cmdStr = cmd.str();
    std::vector<char> cmdBuf(cmdStr.begin(), cmdStr.end());
    cmdBuf.push_back('\0');

    BOOL ok = CreateProcessA(NULL, cmdBuf.data(), NULL, NULL, TRUE,
                              CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(hWritePipe);

    if (!ok) {
        logError("Failed to execute MSBuild (error: " + std::to_string(GetLastError()) + ")");
        CloseHandle(hReadPipe);
        return false;
    }

    std::string output;
    char buf[1024];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        output += buf;
        if (output.size() > 50000) break;
    }
    CloseHandle(hReadPipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    if (exitCode != 0) {
        CI_LOG_E("CallableDLLLoader: Build failed (" << elapsed << "ms, exit=" << exitCode << ")");
        _lastError = "Build failed (exit " + std::to_string(exitCode) + ")";
        return false;
    }

    CI_LOG_I("CallableDLLLoader: Build succeeded (" << elapsed << "ms)");
    return true;
}

std::string CallableDLLLoader::findMSBuild() {
    const std::vector<std::string> paths = {
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\MSBuild\\Current\\Bin\\MSBuild.exe",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\MSBuild\\Current\\Bin\\MSBuild.exe",
    };
    for (const auto& p : paths) {
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

std::filesystem::file_time_type CallableDLLLoader::getLastWriteTime(
    const std::filesystem::path& path) {
    try {
        return std::filesystem::last_write_time(path);
    } catch (...) {
        return {};
    }
}

void CallableDLLLoader::logError(std::string_view msg) {
    _lastError = msg;
    CI_LOG_E("CallableDLLLoader: " << msg);
}

} // namespace newtype::runtime
