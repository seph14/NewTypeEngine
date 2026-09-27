#include "newtype/runtime/CallableDLLLoader.h"
#include "newtype/render/CallableAbiVersion.h"
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

void CallableDLLLoader::setHostFunctions(
    HostRegisterFn registerFn, HostClearFn clearFn, HostParamRegisterFn paramFn) {
    _hostRegisterFn = registerFn;
    _hostClearFn    = clearFn;
    _hostParamFn    = paramFn;
}

bool CallableDLLLoader::load(const std::string& dllPath, const std::string& sourcePath) {
    if (!_hostRegisterFn || !_hostClearFn) {
        CI_LOG_E("CallableDLLLoader: Host functions not set - call setHostFunctions() first");
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

    CI_LOG_D("CallableDLLLoader: Initialized (DLL: " << _dllPath
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

    // Layout-ABI guard (track B2, docs/resolver_params_abi_plan.md): the DLL
    // shares SurfaceData / MaterialData layouts with the exe by header
    // inclusion, but the reload trigger watches only the DLL SOURCE
    // timestamp — a header-only layout change would leave a stale-layout DLL
    // silently loaded (UB). Every DLL built against the current headers
    // exports ntCallableAbiVersion(); a mismatch (or a pre-guard DLL with no
    // export) forces one rebuild before use.
    if (auto abiFn = reinterpret_cast<std::uint32_t(*)()>(
            GetProcAddress(_module, "ntCallableAbiVersion")); abiFn) {
        std::uint32_t dllAbi = abiFn();
        if (dllAbi != render::kCallableAbiVersion) {
            CI_LOG_W("CallableDLLLoader: DLL ABI version " << dllAbi
                << " != engine " << render::kCallableAbiVersion
                << " (SurfaceData/MaterialData layout drift) - force-rebuilding");
            FreeLibrary(_module);
            _module = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            if (!buildDLL()) {
                logError("ABI-version rebuild failed");
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            _module = LoadLibraryA(_dllPath.c_str());
            if (!_module) {
                logError("Failed to reload DLL after ABI-version rebuild");
                return false;
            }
            abiFn = reinterpret_cast<std::uint32_t(*)()>(
                GetProcAddress(_module, "ntCallableAbiVersion"));
            if (!abiFn || abiFn() != render::kCallableAbiVersion) {
                logError("DLL ABI version still mismatched after rebuild - the DLL "
                         "source is built against different engine headers than "
                         "this exe; refusing to load (stale-layout callables "
                         "would be silent UB)");
                FreeLibrary(_module);
                _module = nullptr;
                return false;
            }
        }
    } else {
        CI_LOG_W("CallableDLLLoader: DLL predates the ABI-version export "
                 "(ntCallableAbiVersion) - force-rebuilding");
        FreeLibrary(_module);
        _module = nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (!buildDLL()) {
            logError("ABI-version rebuild failed (pre-guard DLL)");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        _module = LoadLibraryA(_dllPath.c_str());
        if (!_module) {
            logError("Failed to reload DLL after ABI-version rebuild (pre-guard)");
            return false;
        }
    }

    // ABI v2 preferred: same callable registration plus runtime tuning
    // params (docs/resolver_params_abi_plan.md). Falls back to v1 when the
    // export is absent (older DLLs) or when the host provided no paramFn.
    _dllRegisterFn2 = reinterpret_cast<DLLRegisterFn2>(
        GetProcAddress(_module, "registerMaterialCallables2"));
    _dllRegisterFn = reinterpret_cast<DLLRegisterFn>(
        GetProcAddress(_module, "registerMaterialCallables"));
    _dllUnregisterFn = reinterpret_cast<DLLUnregisterFn>(
        GetProcAddress(_module, "unregisterMaterialCallables"));

    if (_dllRegisterFn2 && _hostParamFn) {
        _dllRegisterFn2(_hostRegisterFn, _hostParamFn, _hostClearFn);
        CI_LOG_I("CallableDLLLoader: Registered custom material callables (ABI v2 - params)");
        return true;
    }
    if (_dllRegisterFn2 && !_hostParamFn) {
        // Host has no params support: register callables only (DLL sees a
        // null paramFn and must skip param registration).
        _dllRegisterFn2(_hostRegisterFn, nullptr, _hostClearFn);
        CI_LOG_I("CallableDLLLoader: Registered custom material callables (ABI v2, no host params)");
        return true;
    }

    if (!_dllRegisterFn) {
        logError("registerMaterialCallables not found in DLL");
        FreeLibrary(_module);
        _module = nullptr;
        return false;
    }

    // v1 fallback
    _dllRegisterFn(_hostRegisterFn, _hostClearFn);
    CI_LOG_I("CallableDLLLoader: Registered custom material callables (v1 ABI - no params)");
    return true;
}

void CallableDLLLoader::unloadDLL() {
    if (!_module) return;

    // Unregister before unloading
    if (_dllUnregisterFn && _hostClearFn) {
        _dllUnregisterFn(_hostClearFn);
        CI_LOG_D("CallableDLLLoader: Unregistered custom material callables");
    }

    _dllRegisterFn   = nullptr;
    _dllRegisterFn2  = nullptr;
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
                CI_LOG_D("CallableDLLLoader: DLL not found, building...");
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
    CI_LOG_D("CallableDLLLoader: Detected source change, rebuilding...");

    // Unload old DLL (releases file lock)
    unloadDLL();

    // Give Windows time to release the file lock
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Backup old DLL before build so we can restore on failure. COPY, not
    // rename: the DLL is already unloaded (no file lock), and keeping the
    // original in place preserves the linker's incremental state — renaming
    // it away forced a FULL link (full PDB rewrite) on every reload, which
    // is exactly what collides with a locked/corrupt PDB (LNK1201).
    auto backupPath = _dllPath + ".bak";
    bool hadBackup = false;
    if (std::filesystem::exists(_dllPath)) {
        std::error_code ec;
        std::filesystem::copy_file(_dllPath, backupPath,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) {
            hadBackup = true;
            CI_LOG_D("CallableDLLLoader: Backed up DLL to " << backupPath);
        } else {
            CI_LOG_W("CallableDLLLoader: Failed to backup old DLL: " << ec.message());
        }
    }

    // Rebuild
    if (!buildDLL()) {
        logError("Failed to rebuild callable DLL — restoring previous version");
        if (hadBackup && std::filesystem::exists(backupPath)) {
            std::error_code ec;
            std::filesystem::copy_file(backupPath, _dllPath,
                                       std::filesystem::copy_options::overwrite_existing, ec);
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

    // INCREMENTAL build (no :Rebuild): the reload flow has already unloaded
    // the DLL by this point, so MSBuild can overwrite it; the changed source
    // recompiles the single TU (PCH warm) and relinks — ~1s vs ~10s for a
    // forced full rebuild of a one-file project.
    std::ostringstream cmd;
    cmd << "\"" << msbuild << "\""
        << " \"" << slnPath.string() << "\""
        << " /t:" << targetName
        << " /p:Configuration=Debug_Runtime"
        << " /p:Platform=x64"
        << " /v:minimal"
        << " /nologo";
    const std::string cmdStr = cmd.str();

    // One MSBuild invocation: returns (exitCode, output, elapsedMs).
    auto runBuild = [&]() {
        struct Result { DWORD exitCode; std::string output; long long elapsedMs; };
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
            return Result{ 1u, "CreatePipe failed", 0 };
        }
        SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.hStdOutput = hWritePipe;
        si.hStdError  = hWritePipe;
        si.wShowWindow = SW_HIDE;

        std::vector<char> cmdBuf(cmdStr.begin(), cmdStr.end());
        cmdBuf.push_back('\0');

        BOOL ok = CreateProcessA(NULL, cmdBuf.data(), NULL, NULL, TRUE,
                                  CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
        CloseHandle(hWritePipe);

        if (!ok) {
            logError("Failed to execute MSBuild (error: " + std::to_string(GetLastError()) + ")");
            CloseHandle(hReadPipe);
            return Result{ 1u, "CreateProcessA failed", 0 };
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
        return Result{ exitCode, std::move(output), elapsed };
    };

    auto result = runBuild();

    // LNK1201 recovery: "error writing to program database" — the PDB is
    // locked (e.g. a debugger holds it after symbol load) or corrupt from an
    // interrupted link. Deleting it forces the linker to recreate a fresh PDB;
    // retry once before giving up. (A debugger-held PDB cannot be deleted —
    // the retry then fails with the same error and the output explains it.)
    if (result.exitCode != 0u &&
        result.output.find("LNK1201") != std::string::npos) {
        CI_LOG_W("CallableDLLLoader: LNK1201 writing the PDB - deleting stale "
                 "PDB/ILK and retrying once");
        auto dllDir = std::filesystem::path(_dllPath).parent_path();
        auto stem   = std::filesystem::path(_dllPath).stem().string();
        for (const char* ext : { ".pdb", ".ilk" }) {
            std::error_code ec;
            auto p = dllDir / (stem + ext);
            std::filesystem::remove(p, ec);
            if (ec) {
                CI_LOG_W("CallableDLLLoader: Could not delete " << p.string()
                         << " (" << ec.message() << ") - likely held open by a "
                         << "debugger; hot-reload needs a session without the "
                         << "debugger attached (Ctrl+F5)");
            }
        }
        result = runBuild();
    }

    auto exitCode = result.exitCode;
    auto output   = std::move(result.output);
    auto elapsed  = result.elapsedMs;

    if (exitCode != 0) {
        CI_LOG_E("CallableDLLLoader: Build failed (" << elapsed << "ms, exit=" << exitCode << ")");
        // Surface the compiler errors — without this a mid-edit save (transient
        // syntax error) looks like an unexplained silent no-op after the
        // backup restore.
        if (!output.empty()) {
            CI_LOG_E("CallableDLLLoader: MSBuild output:\n" << output);
        } else {
            CI_LOG_E("CallableDLLLoader: No MSBuild output captured");
        }
        _lastError = "Build failed (exit " + std::to_string(exitCode) + ")";
        return false;
    }

    CI_LOG_D("CallableDLLLoader: Build succeeded (" << elapsed << "ms)");
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
