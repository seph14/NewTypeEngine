# Prebuilt Engine Library (win-x64 distro)

How NewTypeEngine ships to users as a prebuilt static library, and the rules
that keep the prebuilt/consumer ABI stable. This document is the user/maintainer guide
for what was built.

## Overview

A consumer project compiles **only its own `App.cpp`** and links
`NewTypeEngine.lib` — no engine sources, no FidelityFX copy, no cinder build
knowledge. Everything resolves inside one self-contained distro folder:

```
NewTypeEngine-<ver>-win-x64/
    include/newtype/      engine headers (EngineVersion.h stamped with <ver>)
    lib/Debug/            NewTypeEngine.lib (+ .pdb) — serves Debug AND Debug_Runtime
    lib/Release/          NewTypeEngine.lib (+ .pdb)
    props/EngineCommon.props   central compile/link blob (roots default to ../deps)
    deps/Cinder/          include/ + lib/msw/x64/{Debug_MD,Release_MD}/v143/cinder.lib
    deps/LuisaCompute/    include/ + src/ext/ headers + build-dx[-debug] import libs
                          + runtime DLLs + luisa_embed_device_lib.exe
    deps/FidelityFX/      FSR 3.1 headers + runtime DLLs
    README.md
```

Scaffold a consumer with:

```bash
python tools/generate_project.py --path D:/Projects --name MyDemo            # prebuilt (default)
python tools/generate_project.py --path . --name Proto --engine source      # copy-tree mode (previous behaviour)
```

`--engine-root` selects the distro (default: newest `dist/NewTypeEngine-*-win-x64`
next to the engine checkout).

## Why this shape

- **Static lib for the engine, DLLs only where sharing is required.** The
  hot-reload boundary (runtime shader DLLs ↔ host) needs *shared* state for
  exactly two things: the LuisaCompute runtime/ast/dsl DLLs and the `/MD(d)`
  CRT. Both are shared automatically because consumers link the same
  LuisaCompute import libs and use the same CRT flavor. Cinder is statically
  linked everywhere — the runtime shader DLLs reference zero cinder symbols
  (verified: their link lines carry only the six luisa import libs), and
  shared cinder never exported the D3D12 renderer the DX present path
  instantiates (`RendererD3d12` has no `CI_API` in DLL builds; exporting it is
  an optional cinder-fork patch, see *Out of scope* below).
- **LuisaCompute stays DLL.** `SHARED` is hardcoded upstream, backends are
  plugin-scanned from the exe directory at context creation, and the shader
  hot-reload boundary needs one shared copy of `luisa-runtime/ast/dsl` state.
  Consumers' post-build events copy the DLLs from the distro.
- **One Debug lib serves Debug and Debug_Runtime.** The RT (DLL hot-reload)
  machinery is compiled into every `_DEBUG` binary and keyed on `_DEBUG`, not
  `RT_RUNTIME`. `RT_RUNTIME` remains a *consumer-side behavioral* define only:
  it flips the `ShaderManager::loadShader` hot-reload default and
  `isRuntimeMode()`, and the dev test scenes use it to opt into DLL-backed
  materials. It deliberately does **not** change engine class layout, so
  plain-Debug and Debug_Runtime TUs link the same library with no dllimport
  mixing. Consequently the machinery guards (`Pipeline::_callableDLL`,
  `ShaderManager::_dllLoaders`, `processPendingReloads`, the
  `CallableDLLLoader`/`DLLHotReload` sources) are `#ifdef _DEBUG`; the
  shader-DLL boundary macros (`CUSTOM_MATERIAL_API`, `rt_virtual`, the
  `*_EXPORTS` defines in `runtime_shaders/*`) stay on `RT_RUNTIME`.

### Watch-thread note

`ShaderManager::launch()` (the file-watch thread) used to have no caller. It
now starts lazily with the first successful shader-DLL load, guarded by
`_dllMutex` (main-thread map mutation vs. watch-thread iteration). Plain
Debug apps that never load a shader DLL pay nothing — no thread, no polling.

## Frozen-macro policy (ABI fingerprint)

`include/newtype/core/EngineVersion.h` enforces the frozen-macro set at
**link time** using MSVC's `detect_mismatch`: every object file (library
and consumer) carries one record per frozen input (`NT_DEBUG_VIZ`,
`NT_ALLOW_RASTER_FEATURES`, `NT_ENABLE_PROCEDURAL`, `NT_ENABLE_SHARC`,
`NT_SHARC_COMPACT`, `NT_SHARC_64_BIT_ATOMICS`, `NT_ENABLE_VIDEO_RECORDER`,
`NT_ENABLE_EDITOR`, plus `EA_DLL`, `FMT_HEADER_ONLY`, `MARL_DLL` and the
`_DEBUG`/CRT flavor), and the linker fails with
LNK2038/LNK1319 when objects with different values are linked — a drifted
consumer `/D` breaks the **build**, not the run, and cannot be dead-stripped
by `/OPT:REF`. (The version string is deliberately not a link-time key: the
lib is built against the repo header while distro headers carry the
stamped version.) A runtime FNV-1a fingerprint compare
(`newtype::detail::engineAbiCheckRaw()`, wired as an eager per-TU static
guard plus ShaderManager dispatch backstops) remains as defense-in-depth.
Most `NT_*` macros in `Config.h` are additionally unconditionally `#define`d,
so `/D`-ing them is structurally ignored. The runtime shader DLLs neither
link EngineVersion.cpp nor include this header and are unaffected.

Rules:

- Prebuilt consumers must not `/D` any macro in that set. They all ship set
  in the distro's `EngineCommon.props`.
- `RT_RUNTIME` is intentionally **not** fingerprinted (Debug and
  Debug_Runtime consumers share one Debug lib).
- The `#ifndef` guards in `Config.h` (e.g. `NT_ENABLE_VALIDATION`) stay:
  harnesses need them. Anything not in the fingerprint list but layout
  suspect should be added to `NT_ABI_STRING` (bump the `ntabi` tag).

## EngineCommon.props — the central blob

`vc2022/EngineCommon.props` is generated (do not hand-edit):

```bash
python tools/generate_project.py --emit-engine-common
```

`tools/generate_project.py:engine_common_props_text()` is the single source;
`package_dist.py` writes the same text into the distro, and source-mode
generated projects get their own copy. It provides, per configuration:

- the full preprocessor define blob (Debug and Debug_Runtime share one set),
- include directories (engine include via `$(NewTypeEngineInclude)` +
  cinder + luisa ext + FidelityFX),
- compiler options (C++20, `/utf-8`, `/Zc:__cplusplus /Zc:preprocessor`,
  warnings, `/MP`),
- `$(EngineSystemLibs)`, `$(EngineLuisaLibsDebug/Release)`,
  `$(EngineLinkDepsDebug/Release)` link macros.

Importers append project-specific entries with `%(PreprocessorDefinitions)`
etc. inheritance — the dev vcxproj appends `RT_RUNTIME` (Debug_Runtime) and
its LCS extras; shader DLL projects append `*_EXPORTS` +
`_CRT_SECURE_NO_WARNINGS` and link only `$(EngineLuisaLibsDebug)`; generated
projects append `RT_RUNTIME` for Debug_Runtime. All dependency roots are
`Condition="'' == ''"` overridable, defaulting to distro-relative `deps\`.

## The library project

`vc2022/NewTypeEngineLib/NewTypeEngineLib.vcxproj` — `StaticLibrary`,
Debug|x64 + Release|x64, v143/C++20, `/MDd`//`/MD`, no `/GL`/LTCG, built
against the static `Debug_MD`/`Release_MD` cinder. It compiles the shared
`ENGINE_SOURCES` manifest (74 files — same list `generate_project.py` uses
for source-mode projects, keeping the two consumption modes content-identical)
+ `glad.c` + the two RT machinery sources in Debug. Output name is
`NewTypeEngine.lib` (`TargetName`).

In `NewTypeEngine.sln` the lib is mapped **ActiveCfg-only**: normal solution
builds skip it (the dev exe stays source-compiled; the owner inner loop is
unchanged). It builds on demand or via `package_dist.py`. Debug_Runtime maps
to the lib's Debug config.

Static-lib linker discarding is not a concern: engine core is explicitly
constructed by the app (no self-registering statics). If that ever changes,
`/WHOLEARCHIVE:NewTypeEngine.lib` on the consumer link line is the documented
fallback.

## Packaging

```bash
python tools/package_dist.py                    # build lib + assemble dist/
python tools/package_dist.py --zip              # + versioned zip
python tools/package_dist.py --verify           # + smoke test (see below)
python tools/package_dist.py --skip-build       # reassemble from built libs
```

Dependency checkouts are resolved via `--cinder` / `--luisa` (defaults are
the dev machine paths). The distro (and `dist/` generally) is git-ignored
and never committed.

## Validation

- **Dev configs**: Debug_Runtime builds and links again after the de-share
  (it previously could not link the DX present path against shared cinder);
  hot reload of the runtime shader DLLs and the custom-material callable DLL
  works as before (machinery now simply keyed on `_DEBUG`).
- **Lib**: builds clean in Debug + Release.
- **`package_dist.py --verify`**: scaffolds a scratch `--engine prebuilt`
  project against the distro, msbuilds Debug + Release + Debug_Runtime, and
  launches the Debug exe briefly (clean 20 s run = init succeeded).
- **Source-mode generator**: output diffed against the pre-refactor baseline;
  only intentional changes (EngineCommon import, blob moved to the sheet,
  EngineVersion/MeshVertex manifest additions).

## Addon interaction

`tools/cinder_blocks.py` injects into the generated vcxproj's per-config
elements (defines, include paths, link inputs) — those elements still exist
(they carry the appended extras), so `add sim` / `add video` / `add directml`
keep working on source-mode projects. The prebuilt lib already contains the
video/media sources; the sim (LuisaComputeSimulator) sources are dev-only and
not in the lib — sim stays a source-mode-only addon for now.

## Out of scope (Phase 2 candidates)

Cinder-fork DLL-mode DX export (`CI_API` on `RendererD3d12`, un-stub
`createFactory`/`createDevice`, add `d3d12.lib;dxgi.lib` to the two Shared
link lines) — only needed if `--engine source --link dll` + DX is wanted;
cinder `.lib` slimming (size-only win); dogfooding the lib in the dev exe;
LTCG variant; shader-DLL project templates in the generator; hosted
releases/CI; CUDA variant.
