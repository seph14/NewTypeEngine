#!/usr/bin/env python3
"""Assemble a prebuilt NewTypeEngine win-x64 distro.

Builds the NewTypeEngineLib static library (Debug + Release) with MSBuild and
packages it with the engine headers, static cinder, the LuisaCompute import
libs + runtime DLLs, and the vendored FidelityFX + NVIDIA DLSS + Valve Steam
Audio SDKs:

    dist/NewTypeEngine-<ver>-win-x64/
        include/newtype/**           (EngineVersion.h stamped with <ver>)
        lib/{Debug,Release}/NewTypeEngine.lib (+ .pdb)
        props/EngineCommon.props     (roots default to ../deps)
        deps/Cinder/                 (include + Debug_MD/Release_MD cinder.lib)
        deps/LuisaCompute/           (include + src/ext + build-dx[-debug])
        deps/FidelityFX/
        deps/NVIDIA/DLSS/
        deps/Valve/SteamAudio/
        README.md

Consumers scaffold against it with:
    python tools/generate_project.py --path ... --name ... [--engine-root <distro>]

Dependency sources default to the dev machine's local checkouts and can be
overridden with --cinder / --luisa. The distro is never committed to git
(dist/ is ignored); --zip additionally produces a versioned zip.

    python tools/package_dist.py                     # build + assemble
    python tools/package_dist.py --skip-build        # assemble from existing libs
    python tools/package_dist.py --verify            # + smoke: build & run a
                                                     # scratch prebuilt project
"""

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
VSWHERE = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
)

LUISA_LINK_LIBS = [
    "luisa-ast.lib", "luisa-core.lib", "luisa-dsl.lib",
    "luisa-osl.lib", "luisa-runtime.lib", "luisa-xir.lib",
]


def find_msbuild() -> str:
    if VSWHERE.exists():
        out = subprocess.run(
            [
                str(VSWHERE), "-latest", "-products", "*",
                "-requires", "Microsoft.Component.MSBuild",
                "-find", r"MSBuild\**\Bin\MSBuild.exe",
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        if out:
            return out[0]
    raise RuntimeError(
        "MSBuild.exe not found via vswhere; build from a VS2022 developer "
        "prompt or pass a path via PATH."
    )


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def build_libs(msbuild: str) -> None:
    proj = ENGINE_ROOT / "vc2022" / "NewTypeEngineLib" / "NewTypeEngineLib.vcxproj"
    for cfg in ("Debug", "Release"):
        r = run([
            msbuild, str(proj),
            f"/p:Configuration={cfg}", "/p:Platform=x64",
            "/m", "/v:m", "/nologo",
        ])
        if r.returncode != 0:
            raise SystemExit(f"error: NewTypeEngineLib {cfg} build failed")


def copy_tree(src: Path, dst: Path, skip=None) -> int:
    """Copy a directory tree; returns file count. skip: predicate on Path."""
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if skip and skip(p):
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def assemble(distro: Path, cinder: Path, luisa: Path, version: str) -> None:
    sys.path.insert(0, str(ENGINE_ROOT / "tools"))
    from generate_project import engine_common_props_text

    if distro.exists():
        print(f"  removing existing {distro}")
        shutil.rmtree(distro)
    distro.mkdir(parents=True)

    # --- engine headers (full tree; EngineVersion.h stamped) -------------
    n = copy_tree(ENGINE_ROOT / "include" / "newtype", distro / "include" / "newtype")
    ver_header = distro / "include" / "newtype" / "core" / "EngineVersion.h"
    text = ver_header.read_text(encoding="utf-8")
    stamped = text.replace(
        '#define NT_ENGINE_VERSION "source"',
        f'#define NT_ENGINE_VERSION "{version}"',
    )
    if stamped == text:
        raise SystemExit("error: could not stamp EngineVersion.h "
                         "(NT_ENGINE_VERSION placeholder not found)")
    ver_header.write_text(stamped, encoding="utf-8", newline="\r\n")
    print(f"  include/newtype: {n} files (EngineVersion.h -> {version})")

    # --- static libs ------------------------------------------------------
    lib_proj_out = ENGINE_ROOT / "vc2022" / "NewTypeEngineLib" / "x64"
    for cfg in ("Debug", "Release"):
        src_lib = lib_proj_out / cfg / "NewTypeEngine.lib"
        if not src_lib.exists():
            raise SystemExit(f"error: {src_lib} missing (build first)")
        dst_dir = distro / "lib" / cfg
        dst_dir.mkdir(parents=True)
        shutil.copy2(src_lib, dst_dir / "NewTypeEngine.lib")
        pdb = lib_proj_out / cfg / "NewTypeEngine.pdb"
        if pdb.exists():
            shutil.copy2(pdb, dst_dir / "NewTypeEngine.pdb")
        print(f"  lib/{cfg}: NewTypeEngine.lib"
              + (" + pdb" if pdb.exists() else ""))

    # --- props --------------------------------------------------------------
    props_dir = distro / "props"
    props_dir.mkdir()
    (props_dir / "EngineCommon.props").write_text(
        engine_common_props_text(), encoding="utf-8", newline="\r\n"
    )
    print("  props/EngineCommon.props")

    # --- cinder (static MD only; no shared cinder, no ANGLE DLLs) ----------
    n = copy_tree(cinder / "include", distro / "deps" / "Cinder" / "include")
    for cfg in ("Debug_MD", "Release_MD"):
        src = cinder / "lib" / "msw" / "x64" / cfg / "v143" / "cinder.lib"
        if not src.exists():
            raise SystemExit(f"error: {src} missing (build cinder {cfg} first)")
        dst = distro / "deps" / "Cinder" / "lib" / "msw" / "x64" / cfg / "v143"
        dst.mkdir(parents=True)
        shutil.copy2(src, dst / "cinder.lib")
        pdb = src.with_suffix(".pdb")
        if pdb.exists():
            shutil.copy2(pdb, dst / "cinder.pdb")
    print(f"  deps/Cinder: include ({n} files) + Debug_MD/Release_MD cinder.lib")

    # --- LuisaCompute (include + ext headers + import libs + DLLs) ---------
    lc = distro / "deps" / "LuisaCompute"
    n_inc = copy_tree(luisa / "include", lc / "include")
    n_ext = copy_tree(luisa / "src" / "ext", lc / "src" / "ext",
                      skip=lambda p: p.suffix.lower() not in {
                          ".h", ".hh", ".hpp", ".hxx", ".inl", ".inc",
                          ".natvis", ".txt", ".md", ".cmake", ".py",
                      } and p.name.lower() not in {"license", "licence"})
    for build, sub in (("build-dx", "Release"), ("build-dx-debug", "Debug")):
        # import libs actually linked by consumers
        lib_dst = lc / build / "lib"
        lib_dst.mkdir(parents=True)
        for name in LUISA_LINK_LIBS:
            src = luisa / build / "lib" / name
            if not src.exists():
                raise SystemExit(f"error: {src} missing")
            shutil.copy2(src, lib_dst / name)
        # runtime DLLs + the embed tool (backends are plugin-scanned from
        # the exe dir; dxcompiler/dxil are loaded by the DX backend)
        bin_dst = lc / build / "bin"
        bin_dst.mkdir(parents=True)
        n_dll = 0
        for p in (luisa / build / "bin").iterdir():
            if p.suffix.lower() == ".dll":
                shutil.copy2(p, bin_dst / p.name)
                n_dll += 1
            elif p.name == "luisa_embed_device_lib.exe":
                shutil.copy2(p, bin_dst / p.name)
    print(f"  deps/LuisaCompute: include ({n_inc}) + src/ext ({n_ext}) + "
          f"6 import libs + DLLs per config")

    # --- FidelityFX ---------------------------------------------------------
    n = copy_tree(ENGINE_ROOT / "external" / "FidelityFX", distro / "deps" / "FidelityFX")
    print(f"  deps/FidelityFX: {n} files")

    # --- NVIDIA DLSS ----------------------------------------------------------
    # Headers (NgxContext/DlssSrBackend/DlssRrDenoiser compile against them),
    # the nvsdk_ngx_d[_dbg].lib stubs consumers link via $(DlssLib*), and the
    # nvngx_dlss[d].dll feature DLLs the post-build deploys next to the exe.
    n = copy_tree(ENGINE_ROOT / "external" / "NVIDIA" / "DLSS",
                  distro / "deps" / "NVIDIA" / "DLSS")
    print(f"  deps/NVIDIA/DLSS: {n} files")

    # --- Valve Steam Audio ----------------------------------------------------
    # phonon.h (SpatialAudioSystem compiles against it), the phonon.lib import
    # lib consumers link via $(SteamAudioLib), and phonon.dll (53 MB) the
    # post-build deploys next to the exe for the HOA spatial audio system.
    n = copy_tree(ENGINE_ROOT / "external" / "Valve" / "SteamAudio",
                  distro / "deps" / "Valve" / "SteamAudio")
    print(f"  deps/Valve/SteamAudio: {n} files")

    # --- README -------------------------------------------------------------
    (distro / "README.md").write_text(DISTRO_README.format(version=version),
                                      encoding="utf-8", newline="\r\n")
    print("  README.md")


def verify(distro: Path, msbuild: str) -> bool:
    """Smoke test: scaffold a scratch prebuilt project against the distro,
    build Debug/Release/Debug_Runtime, and launch the Debug exe briefly."""
    sys.path.insert(0, str(ENGINE_ROOT / "tools"))
    scratch = distro.parent / "verify_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    r = run([
        sys.executable, str(ENGINE_ROOT / "tools" / "generate_project.py"),
        "--path", str(scratch.parent), "--name", scratch.name,
        "--engine", "prebuilt", "--engine-root", str(distro),
    ])
    if r.returncode != 0:
        return False
    sln = scratch / "vc2022" / f"{scratch.name}.sln"
    for cfg in ("Debug", "Release", "Debug_Runtime"):
        r = run([
            msbuild, str(sln),
            f"/p:Configuration={cfg}", "/p:Platform=x64",
            "/m", "/v:m", "/nologo",
        ])
        if r.returncode != 0:
            print(f"verify: {cfg} build FAILED")
            return False
        print(f"verify: {cfg} build OK")
    # Launch the Debug exe briefly (engine startup compiles/loads a large
    # shader set; surviving >20s without exiting means init succeeded).
    exe = scratch / "vc2022" / "x64" / "Debug" / f"{scratch.name}.exe"
    print(f"verify: launching {exe} for 20s ...")
    proc = subprocess.Popen([str(exe)], cwd=str(exe.parent))
    deadline = time.time() + 20.0
    exited_early = None
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            exited_early = rc
            break
        time.sleep(0.5)
    if exited_early is not None:
        print(f"verify: exe exited early with code {exited_early} — FAIL")
        return False
    proc.kill()
    proc.wait(timeout=10)
    print("verify: exe ran 20s cleanly — OK")
    # The just-killed exe can hold locks for a moment; retry the removal.
    for _ in range(3):
        try:
            shutil.rmtree(scratch)
            break
        except OSError:
            time.sleep(1.0)
    else:
        shutil.rmtree(scratch, ignore_errors=True)
    return True


DISTRO_README = """\
# NewTypeEngine {version} (win-x64 prebuilt)

Prebuilt static library + headers for NewTypeEngine. Requirements:
Visual Studio 2022 (toolset v143) and the Windows 10 SDK.

## Quick start

From an NewTypeEngine source checkout (or just its tools/ folder):

    python tools/generate_project.py --path D:/Projects --name MyDemo \\
        --engine-root <this folder>

The generated project compiles only its own `src/MyDemoApp.cpp` and links
`lib/Debug/NewTypeEngine.lib` (Debug and Debug_Runtime) or
`lib/Release/NewTypeEngine.lib` (Release) plus static cinder and the
LuisaCompute import libs from `deps/`. Post-build copies the LuisaCompute,
FidelityFX, DLSS and Steam Audio runtime DLLs next to the exe.

## Frozen-macro policy

The engine ABI depends on a frozen set of macros (all `NT_*` in
`newtype/core/Config.h`, plus `EA_DLL`, `FMT_HEADER_ONLY`, `MARL_DLL` and
the CRT flavor). Do **not** define any of them in your project — they ship
set in `props/EngineCommon.props`, and a mismatch fails the LINK with
LNK2038/LNK1319 via `detect_mismatch` (see
`include/newtype/core/EngineVersion.h`); a runtime fingerprint check is a
further backstop.
`NT_ENABLE_VALIDATION` and the other `#ifndef`-guarded entries in Config.h
remain overridable where they do not affect layout — when in doubt, don't.

Configurations: Debug and Debug_Runtime both link the Debug library and
share one class layout; Debug_Runtime additionally defines RT_RUNTIME
(shader-DLL hot-reload defaults). Release links the Release library.

## Layout

    include/newtype/   engine headers
    lib/Debug|Release/ NewTypeEngine.lib (+ .pdb)
    props/             EngineCommon.props (imported by generated projects)
    deps/Cinder/       static /MD(d) cinder (Debug_MD / Release_MD)
    deps/LuisaCompute/ headers + import libs + runtime DLLs (stays DLL:
                       plugin-scanned backends + shader hot-reload shared
                       state)
    deps/FidelityFX/   FSR 3.1 headers + runtime DLLs
    deps/NVIDIA/DLSS/  DLSS 4.5 headers + nvsdk_ngx stub libs + feature DLLs
    deps/Valve/SteamAudio/  phonon.h + phonon.lib + phonon.dll (HOA audio)

See docs/prebuilt_dist.md in the engine repository for the full story.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="0.1.0",
                    help="Distro/version string stamped into EngineVersion.h "
                         "and the folder/zip name")
    ap.add_argument("--out", type=Path, default=ENGINE_ROOT / "dist",
                    help="Output directory (default: <engine>/dist)")
    ap.add_argument("--cinder", type=Path,
                    default=Path(r"../../Cinder"),
                    help="Cinder checkout with lib/msw/x64/{{Debug_MD,Release_MD}}")
    ap.add_argument("--luisa", type=Path,
                    default=Path(r"../../LuisaCompute"),
                    help="LuisaCompute checkout with build-dx[-debug]")
    ap.add_argument("--skip-build", action="store_true",
                    help="Do not msbuild the lib; package existing outputs")
    ap.add_argument("--zip", action="store_true",
                    help="Additionally write a versioned zip next to the distro")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke test: scratch prebuilt project + build all "
                         "three configs + brief launch")
    args = ap.parse_args()

    distro = args.out / f"NewTypeEngine-{args.version}-win-x64"

    msbuild = None
    if not args.skip_build or args.verify:
        msbuild = find_msbuild()

    if not args.skip_build:
        print("== building NewTypeEngineLib (Debug + Release)")
        build_libs(msbuild)

    print(f"== assembling {distro}")
    assemble(distro, args.cinder, args.luisa, args.version)

    if args.zip:
        zip_path = distro.with_suffix(".zip")
        print(f"== zipping -> {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in distro.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(distro.parent))
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"   {zip_path.name}: {size_mb:.1f} MiB")

    if args.verify:
        print("== verify")
        if not verify(distro, msbuild):
            return 1
        print("verify: PASS")

    print(f"\nDistro ready: {distro}")
    print("Scaffold consumers with:")
    print(f"  python tools/generate_project.py --path <dir> --name <Name> "
          f"--engine-root {distro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
