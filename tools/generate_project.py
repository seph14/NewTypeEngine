#!/usr/bin/env python3
"""NewTypeEngine project generator.

Creates a new VS2022 project from an app template with a renamed app class.

Engine consumption modes (--engine):
    prebuilt (default) — link the prebuilt NewTypeEngine.lib from a packaged
        distro (tools/package_dist.py). The project compiles only its own
        App.cpp; engine headers, the static lib, cinder/luisa deps and the
        FidelityFX SDK all resolve inside the distro via --engine-root.
    source — copy the engine source tree into <path>/<name>/engine/ so the
        new project owns a private, freely editable copy and compiles it
        (today's behaviour; --link dll|static applies).

App templates (--mode):
    dx (default) — src/TemplateDX.cpp. Presents through Cinder's D3D12
        renderer: the Luisa device adopts its ID3D12Device and the frame
        handoff is an on-device copy into the back buffer (core::DxPresent,
        no GL context). UIFrame handling (beginUiFrame in update) and
        recorder updates (updateRecorder in draw) are wired for that path.
    gl — src/Template.cpp. Classic RendererGl present path: the engine
        renders through the DX12->GL interop and the app blits the texture
        with gl::draw.

Usage:
    python tools/generate_project.py --path D:/Projects --name MyDemo
    python tools/generate_project.py --path . --name TestProject --engine source --link static
    python tools/generate_project.py --path . --name GlProto --engine source --mode gl
    python tools/generate_project.py --emit-engine-common   # refresh vc2022/EngineCommon.props
"""

import argparse
import shutil
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Engine source/header manifest (paths relative to engine src/ or include/)
# ---------------------------------------------------------------------------
# Mirrors the curated compile set of the dev vcxproj and the prebuilt
# NewTypeEngineLib. Dev-only experimental modules (newtype/exp, newtype/physics,
# newtype/sim) stay out; everything the curated sources reference at link time
# with default Config.h flags must stay in (DxPresent, PassSharc, Gizmo,
# Fsr31Backend, ...). EngineVersion.cpp must be in every binary that links
# against the engine headers (it pins the ABI-fingerprint guard).

ENGINE_SOURCES = [
    "newtype/core/DxGLInterop.cpp",
    "newtype/core/DxPresent.cpp",
    "newtype/core/EngineVersion.cpp",
    "newtype/core/LuisaGLInterop.cpp",
    "newtype/core/PassDenoiser.cpp",
    "newtype/core/PassDenoiserAtrous.cpp",
    "newtype/core/PassDenoiserHistory.cpp",
    "newtype/core/PassDenoiserHitAndTemporal.cpp",
    "newtype/core/PassDenoiserPrefilter.cpp",
    "newtype/core/PassDenoiserPrepass.cpp",
    "newtype/core/PassDenoiserUtility.cpp",
    "newtype/core/PassDI.cpp",
    "newtype/core/PassGI.cpp",
    "newtype/core/PipelineInit.cpp",
    "newtype/core/PipelineRender.cpp",
    "newtype/core/PipelineUpdate.cpp",
    "newtype/core/Renderer.cpp",
    "newtype/core/ShaderManager.cpp",
    "newtype/feature/BloomFeature.cpp",
    "newtype/feature/ChromaticAberrationFeature.cpp",
    "newtype/feature/DoFFeature.cpp",
    "newtype/feature/FxaaFeature.cpp",
    "newtype/feature/Gizmo.cpp",
    "newtype/feature/MotionBlurFeature.cpp",
    "newtype/feature/PointCloud.cpp",
    "newtype/feature/RasterContext.cpp",
    "newtype/feature/Trail.cpp",
    "newtype/media/VideoAudioNode.cpp",
    "newtype/audio/AmbisonicsNodes.cpp",
    "newtype/audio/SpatialAudio.cpp",
    "newtype/media/VideoAudioStream.cpp",
    "newtype/media/VideoDecoderD3D11.cpp",
    "newtype/media/VideoPlayer.cpp",
    "newtype/media/VideoTextureBridge.cpp",
    "newtype/plugin/MovieWriter.cpp",
    "newtype/render/BSDF.cpp",
    "newtype/render/BSSRDF.cpp",
    "newtype/render/LightSampler.cpp",
    "newtype/render/MaterialBuiltins.cpp",
    "newtype/render/MaterialPool.cpp",
    "newtype/render/PassSSS.cpp",
    "newtype/render/PassSharc.cpp",
    "newtype/render/Sampling.cpp",
    "newtype/render/TextureConverter.cpp",
    "newtype/scene/DeformableMesh.cpp",
    "newtype/scene/Geometry.cpp",
    "newtype/scene/InstancedMesh.cpp",
    "newtype/scene/Interaction.cpp",
    "newtype/scene/LightShape.cpp",
    "newtype/scene/MeshShape.cpp",
    "newtype/scene/ModelLoader.cpp",
    "newtype/scene/ProceduralGeometry.cpp",
    "newtype/scene/Shape.cpp",
    "newtype/scene/Transform.cpp",
    "newtype/scene/VATLoader.cpp",
    "newtype/scene/VATMesh.cpp",
    "newtype/scene/VoxelGrid.cpp",
    "newtype/timeline/CurveUI.cpp",
    "newtype/timeline/EasingCurve.cpp",
    "newtype/timeline/EventTrigger.cpp",
    "newtype/timeline/Gradient.cpp",
    "newtype/timeline/PathHelper.cpp",
    "newtype/timeline/Timeline.cpp",
    "newtype/ngx/NgxContext.cpp",
    "newtype/ngx/DlssRayReconstruction.cpp",
    "newtype/upscal/DlssSrBackend.cpp",
    "newtype/upscal/Fsr31Backend.cpp",
    "newtype/util/Camera.cpp",
    "newtype/util/CommandBuffer.cpp",
    "newtype/util/LutLoader.cpp",
    "newtype/util/Mesh.cpp",
    "newtype/util/Noise.cpp",
    "newtype/util/Profiler.cpp",
    "newtype/util/Rand.cpp",
    "newtype/util/Recorder.cpp",
    "newtype/util/Rng.cpp",
    "newtype/util/SoundController.cpp",
    "newtype/util/UiHelper.cpp",
    "newtype/util/Utilities.cpp",
]

RT_SOURCES = [
    "newtype/runtime/CallableDLLLoader.cpp",
    "newtype/runtime/DLLHotReload.cpp",
]

ENGINE_HEADERS = [
    "newtype/NewType.h",
    "newtype/core/BindingGroups.h",
    "newtype/core/Config.h",
    "newtype/core/DxGLInterop.h",
    "newtype/core/DxPresent.h",
    "newtype/core/EngineVersion.h",
    "newtype/core/FeatureContext.h",
    "newtype/core/FeaturePoint.h",
    "newtype/core/FrameContext.h",
    "newtype/core/IFeature.h",
    "newtype/core/IShaderGenerator.h",
    "newtype/core/IVoxelGridUser.h",
    "newtype/core/LuisaGLInterop.h",
    "newtype/core/Pipeline.h",
    "newtype/core/Renderer.h",
    "newtype/core/ShaderManager.h",
    "newtype/feature/BloomFeature.h",
    "newtype/feature/ChromaticAberrationFeature.h",
    "newtype/feature/DoFFeature.h",
    "newtype/feature/FxaaFeature.h",
    "newtype/feature/Gizmo.h",
    "newtype/feature/MotionBlurFeature.h",
    "newtype/feature/PointCloud.h",
    "newtype/feature/RasterBase.h",
    "newtype/feature/RasterContext.h",
    "newtype/feature/Trail.h",
    "newtype/media/AudioRingBuffer.h",
    "newtype/media/NativeTextureDesc.h",
    "newtype/media/VideoAudioNode.h",
    "newtype/audio/AmbisonicsNodes.h",
    "newtype/audio/SpatialAudio.h",
    "newtype/media/VideoAudioStream.h",
    "newtype/media/VideoDecoderD3D11.h",
    "newtype/media/VideoPlayer.h",
    "newtype/media/VideoTextureBridge.h",
    "newtype/plugin/MovieWriter.h",
    "newtype/render/BSDF.h",
    "newtype/render/BSSRDF.h",
    "newtype/render/DDAMarch.h",
    "newtype/render/GIBounce.h",
    "newtype/render/GIReservoir.h",
    "newtype/render/GIShading.h",
    "newtype/render/LightSampler.h",
    "newtype/render/Lobe.h",
    "newtype/render/Material.h",
    "newtype/render/MaterialPool.h",
    "newtype/render/MaterialSimilarity.h",
    "newtype/render/MetalData.h",
    "newtype/render/Packing.h",
    "newtype/render/PassDI.h",
    "newtype/render/PassDenoiser.h",
    "newtype/render/PassGI.h",
    "newtype/render/PassSSS.h",
    "newtype/render/PassSharc.h",
    "newtype/render/ProcBindlessSlots.h",
    "newtype/render/ProceduralTrace.h",
    "newtype/render/ReSTIR.h",
    "newtype/render/RelaxDenoiser.h",
    "newtype/render/Sampling.h",
    "newtype/render/Sharc.h",
    "newtype/render/Shading.h",
    "newtype/render/SurfaceData.h",
    "newtype/render/SurfaceResolver.h",
    "newtype/render/TextureConverter.h",
    "newtype/scene/DeformableMesh.h",
    "newtype/scene/Geometry.h",
    "newtype/scene/InstancedMesh.h",
    "newtype/scene/Interaction.h",
    "newtype/scene/LightShape.h",
    "newtype/scene/MeshShape.h",
    "newtype/scene/ModelLoader.h",
    "newtype/scene/ProceduralGeometry.h",
    "newtype/scene/Shape.h",
    "newtype/scene/Transform.h",
    "newtype/scene/VATLoader.h",
    "newtype/scene/VATMesh.h",
    "newtype/scene/VoxelGrid.h",
    "newtype/timeline/CurveUI.h",
    "newtype/timeline/EasingCurve.h",
    "newtype/timeline/EventTrigger.h",
    "newtype/timeline/Gradient.h",
    "newtype/timeline/PathHelper.h",
    "newtype/timeline/Timeline.h",
    "newtype/ngx/NgxContext.h",
    "newtype/ngx/DlssRayReconstruction.h",
    "newtype/upscal/DlssSrBackend.h",
    "newtype/upscal/Fsr31Backend.h",
    "newtype/upscal/UpscalerBackend.h",
    "newtype/util/AccumulationTime.h",
    "newtype/util/Camera.h",
    "newtype/util/CommandBuffer.h",
    "newtype/util/CompileProfiler.h",
    "newtype/util/LutLoader.h",
    "newtype/util/Mesh.h",
    "newtype/util/MeshVertex.h",
    "newtype/util/Noise.h",
    "newtype/util/ParallelFor.h",
    "newtype/util/PixelOps.h",
    "newtype/util/Profiler.h",
    "newtype/util/Rand.h",
    "newtype/util/Recorder.h",
    "newtype/util/Rng.h",
    "newtype/util/Simplex.h",
    "newtype/util/SoundController.h",
    "newtype/util/TypeConv.h",
    "newtype/util/UiHelper.h",
    "newtype/util/Utilities.h",
    "newtype/util/VATData.h",
    "newtype/util/Vertex.h",
]

RT_HEADERS = [
    "newtype/runtime/CallableDLLLoader.h",
    "newtype/runtime/DLLHotReload.h",
    "newtype/runtime/Virtual.h",
]

# Headers that get copied into the new project's own include/ directory
# (see the `copies` list in main). Referenced relative to vcx_dir, not engine_inc.
LOCAL_HEADERS = [
    "Resources.h",
]

# App entry-point template per --mode (file names under the engine's src/).
# Both carry the same class (NewTypeEngine) and window title, which
# generate_cpp() renames; they differ only in the presentation path.
APP_TEMPLATES = {
    "dx": "TemplateDX.cpp",  # D3D12 present (DxPresent, no GL context)
    "gl": "Template.cpp",    # RendererGl present (DX12->GL interop blit)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# VS well-known filter GUIDs
GUID_SRC = "{4FC737F1-C7A5-4376-A066-2A32D752A2FF}"
GUID_HDR = "{93995380-89BD-4b04-88EB-625FBE52EBFB}"
GUID_RES = "{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}"


def make_guid(name: str) -> str:
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_DNS, f"newtype.{name}")) + "}"


def file_to_filter(rel_path_str: str, kind: str) -> str:
    """Map a file path to a VS filter name.

    e.g. "newtype/core/Renderer.cpp" -> "Source Files\\newtype\\core"
         "Resources.h"               -> "Header Files"
    """
    parts = rel_path_str.replace("/", "\\").split("\\")
    # e.g. ["newtype", "core", "Renderer.cpp"]
    sub = "\\".join(parts[:-1])  # "newtype\\core"
    if sub:
        return f"{kind}\\{sub}"
    return kind


def collect_filter_names(file_list: list[str], kind: str) -> list[str]:
    """Return deduplicated ordered filter names for a set of files."""
    seen = set()
    result = []
    for f in file_list:
        name = file_to_filter(f, kind)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Preprocessor definitions
# ---------------------------------------------------------------------------

PREPROC_COMMON = (
    "WIN32;{MODE};_WINDOWS;NOMINMAX;_WIN32_WINNT=0x0601;"
    "LUISA_PLATFORM_WINDOWS=1;_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR;"
    "SPDLOG_FWRITE_UNLOCKED;SPDLOG_NO_EXCEPTIONS;SPDLOG_NO_THREAD_ID;"
    "SPDLOG_DISABLE_DEFAULT_LOGGER;FMT_USE_CONSTEVAL=0;FMT_EXCEPTIONS=0;"
    "FMT_HEADER_ONLY=1;FMT_USE_NOEXCEPT=1;XXH_INLINE_ALL;"
    "EA_PRAGMA_ONCE_SUPPORTED=1;EA_HAVE_CPP11_CONTAINERS=1;"
    "EA_HAVE_CPP11_ATOMIC=1;EA_HAVE_CPP11_CONDITION_VARIABLE=1;"
    "EA_HAVE_CPP11_MUTEX=1;EA_HAVE_CPP11_THREAD=1;EA_HAVE_CPP11_FUTURE=1;"
    "EA_HAVE_CPP11_TYPE_TRAITS=1;EA_HAVE_CPP11_TUPLES=1;"
    "EA_HAVE_CPP11_REGEX=1;EA_HAVE_CPP11_RANDOM=1;EA_HAVE_CPP11_CHRONO=1;"
    "EA_HAVE_CPP11_SCOPED_ALLOCATOR=1;EA_HAVE_CPP11_INITIALIZER_LIST=1;"
    "EA_HAVE_CPP11_SYSTEM_ERROR=1;EA_HAVE_CPP11_TYPEINDEX=1;"
    "EASTL_USER_LITERALS_ENABLED=0;EASTL_STD_ITERATOR_CATEGORY_ENABLED=1;"
    "EASTL_STD_TYPE_TRAITS_AVAILABLE=1;EASTL_MOVE_SEMANTICS_ENABLED=1;"
    "EASTL_VARIADIC_TEMPLATES_ENABLED=1;EASTL_VARIABLE_TEMPLATES_ENABLED=1;"
    "EASTL_INLINE_VARIABLE_ENABLED=1;EASTL_HAVE_CPP11_TYPE_TRAITS=1;"
    "EASTL_INLINE_NAMESPACES_ENABLED=1;EASTL_ALLOCATOR_EXPLICIT_ENABLED=1;"
    "EA_DLL=1;EASTL_USER_DEFINED_ALLOCATOR=1;"
    "EASTL_DEPRECATIONS_FOR_2024_APRIL=EA_DISABLED;"
    "REPROC_SHARED;REPROCXX_SHARED;MARL_DLL=1;"
    "LUISA_ENABLE_XIR=1;LUISA_ENABLE_DSL=1;_WITH_FMT=;{EXTRA}"
    "NTDDI_VERSION=0x06010000;%(PreprocessorDefinitions)"
)

PREPROC_DEBUG = PREPROC_COMMON.replace("{MODE}", "_DEBUG").replace("{EXTRA}", "")
PREPROC_RELEASE = PREPROC_COMMON.replace("{MODE}", "NDEBUG").replace("{EXTRA}", "")

# ---------------------------------------------------------------------------
# EngineCommon.props — the centralized compiler/link blob
# ---------------------------------------------------------------------------
# Single source of truth for the define/include/option duplication that used
# to be copy-pasted across the dev vcxproj, the four runtime-shader projects,
# the prebuilt library project and every generated project. This function's
# text is written to vc2022/EngineCommon.props (--emit-engine-common), into
# every source-mode generated project, and into <distro>/props/ by
# tools/package_dist.py. Roots are Condition='' overridable so an importing
# project (or an earlier props sheet) can repoint them at local checkouts.


def engine_common_props_text() -> str:
    include_dirs = (
        "$(NewTypeEngineInclude);"
        "$(CinderInclude);$(CinderIncludeAngle);"
        "$(LuisaComputeInclude);$(LuisaComputeSpdlog);$(LuisaComputeXxHash);"
        "$(LuisaComputeMagicEnum);$(LuisaComputeEASTL);$(LuisaComputeEABase);"
        "$(LuisaComputeReproc);$(LuisaComputeReprocXX);$(LuisaComputeMarl);"
        "$(LuisaComputeHalf);$(LuisaComputeStb);$(FidelityFXInclude);$(DlssInclude);$(SteamAudioInclude);"
        "%(AdditionalIncludeDirectories)"
    )
    def luisa_libs(cfg: str) -> str:
        m = f"$(LuisaComputeLib{cfg})"
        return (
            f"{m}\\luisa-dsl.lib;{m}\\luisa-xir.lib;{m}\\luisa-osl.lib;"
            f"{m}\\luisa-runtime.lib;{m}\\luisa-ast.lib;{m}\\luisa-core.lib"
        )

    luisa_libs_debug = luisa_libs("Debug")
    luisa_libs_release = luisa_libs("Release")

    def root(name: str, default: str) -> str:
        return (
            f"    <{name} Condition=\"'$({name})' == ''\">{default}</{name}>"
        )

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<!--
  EngineCommon.props — centralized NewTypeEngine compile/link settings.

  GENERATED FILE — do not hand-edit. Regenerate with:
      python tools/generate_project.py  (emit EngineCommon flag)
  (tools/generate_project.py:engine_common_props_text() is the source.)

  Provides, for the Debug / Debug_Runtime / Release configurations:
    - the full preprocessor define blob (Cinder/LuisaCompute/EASTL/fmt/...),
    - the engine + dependency include directories,
    - common compiler options (C++20, /utf-8, /Zc, warnings, /MP),
    - config-keyed link-dependency macros ($(EngineLinkDepsDebug/Release)).

  Import after Microsoft.Cpp.props (before the per-config user sheets) and
  append project-specific entries with %(PreprocessorDefinitions) /
  %(AdditionalIncludeDirectories) / %(AdditionalDependencies) inheritance.

  Dependency roots are Condition='' overridable: set CinderRoot /
  LuisaComputeRoot / FidelityFXRoot / NewTypeEngineRoot / NewTypeEngineInclude
  in an earlier props file to repoint them (e.g. at local checkouts). The
  defaults resolve relative to this sheet's own location, which for a
  packaged distro (<distro>/props/EngineCommon.props) is:
      <distro>/include, <distro>/deps/Cinder, <distro>/deps/LuisaCompute, ...
  Debug and Debug_Runtime intentionally share one define set and one link
  set: the prebuilt Debug library serves both consumers (RT_RUNTIME is a
  consumer-side behavioral define appended by the importing project, never
  a layout-affecting one — see docs/prebuilt_dist.md).
-->
<Project>
  <PropertyGroup>
    <!-- Overridable roots (first definition wins; set before importing) -->
{root("NewTypeEngineRoot", "$(MSBuildThisFileDirectory)..")}
{root("NewTypeEngineInclude", "$(NewTypeEngineRoot)\\include")}
{root("CinderRoot", "$(MSBuildThisFileDirectory)..\\deps\\Cinder")}
{root("CinderInclude", "$(CinderRoot)\\include")}
{root("CinderIncludeAngle", "$(CinderRoot)\\include\\ANGLE")}
    <!-- Static cinder libs with dynamic CRT (/MDd //MD), built from the
         cinder fork's Debug_MD // Release_MD configs. Shared cinder is not
         used by the engine (the D3D12 renderer is not exported from
         cinder.dll); "link dll" source-mode projects override these. -->
{root("CinderLibDebug", "$(CinderRoot)\\lib\\msw\\x64\\Debug_MD\\v143\\cinder.lib")}
{root("CinderLibRelease", "$(CinderRoot)\\lib\\msw\\x64\\Release_MD\\v143\\cinder.lib")}
{root("LuisaComputeRoot", "$(MSBuildThisFileDirectory)..\\deps\\LuisaCompute")}
{root("LuisaComputeInclude", "$(LuisaComputeRoot)\\include")}
{root("LuisaComputeSpdlog", "$(LuisaComputeRoot)\\src\\ext\\spdlog\\include")}
{root("LuisaComputeXxHash", "$(LuisaComputeRoot)\\src\\ext\\xxHash")}
{root("LuisaComputeMagicEnum", "$(LuisaComputeRoot)\\src\\ext\\magic_enum\\include")}
{root("LuisaComputeEASTL", "$(LuisaComputeRoot)\\src\\ext\\EASTL\\include")}
{root("LuisaComputeEABase", "$(LuisaComputeRoot)\\src\\ext\\EASTL\\packages\\EABase\\include\\Common")}
{root("LuisaComputeReproc", "$(LuisaComputeRoot)\\src\\ext\\reproc\\reproc\\include")}
{root("LuisaComputeReprocXX", "$(LuisaComputeRoot)\\src\\ext\\reproc\\reproc++\\include")}
{root("LuisaComputeMarl", "$(LuisaComputeRoot)\\src\\ext\\marl\\include")}
{root("LuisaComputeHalf", "$(LuisaComputeRoot)\\src\\ext\\half\\include")}
{root("LuisaComputeStb", "$(LuisaComputeRoot)\\src\\ext\\stb")}
{root("LuisaComputeLibDebug", "$(LuisaComputeRoot)\\build-dx-debug\\lib\\")}
{root("LuisaComputeLibRelease", "$(LuisaComputeRoot)\\build-dx\\lib\\")}
{root("LuisaComputeBinDebug", "$(LuisaComputeRoot)\\build-dx-debug\\bin")}
{root("LuisaComputeBinRelease", "$(LuisaComputeRoot)\\build-dx\\bin")}
{root("FidelityFXRoot", "$(MSBuildThisFileDirectory)..\\deps\\FidelityFX")}
{root("FidelityFXInclude", "$(FidelityFXRoot)\\include")}
{root("FidelityFXBin", "$(FidelityFXRoot)\\bin")}
{root("DlssRoot", "$(MSBuildThisFileDirectory)..\\deps\\NVIDIA\\DLSS")}
{root("DlssInclude", "$(DlssRoot)\\include")}
{root("DlssLibDebug", "$(DlssRoot)\\lib\\x64\\nvsdk_ngx_d_dbg.lib")}
{root("DlssLibRelease", "$(DlssRoot)\\lib\\x64\\nvsdk_ngx_d.lib")}
{root("DlssBin", "$(DlssRoot)\\bin")}
{root("SteamAudioRoot", "$(MSBuildThisFileDirectory)..\\deps\\Valve\\SteamAudio")}
{root("SteamAudioInclude", "$(SteamAudioRoot)\\include")}
{root("SteamAudioLib", "$(SteamAudioRoot)\\lib\\x64\\phonon.lib")}
{root("SteamAudioBin", "$(SteamAudioRoot)\\bin")}
    <!-- System libs every linking project needs (Recorder/MovieWriter -> MF;
         media decode -> D3D11; always-safe Windows system libs) -->
    <EngineSystemLibs>d3d12.lib;d3d11.lib;dxgi.lib;dxguid.lib;mf.lib;mfplat.lib;mfreadwrite.lib;mfuuid.lib;dbghelp.lib</EngineSystemLibs>
    <!-- The six LuisaCompute import libs (shared DLLs — plugin-scan backends
         and the shader hot-reload boundary need one shared copy). Used alone
         by the runtime shader DLL projects. -->
    <EngineLuisaLibsDebug>{luisa_libs_debug}</EngineLuisaLibsDebug>
    <EngineLuisaLibsRelease>{luisa_libs_release}</EngineLuisaLibsRelease>
    <!-- Full link set for exe/lib projects: cinder + system + luisa + the
         NGX stub (config-keyed: _d = /MD release CRT, _d_dbg = /MDd debug). -->
    <EngineLinkDepsDebug>$(CinderLibDebug);$(EngineSystemLibs);$(EngineLuisaLibsDebug);$(DlssLibDebug);$(SteamAudioLib)</EngineLinkDepsDebug>
    <EngineLinkDepsRelease>$(CinderLibRelease);$(EngineSystemLibs);$(EngineLuisaLibsRelease);$(DlssLibRelease);$(SteamAudioLib)</EngineLinkDepsRelease>
    <PlatformToolset Condition="'$(PlatformToolset)' == ''">v143</PlatformToolset>
  </PropertyGroup>
  <!-- Debug + Debug_Runtime share one blob (the prebuilt Debug lib serves
       both; importers append RT_RUNTIME for Debug_Runtime). -->
  <ItemDefinitionGroup Condition="'$(Configuration)' == 'Debug' or '$(Configuration)' == 'Debug_Runtime'">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <AdditionalOptions>/Zc:__cplusplus /utf-8 /Zc:preprocessor %(AdditionalOptions)</AdditionalOptions>
      <PreprocessorDefinitions>{PREPROC_DEBUG}</PreprocessorDefinitions>
      <BasicRuntimeChecks>EnableFastChecks</BasicRuntimeChecks>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <DisableSpecificWarnings>4244;4267;4068;5105</DisableSpecificWarnings>
    </ClCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)' == 'Release'">
    <ClCompile>
      <Optimization>MaxSpeed</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <AdditionalOptions>/Zc:__cplusplus /utf-8 /Zc:preprocessor %(AdditionalOptions)</AdditionalOptions>
      <PreprocessorDefinitions>{PREPROC_RELEASE}</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <DisableSpecificWarnings>4244;4267;4068;5105</DisableSpecificWarnings>
    </ClCompile>
  </ItemDefinitionGroup>
</Project>
"""


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_cpp(template_path: Path, class_name: str, project_name: str) -> str:
    src = template_path.read_text(encoding="utf-8")
    src = src.replace(
        '"NewTypeEngine - realtime pathtracing with ReSTIR"',
        f'"{project_name}"',
    )
    src = src.replace("NewTypeEngine", class_name)
    return src


# ---------------------------------------------------------------------------
# Cinder link modes (source-mode --link)
# ---------------------------------------------------------------------------

# Lines substituted for the @CINDER_LIBS@ placeholder in
# vc2022/EngineTemplate.props. Mirrors the EngineCommon.props defaults.
#
# dll (default) works against a stock cinder checkout: CinderLib* are the
# import libs for cinder.dll. static points CinderLib* at the /MD static
# libs built by the custom cinder fork (Debug_MD // Release_MD configs).
# Both modes link Debug_Runtime exactly like Debug (de-shared: the runtime
# shader DLLs reference zero cinder symbols, and shared cinder never
# exported the D3D12 renderer).
PROPS_CINDER_DLL = """\
    <!-- Cinder linked as a DLL (source-mode --link dll). CinderLib* are the
         import libs for cinder.dll and exist in a stock cinder checkout. -->
    <CinderLibDebug>$(CinderRoot)\\lib\\msw\\x64\\Debug_Shared\\v143\\cinder.lib</CinderLibDebug>
    <CinderLibRelease>$(CinderRoot)\\lib\\msw\\x64\\Release_Shared\\v143\\cinder.lib</CinderLibRelease>"""

PROPS_CINDER_STATIC = """\
    <!-- Static cinder libs with dynamic CRT (/MDd //MD), matching the engine's
         CRT. Built from the custom cinder fork's cinder.vcxproj
         Debug_MD // Release_MD configs. The original Debug // Release static
         configs (/MT) are untouched. -->
    <CinderLibDebug>$(CinderRoot)\\lib\\msw\\x64\\Debug_MD\\v143\\cinder.lib</CinderLibDebug>
    <CinderLibRelease>$(CinderRoot)\\lib\\msw\\x64\\Release_MD\\v143\\cinder.lib</CinderLibRelease>"""


def generate_props(engine_root: Path, link: str) -> str:
    """Read the engine-wide .props template (NOT the engine dev project's props).

    The template at vc2022/EngineTemplate.props carries only core engine
    dependencies (Cinder + LuisaCompute) plus an @CINDER_LIBS@ placeholder
    that is substituted per --link mode. Optional features (LCS, video)
    are added on demand via tools/cinder_blocks.py — see engine_addons/.
    """
    text = (engine_root / "vc2022" / "EngineTemplate.props").read_text(encoding="utf-8")
    token_line = "    @CINDER_LIBS@\n"
    if token_line not in text:
        raise RuntimeError(
            "vc2022/EngineTemplate.props is missing the @CINDER_LIBS@ "
            "placeholder line needed by tools/generate_project.py"
        )
    block = PROPS_CINDER_DLL if link == "dll" else PROPS_CINDER_STATIC
    # Replace only the standalone placeholder line, not any incidental
    # mention of the token in the surrounding comment.
    text = text.replace(token_line, block + "\n", 1)
    # Source mode compiles the engine from <project>/engine/ — point the
    # central sheet's engine include there (the sheet's own default points
    # at the sheet's parent, which is the project root, not engine/).
    anchor = "    <!-- Cinder Paths -->"
    extra = (
        "    <!-- Engine tree copied to <project>/engine/ by this generator;\n"
        "         consumed via vc2022/EngineCommon.props. -->\n"
        "    <NewTypeEngineInclude>$(ProjectDir)..\\engine\\include</NewTypeEngineInclude>\n\n"
    )
    if anchor not in text:
        raise RuntimeError(
            "vc2022/EngineTemplate.props is missing the Cinder Paths anchor "
            "needed by tools/generate_project.py"
        )
    return text.replace(anchor, extra + anchor, 1)


def generate_props_prebuilt(distro_root: Path) -> str:
    """Props for a --engine prebuilt project: everything resolves inside the
    packaged distro (NewTypeEngineRoot/deps/...)."""
    d = str(distro_root)
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<!--
  Prebuilt-engine props (generated). All dependencies resolve inside the
  NewTypeEngine distro; compile/link settings come from the distro's
  props/EngineCommon.props (imported by the vcxproj). Links the prebuilt
  static lib + static /MD cinder + LuisaCompute import libs.
-->
<Project>
  <PropertyGroup>
    <NewTypeEngineRoot>{d}</NewTypeEngineRoot>
    <CinderRoot>$(NewTypeEngineRoot)\\deps\\Cinder</CinderRoot>
    <CinderInclude>$(CinderRoot)\\include</CinderInclude>
    <CinderIncludeAngle>$(CinderRoot)\\include\\ANGLE</CinderIncludeAngle>
    <CinderLibDebug>$(CinderRoot)\\lib\\msw\\x64\\Debug_MD\\v143\\cinder.lib</CinderLibDebug>
    <CinderLibRelease>$(CinderRoot)\\lib\\msw\\x64\\Release_MD\\v143\\cinder.lib</CinderLibRelease>
    <LuisaComputeRoot>$(NewTypeEngineRoot)\\deps\\LuisaCompute</LuisaComputeRoot>
    <LuisaComputeInclude>$(LuisaComputeRoot)\\include</LuisaComputeInclude>
    <LuisaComputeSpdlog>$(LuisaComputeRoot)\\src\\ext\\spdlog\\include</LuisaComputeSpdlog>
    <LuisaComputeXxHash>$(LuisaComputeRoot)\\src\\ext\\xxHash</LuisaComputeXxHash>
    <LuisaComputeMagicEnum>$(LuisaComputeRoot)\\src\\ext\\magic_enum\\include</LuisaComputeMagicEnum>
    <LuisaComputeEASTL>$(LuisaComputeRoot)\\src\\ext\\EASTL\\include</LuisaComputeEASTL>
    <LuisaComputeEABase>$(LuisaComputeRoot)\\src\\ext\\EASTL\\packages\\EABase\\include\\Common</LuisaComputeEABase>
    <LuisaComputeReproc>$(LuisaComputeRoot)\\src\\ext\\reproc\\reproc\\include</LuisaComputeReproc>
    <LuisaComputeReprocXX>$(LuisaComputeRoot)\\src\\ext\\reproc\\reproc++\\include</LuisaComputeReprocXX>
    <LuisaComputeMarl>$(LuisaComputeRoot)\\src\\ext\\marl\\include</LuisaComputeMarl>
    <LuisaComputeHalf>$(LuisaComputeRoot)\\src\\ext\\half\\include</LuisaComputeHalf>
    <LuisaComputeStb>$(LuisaComputeRoot)\\src\\ext\\stb</LuisaComputeStb>
    <LuisaComputeLibDebug>$(LuisaComputeRoot)\\build-dx-debug\\lib\\</LuisaComputeLibDebug>
    <LuisaComputeLibRelease>$(LuisaComputeRoot)\\build-dx\\lib\\</LuisaComputeLibRelease>
    <LuisaComputeBinDebug>$(LuisaComputeRoot)\\build-dx-debug\\bin</LuisaComputeBinDebug>
    <LuisaComputeBinRelease>$(LuisaComputeRoot)\\build-dx\\bin</LuisaComputeBinRelease>
    <FidelityFXRoot>$(NewTypeEngineRoot)\\deps\\FidelityFX</FidelityFXRoot>
    <FidelityFXInclude>$(FidelityFXRoot)\\include</FidelityFXInclude>
    <FidelityFXBin>$(FidelityFXRoot)\\bin</FidelityFXBin>
    <DlssRoot>$(NewTypeEngineRoot)\\deps\\NVIDIA\\DLSS</DlssRoot>
    <DlssInclude>$(DlssRoot)\\include</DlssInclude>
    <DlssLibDebug>$(DlssRoot)\\lib\\x64\\nvsdk_ngx_d_dbg.lib</DlssLibDebug>
    <DlssLibRelease>$(DlssRoot)\\lib\\x64\\nvsdk_ngx_d.lib</DlssLibRelease>
    <DlssBin>$(DlssRoot)\\bin</DlssBin>
  </PropertyGroup>
</Project>
"""


def _postbuild(cfg: str, link: str) -> str:
    # Configs that link shared cinder need its DLLs (plus ANGLE) next to
    # the exe — in source mode that is every config under --link dll.
    # Prebuilt mode always links static cinder (no cinder copies at all).
    lc = "Debug" if cfg == "Debug_Runtime" else cfg
    parts = []
    if link == "dll":
        parts.append(f'xcopy /y "$(CinderBin{lc})\\cinder.dll" "$(OutDir)\\"')
        parts.append(f'xcopy /y "$(CinderBin{lc})\\d3dcompiler_46.dll" "$(OutDir)\\"')
        parts.append(f'xcopy /y "$(CinderBin{lc})\\libEGL.dll" "$(OutDir)\\"')
        parts.append(f'xcopy /y "$(CinderBin{lc})\\libGLESv2.dll" "$(OutDir)\\"')
    parts.append(f'xcopy /y "$(LuisaComputeBin{lc})\\*.dll" "$(OutDir)\\"')
    parts.append(f'xcopy /y "$(LuisaComputeBin{lc})\\luisa_embed_device_lib.exe" "$(OutDir)\\"')
    # FSR 3.1 upscaler runtime (loaded dynamically by Fsr31Backend; a
    # missing SDK just degrades to available() == false).
    parts.append(f'xcopy /y "$(FidelityFXBin)\\*.dll" "$(OutDir)\\"')
    # DLSS 4.5 runtime (feature DLLs found by the driver-side NGX core;
    # missing DLLs degrade to NGX init failure, not a crash).
    parts.append(f'xcopy /y "$(DlssBin)\\*.dll" "$(OutDir)\\"')
    return " &amp; ".join(parts)


def _config_groups(
    include_dirs: str,
    res_include: str,
    deps,
    postbuild,
    debug_defines: str,
    release_defines: str,
    rt_defines: str,
) -> str:
    """The three per-config ItemDefinitionGroups shared by source- and
    prebuilt-mode vcxprojs. Everything heavy lives in EngineCommon.props;
    these blocks only append project-specific extras (which also keeps the
    cinder_blocks.py marker injections working)."""
    return f"""\
  <!-- Debug_Runtime (appends the consumer-side RT_RUNTIME define) -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <PreprocessorDefinitions>{rt_defines}</PreprocessorDefinitions>
      <!-- Inherit-only anchor: gives cinder_blocks.py's AdditionalOptions
           suffix injection (e.g. the directml addon's bigobj flag) a
           per-config element to land on; the options themselves come from
           EngineCommon.props. -->
      <AdditionalOptions>%(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{deps("Debug_Runtime")}</AdditionalDependencies>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Windows</SubSystem>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{postbuild("Debug_Runtime")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <!-- Debug -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <PreprocessorDefinitions>{debug_defines}</PreprocessorDefinitions>
      <AdditionalOptions>%(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{deps("Debug")}</AdditionalDependencies>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Windows</SubSystem>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{postbuild("Debug")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <!-- Release -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <PreprocessorDefinitions>{release_defines}</PreprocessorDefinitions>
      <AdditionalOptions>%(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <ProjectReference>
      <LinkLibraryDependencies>true</LinkLibraryDependencies>
    </ProjectReference>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{deps("Release")}</AdditionalDependencies>
      <GenerateDebugInformation>false</GenerateDebugInformation>
      <GenerateMapFile>true</GenerateMapFile>
      <SubSystem>Windows</SubSystem>
      <OptimizeReferences>true</OptimizeReferences>
      <EnableCOMDATFolding></EnableCOMDATFolding>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{postbuild("Release")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>"""


def generate_vcxproj(
    project_name: str,
    class_name: str,
    project_guid: str,
    link: str,
) -> str:
    """Source-mode vcxproj: compiles the copied engine tree + App.cpp.
    Compile/link settings come from the local EngineCommon.props copy."""
    # The engine tree is copied to <project>/engine/ by main(); the vcxproj
    # references that local copy, never the engine checkout itself.
    engine_src = "../engine/src"

    include_dirs = "../include;%(AdditionalIncludeDirectories)"
    res_include = "$(CinderInclude);$(NewTypeEngineInclude)"
    empty = "%(PreprocessorDefinitions)"

    def _deps(cfg: str) -> str:
        # Debug_Runtime maps onto the Debug-flavoured link set (both the
        # luisa libs and the cinder lib choice from --link).
        key = "Release" if cfg == "Release" else "Debug"
        return f"$(EngineLinkDeps{key});%(AdditionalDependencies)"

    def _src_xml(p: str) -> str:
        return f'    <ClCompile Include="{engine_src}/{p}" />'

    def _hdr_xml(p: str) -> str:
        return f'    <ClInclude Include="../engine/include/{p}" />'

    def _local_hdr_xml(p: str) -> str:
        return f'    <ClInclude Include="../include/{p}" />'

    # Build ClCompile / ClInclude sections
    src_items = "\n".join(_src_xml(s) for s in ENGINE_SOURCES)
    rt_src_items = "\n".join(_src_xml(s) for s in RT_SOURCES)
    hdr_items = "\n".join(_hdr_xml(h) for h in ENGINE_HEADERS)
    rt_hdr_items = "\n".join(_hdr_xml(h) for h in RT_HEADERS)
    local_hdr_items = "\n".join(_local_hdr_xml(h) for h in LOCAL_HEADERS)

    config_groups = _config_groups(
        include_dirs,
        res_include,
        _deps,
        lambda cfg: _postbuild(cfg, link),
        empty,
        empty,
        f"RT_RUNTIME;{empty}",
    )

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="17.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug_Runtime|x64">
      <Configuration>Debug_Runtime</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{project_guid}</ProjectGuid>
    <RootNamespace>{project_name}</RootNamespace>
    <Keyword>Win32Proj</Keyword>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <Import Project="{project_name}.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
    <WholeProgramOptimization>true</WholeProgramOptimization>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <!-- Centralized compiler blob (defines, includes, options, link deps) -->
  <ImportGroup Label="EngineCommon">
    <Import Project="EngineCommon.props" />
  </ImportGroup>
  <ImportGroup Label="ExtensionSettings" />
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup>
    <_ProjectFileVersion>10.0.30319.1</_ProjectFileVersion>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">true</LinkIncremental>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'">true</LinkIncremental>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Release|x64'">false</LinkIncremental>
  </PropertyGroup>
{config_groups}
  <ItemGroup>
    <ResourceCompile Include="Resources.rc" />
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="$(CinderRoot)\\src\\glad\\glad.c" />
    <ClCompile Include="../src/{class_name}.cpp" />
{src_items}
  </ItemGroup>
  <ItemGroup Condition="'$(Configuration)'!='Release'">
{rt_src_items}
  </ItemGroup>
  <ItemGroup>
{hdr_items}
{local_hdr_items}
  </ItemGroup>
  <ItemGroup Condition="'$(Configuration)'!='Release'">
{rt_hdr_items}
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionSettings"></ImportGroup>
</Project>"""


def generate_vcxproj_prebuilt(
    project_name: str,
    class_name: str,
    project_guid: str,
) -> str:
    """Prebuilt-mode vcxproj: compiles ONLY src/<App>.cpp; links the
    distro's lib/{{Debug,Release}}/NewTypeEngine.lib (+ static cinder +
    luisa import libs via the distro's EngineCommon.props). Debug and
    Debug_Runtime both link the Debug lib."""
    include_dirs = "../include;%(AdditionalIncludeDirectories)"
    res_include = "$(CinderInclude);$(NewTypeEngineInclude)"
    empty = "%(PreprocessorDefinitions)"

    def _deps(cfg: str) -> str:
        key = "Release" if cfg == "Release" else "Debug"
        return (
            f"$(NewTypeEngineRoot)\\lib\\{key}\\NewTypeEngine.lib;"
            f"$(EngineLinkDeps{key});%(AdditionalDependencies)"
        )

    config_groups = _config_groups(
        include_dirs,
        res_include,
        _deps,
        lambda cfg: _postbuild(cfg, "static"),
        empty,
        empty,
        f"RT_RUNTIME;{empty}",
    )

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="17.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug_Runtime|x64">
      <Configuration>Debug_Runtime</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{project_guid}</ProjectGuid>
    <RootNamespace>{project_name}</RootNamespace>
    <Keyword>Win32Proj</Keyword>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <Import Project="{project_name}.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
    <WholeProgramOptimization>true</WholeProgramOptimization>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <!-- Centralized compiler blob from the distro -->
  <ImportGroup Label="EngineCommon">
    <Import Project="$(NewTypeEngineRoot)\\props\\EngineCommon.props" />
  </ImportGroup>
  <ImportGroup Label="ExtensionSettings" />
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'" Label="PropertySheets">
    <Import Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup>
    <_ProjectFileVersion>10.0.30319.1</_ProjectFileVersion>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">true</LinkIncremental>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'">true</LinkIncremental>
    <LinkIncremental Condition="'$(Configuration)|$(Platform)'=='Release|x64'">false</LinkIncremental>
  </PropertyGroup>
{config_groups}
  <ItemGroup>
    <ResourceCompile Include="Resources.rc" />
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="../src/{class_name}.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="../include/Resources.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionSettings"></ImportGroup>
</Project>"""


def generate_filters(
    project_name: str,
    class_name: str,
) -> str:
    """Generate .vcxproj.filters with full virtual folder hierarchy."""
    engine_inc = "../engine/include"
    engine_src = "../engine/src"

    # --- Filter definitions ---
    src_filters = collect_filter_names(ENGINE_SOURCES + RT_SOURCES, "Source Files")
    hdr_filters = collect_filter_names(ENGINE_HEADERS + RT_HEADERS, "Header Files")

    # Ensure intermediate filters exist (no source sits at newtype/ root)
    for required in ["Source Files\\newtype", "Header Files\\newtype"]:
        if required not in src_filters and required not in hdr_filters:
            (src_filters if required.startswith("Source") else hdr_filters).append(required)

    filter_defs = []
    # Top-level
    for name, guid in [("Source Files", GUID_SRC), ("Header Files", GUID_HDR), ("Resource Files", GUID_RES)]:
        filter_defs.append(f'    <Filter Include="{name}">\n      <UniqueIdentifier>{guid}</UniqueIdentifier>\n    </Filter>')

    # Sub-filters — deterministic GUID from filter name
    for f in src_filters + hdr_filters:
        if "\\" not in f:
            continue
        guid = make_guid(f"filter.{f}")
        filter_defs.append(f'    <Filter Include="{f}">\n      <UniqueIdentifier>{guid}</UniqueIdentifier>\n    </Filter>')

    filter_defs_xml = "\n".join(filter_defs)

    # --- ClCompile entries ---
    src_entries = []
    # App entry
    src_entries.append(
        f'    <ClCompile Include="../src/{class_name}.cpp">\n'
        f'      <Filter>Source Files</Filter>\n'
        f'    </ClCompile>'
    )
    # glad
    src_entries.append(
        f'    <ClCompile Include="$(CinderRoot)\\src\\glad\\glad.c">\n'
        f'      <Filter>Source Files\\newtype\\core</Filter>\n'
        f'    </ClCompile>'
    )
    for s in ENGINE_SOURCES + RT_SOURCES:
        filt = file_to_filter(s, "Source Files")
        src_entries.append(
            f'    <ClCompile Include="{engine_src}/{s}">\n'
            f'      <Filter>{filt}</Filter>\n'
            f'    </ClCompile>'
        )

    src_xml = "\n".join(src_entries)

    # --- ClInclude entries ---
    hdr_entries = []
    for h in ENGINE_HEADERS + RT_HEADERS:
        filt = file_to_filter(h, "Header Files")
        hdr_entries.append(
            f'    <ClInclude Include="{engine_inc}/{h}">\n'
            f'      <Filter>{filt}</Filter>\n'
            f'    </ClInclude>'
        )
    for h in LOCAL_HEADERS:
        filt = file_to_filter(h, "Header Files")
        hdr_entries.append(
            f'    <ClInclude Include="../include/{h}">\n'
            f'      <Filter>{filt}</Filter>\n'
            f'    </ClInclude>'
        )
    hdr_xml = "\n".join(hdr_entries)

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="17.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
{filter_defs_xml}
  </ItemGroup>
  <ItemGroup>
{src_xml}
  </ItemGroup>
  <ItemGroup>
{hdr_xml}
  </ItemGroup>
  <ItemGroup>
    <ResourceCompile Include="Resources.rc">
      <Filter>Resource Files</Filter>
    </ResourceCompile>
  </ItemGroup>
</Project>"""


def generate_filters_prebuilt(
    project_name: str,
    class_name: str,
) -> str:
    """Prebuilt-mode filters: just the app entry + Resources."""
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="17.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Filter Include="Source Files">
      <UniqueIdentifier>{GUID_SRC}</UniqueIdentifier>
    </Filter>
    <Filter Include="Header Files">
      <UniqueIdentifier>{GUID_HDR}</UniqueIdentifier>
    </Filter>
    <Filter Include="Resource Files">
      <UniqueIdentifier>{GUID_RES}</UniqueIdentifier>
    </Filter>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="../src/{class_name}.cpp">
      <Filter>Source Files</Filter>
    </ClCompile>
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="../include/Resources.h">
      <Filter>Header Files</Filter>
    </ClInclude>
  </ItemGroup>
  <ItemGroup>
    <ResourceCompile Include="Resources.rc">
      <Filter>Resource Files</Filter>
    </ResourceCompile>
  </ItemGroup>
</Project>"""


def generate_sln(project_name: str, project_guid: str) -> str:
    return f"""\
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.14.36717.8
MinimumVisualStudioVersion = 10.0.40219.1
Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "{project_name}", "{project_name}.vcxproj", "{project_guid}"
EndProject
Global
\tGlobalSection(SolutionConfigurationPlatforms) = preSolution
\t\tDebug_Runtime|x64 = Debug_Runtime|x64
\t\tDebug|x64 = Debug|x64
\t\tRelease|x64 = Release|x64
\tEndGlobalSection
\tGlobalSection(ProjectConfigurationPlatforms) = postSolution
\t\t{project_guid}.Debug_Runtime|x64.ActiveCfg = Debug_Runtime|x64
\t\t{project_guid}.Debug_Runtime|x64.Build.0 = Debug_Runtime|x64
\t\t{project_guid}.Debug|x64.ActiveCfg = Debug|x64
\t\t{project_guid}.Debug|x64.Build.0 = Debug|x64
\t\t{project_guid}.Release|x64.ActiveCfg = Release|x64
\t\t{project_guid}.Release|x64.Build.0 = Release|x64
\tEndGlobalSection
\tGlobalSection(SolutionProperties) = preSolution
\t\tHideSolutionNode = FALSE
\tEndGlobalSection
\tGlobalSection(ExtensibilityGlobals) = postSolution
\t\tSolutionGuid = {{{uuid.uuid4()}}}
\tEndGlobalSection
EndGlobal
"""


def generate_rc() -> str:
    return '#include "../include/Resources.h"\n\n1\tICON\t"..\\\\resources\\\\icon.ico"\n'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def copy_engine_tree(engine_root: Path, project_root: Path) -> tuple[int, int]:
    """Copy src/newtype + include/newtype into <project>/engine/.

    The whole tree is copied (not just the manifest files above) so every
    transitively included header exists in the new project and users can
    freely modify any part of the engine. The vcxproj still compiles only
    the curated manifest set.
    """
    pairs = [
        (engine_root / "src" / "newtype", project_root / "engine" / "src" / "newtype"),
        (engine_root / "include" / "newtype", project_root / "engine" / "include" / "newtype"),
    ]
    counts = []
    for src, dst in pairs:
        if not src.is_dir():
            print(f"Error: engine source tree not found at {src}")
            raise SystemExit(1)
        shutil.copytree(src, dst)
        counts.append(sum(1 for p in dst.rglob("*") if p.is_file()))
    return counts[0], counts[1]


def copy_ffx_sdk(engine_root: Path, project_root: Path) -> int:
    """Copy the vendored FidelityFX SDK (headers + runtime DLLs) into
    <project>/external/FidelityFX/.

    Fsr31Backend.cpp is part of the curated source manifest and includes
    <ffx_api/*.h> at compile time; the runtime DLLs are loaded dynamically.
    The .props resolve FidelityFXInclude/FidelityFXBin from
    $(ProjectDir)\\..\\external\\FidelityFX — the same relative layout as the
    engine checkout — so this copy makes them resolve for the new project.
    """
    src = engine_root / "external" / "FidelityFX"
    dst = project_root / "external" / "FidelityFX"
    if not src.is_dir():
        print(f"Error: vendored FidelityFX SDK not found at {src}")
        raise SystemExit(1)
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def copy_dlss_sdk(engine_root: Path, project_root: Path) -> int:
    """Copy the vendored NVIDIA DLSS SDK (headers + nvsdk_ngx_s.lib + feature
    DLLs) into <project>/external/NVIDIA/DLSS/.

    NgxContext.cpp / DlssSrBackend.cpp are part of the curated source
    manifest; the stub lib is linked via $(DlssLib) in EngineSystemLibs and
    the feature DLLs deploy via the post-build copy from $(DlssBin).
    """
    src = engine_root / "external" / "NVIDIA" / "DLSS"
    dst = project_root / "external" / "NVIDIA" / "DLSS"
    if not src.is_dir():
        print(f"Error: vendored DLSS SDK not found at {src}")
        raise SystemExit(1)
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def resolve_default_distro(engine_root: Path) -> Path | None:
    """Newest packaged distro under <engine checkout>/dist/, if any."""
    dist_dir = engine_root / "dist"
    if not dist_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in dist_dir.iterdir() if d.is_dir() and d.name.startswith("NewTypeEngine-")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def validate_distro(distro: Path) -> list[str]:
    """Required paths inside a packaged distro; empty list = valid."""
    problems = []
    for rel in [
        "props/EngineCommon.props",
        "include/newtype/core/Pipeline.h",
        "lib/Debug/NewTypeEngine.lib",
        "lib/Release/NewTypeEngine.lib",
        "deps/Cinder/lib/msw/x64/Debug_MD/v143/cinder.lib",
        "deps/Cinder/lib/msw/x64/Release_MD/v143/cinder.lib",
        "deps/LuisaCompute/build-dx/lib/luisa-core.lib",
        "deps/LuisaCompute/build-dx-debug/lib/luisa-core.lib",
        "deps/FidelityFX/include",
        "deps/NVIDIA/DLSS/include",
    ]:
        if not (distro / rel).exists():
            problems.append(rel)
    return problems


def emit_engine_common(engine_root: Path) -> int:
    """Write the canonical EngineCommon.props into the engine checkout."""
    out = engine_root / "vc2022" / "EngineCommon.props"
    out.write_text(engine_common_props_text(), encoding="utf-8", newline="\r\n")
    print(f"  Wrote:   {out}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new NewTypeEngine project from a DX or GL app template"
    )
    parser.add_argument("--path", help="Parent directory for the new project")
    parser.add_argument("--name", help="Project name (e.g. MyDemo)")
    parser.add_argument(
        "--engine",
        choices=["prebuilt", "source"],
        default="prebuilt",
        help="'prebuilt' (default) links the packaged NewTypeEngine.lib from a "
        "distro built by tools/package_dist.py — the project compiles only "
        "its own App.cpp; 'source' copies the engine tree into the project "
        "and compiles it (freely editable).",
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=None,
        help="Prebuilt mode: path to the unpacked distro root (the folder "
        "containing props/, include/, lib/, deps/). Default: newest "
        "dist/NewTypeEngine-*-win-x64 beside this checkout.",
    )
    parser.add_argument(
        "--link",
        choices=["dll", "static"],
        default="dll",
        help="Source mode only: how Debug/Release link cinder. 'dll' uses the "
        "shared cinder.lib import lib from a stock cinder checkout (default); "
        "'static' uses the /MD static cinder libs built by the custom cinder "
        "fork (Debug_MD // Release_MD configs). Debug_Runtime links like "
        "Debug either way. Ignored (always static cinder) in prebuilt mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["dx", "gl"],
        default="dx",
        help="App template / present path: 'dx' (default) generates from "
        "src/TemplateDX.cpp — Cinder presents through its D3D12 renderer and "
        "the Luisa device adopts its device (no GL context); 'gl' generates "
        "from src/Template.cpp — classic RendererGl present via the "
        "DX12->GL interop. The engine sources compiled are identical.",
    )
    parser.add_argument(
        "--emit-engine-common",
        action="store_true",
        help="Maintenance: regenerate vc2022/EngineCommon.props from this "
        "script's canonical text and exit (no project is created).",
    )
    args = parser.parse_args()

    engine_root = Path(__file__).resolve().parent.parent

    if args.emit_engine_common:
        return emit_engine_common(engine_root)

    if not args.path or not args.name:
        parser.error("--path and --name are required (unless --emit-engine-common)")

    project_name = args.name
    class_name = args.name + "App"
    project_root = Path(args.path).resolve() / project_name
    vcx_dir = project_root / "vc2022"

    template_name = APP_TEMPLATES[args.mode]
    template_path = engine_root / "src" / template_name

    if not template_path.exists():
        print(f"Error: app template not found at {template_path}")
        return 1

    # --- Resolve the engine consumption mode -----------------------------
    if args.engine == "prebuilt":
        distro = (
            args.engine_root.resolve() if args.engine_root else resolve_default_distro(engine_root)
        )
        if distro is None:
            print("Error: no prebuilt distro found (looked under "
                  f"{engine_root / 'dist'}).")
            print("       Build one with: python tools/package_dist.py")
            print("       Or pass --engine-root <distro>, or use --engine source.")
            return 1
        problems = validate_distro(distro)
        if problems:
            print(f"Error: {distro} is not a complete distro; missing:")
            for p in problems:
                print(f"       {p}")
            return 1
        print(f"  Engine:  prebuilt distro at {distro}")
    else:
        distro = None

    engine_dst = project_root / "engine"
    ffx_dst = project_root / "external" / "FidelityFX"
    if args.engine == "source" and (engine_dst.exists() or ffx_dst.exists()):
        print(f"Error: {engine_dst} (or {ffx_dst}) already exists — refusing to")
        print("       overwrite the project's engine copy (it may contain local")
        print("       modifications). Delete it manually or pick a fresh --path/--name.")
        return 1

    project_guid = make_guid(project_name)

    # Create directories
    for d in [
        vcx_dir,
        project_root / "src",
        project_root / "assets" / "models",
        project_root / "assets" / "textures",
        project_root / "resources",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate files
    files = {
        project_root / "src" / f"{class_name}.cpp": generate_cpp(
            template_path, class_name, project_name
        ),
        vcx_dir / f"{project_name}.vcxproj": (
            generate_vcxproj_prebuilt(project_name, class_name, project_guid)
            if distro is not None
            else generate_vcxproj(project_name, class_name, project_guid, args.link)
        ),
        vcx_dir / f"{project_name}.vcxproj.filters": (
            generate_filters_prebuilt(project_name, class_name)
            if distro is not None
            else generate_filters(project_name, class_name)
        ),
        vcx_dir / f"{project_name}.props": (
            generate_props_prebuilt(distro)
            if distro is not None
            else generate_props(engine_root, args.link)
        ),
        vcx_dir / f"{project_name}.sln": generate_sln(project_name, project_guid),
        vcx_dir / "Resources.rc": generate_rc(),
    }
    if distro is None:
        # Source mode gets its own copy of the central compiler blob.
        files[vcx_dir / "EngineCommon.props"] = engine_common_props_text()

    for path, content in files.items():
        path.write_text(content, encoding="utf-8", newline="\r\n")
        print(f"  Created: {path}")

    if distro is None:
        # Copy the engine source tree into the new project
        n_src, n_hdr = copy_engine_tree(engine_root, project_root)
        print(f"  Copied:  {engine_dst} ({n_src} sources, {n_hdr} headers)")

        # Copy the vendored FidelityFX SDK (Fsr31Backend compile+runtime deps)
        n_ffx = copy_ffx_sdk(engine_root, project_root)
        print(f"  Copied:  {ffx_dst} ({n_ffx} files)")

        # Copy the vendored NVIDIA DLSS SDK (NgxContext/DlssSrBackend deps)
        dlss_dst = project_root / "external" / "NVIDIA" / "DLSS"
        n_dlss = copy_dlss_sdk(engine_root, project_root)
        print(f"  Copied:  {dlss_dst} ({n_dlss} files)")

    # Copy static files from engine
    copies = [
        (engine_root / "include" / "Resources.h", project_root / "include" / "Resources.h"),
        (engine_root / "resources" / "icon.ico", project_root / "resources" / "icon.ico"),
    ]
    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            print(f"  Copied:  {dst}")

    print(f"\nProject '{project_name}' created at {project_root}")
    print(f"  Class: {class_name}")
    print(f"  GUID:  {project_guid}")
    print(f"  Mode:  {args.mode} (from {template_name})")
    if distro is not None:
        print(f"  Engine: prebuilt ({distro})")
    else:
        print(f"  Cinder link: {args.link}")
        print(f"  Engine copy: {engine_dst}")
    print(f"\nOpen {vcx_dir / (project_name + '.sln')} in Visual Studio 2022")
    return 0


if __name__ == "__main__":
    exit(main())
