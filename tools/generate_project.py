#!/usr/bin/env python3
"""NewTypeEngine project generator.

Creates a new VS2022 project from Template.cpp with a renamed app class.

Usage:
    python tools/generate_project.py --path D:/Projects --name MyDemo
    python tools/generate_project.py --path . --name TestProject
"""

import argparse
import os
import uuid
from pathlib import Path, PureWindowsPath

# ---------------------------------------------------------------------------
# Engine source/header manifest (paths relative to engine src/ or include/)
# ---------------------------------------------------------------------------

ENGINE_SOURCES = [
    "newtype/core/DxGLInterop.cpp",
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
    "newtype/feature/MotionBlurFeature.cpp",
    "newtype/feature/PointCloud.cpp",
    "newtype/feature/RasterContext.cpp",
    "newtype/feature/Trail.cpp",
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
    "newtype/render/Sampling.cpp",
    "newtype/render/TextureConverter.cpp",
    "newtype/scene/DeformableMesh.cpp",
    "newtype/scene/Geometry.cpp",
    "newtype/scene/InstancedMesh.cpp",
    "newtype/scene/Interaction.cpp",
    "newtype/scene/LightShape.cpp",
    "newtype/scene/MeshShape.cpp",
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
    "newtype/core/Config.h",
    "newtype/core/DxGLInterop.h",
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
    "newtype/feature/MotionBlurFeature.h",
    "newtype/feature/PointCloud.h",
    "newtype/feature/RasterBase.h",
    "newtype/feature/RasterContext.h",
    "newtype/feature/Trail.h",
    "newtype/media/NativeTextureDesc.h",
    "newtype/media/VideoDecoderD3D11.h",
    "newtype/media/VideoPlayer.h",
    "newtype/media/VideoTextureBridge.h",
    "newtype/plugin/MovieWriter.h",
    "newtype/render/BSDF.h",
    "newtype/render/BSSRDF.h",
    "newtype/render/DDAMarch.h",
    "newtype/render/GIReservoir.h",
    "newtype/render/GIShading.h",
    "newtype/render/LightSampler.h",
    "newtype/render/Material.h",
    "newtype/render/MaterialPool.h",
    "newtype/render/MetalData.h",
    "newtype/render/Packing.h",
    "newtype/render/PassDI.h",
    "newtype/render/PassDenoiser.h",
    "newtype/render/PassGI.h",
    "newtype/render/PassSSS.h",
    "newtype/render/ProceduralTrace.h",
    "newtype/render/ReSTIR.h",
    "newtype/render/RelaxDenoiser.h",
    "newtype/render/Sampling.h",
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
    "newtype/util/Camera.h",
    "newtype/util/CommandBuffer.h",
    "newtype/util/CompileProfiler.h",
    "newtype/util/LutLoader.h",
    "newtype/util/Mesh.h",
    "newtype/util/Noise.h",
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# VS well-known filter GUIDs
GUID_SRC = "{4FC737F1-C7A5-4376-A066-2A32D752A2FF}"
GUID_HDR = "{93995380-89BD-4b04-88EB-625FBE52EBFB}"
GUID_RES = "{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}"


def make_guid(name: str) -> str:
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_DNS, f"newtype.{name}")) + "}"


def rel_path(from_dir: Path, to_path: Path) -> str:
    return PureWindowsPath(os.path.relpath(to_path, from_dir)).as_posix()


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
# Preprocessor defines
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
PREPROC_RT = PREPROC_COMMON.replace("{MODE}", "_DEBUG").replace(
    "{EXTRA}", "RT_RUNTIME;CINDER_SHARED;"
)
PREPROC_RELEASE = PREPROC_COMMON.replace("{MODE}", "NDEBUG").replace("{EXTRA}", "")

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


def generate_props(engine_root: Path) -> str:
    """Read the engine-wide .props template (NOT the engine dev project's props).

    The template at vc2022/EngineTemplate.props carries only core engine
    dependencies (Cinder + LuisaCompute). Optional features (LCS, video)
    are added on demand via tools/cinder_blocks.py — see engine_addons/.
    """
    return (engine_root / "vc2022" / "EngineTemplate.props").read_text(encoding="utf-8")


def generate_vcxproj(
    project_name: str,
    class_name: str,
    project_guid: str,
    engine_root: Path,
    vcx_dir: Path,
) -> str:
    engine_inc = rel_path(vcx_dir, engine_root / "include")
    engine_src = rel_path(vcx_dir, engine_root / "src")

    include_dirs = (
        f"{engine_inc};"
        "$(CinderInclude);$(CinderIncludeAngle);"
        "$(LuisaComputeInclude);$(LuisaComputeSpdlog);$(LuisaComputeXxHash);"
        "$(LuisaComputeMagicEnum);$(LuisaComputeEASTL);$(LuisaComputeEABase);"
        "$(LuisaComputeReproc);$(LuisaComputeReprocXX);$(LuisaComputeMarl);"
        "$(LuisaComputeHalf);$(LuisaComputeStb)"
    )
    res_include = f"$(CinderInclude);{engine_inc}"

    def _deps(cfg: str) -> str:
        return (
            f"$(CinderLib{cfg});d3d12.lib;dxgi.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-dsl.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-xir.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-osl.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-runtime.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-ast.lib;"
            f"$(LuisaComputeLib{cfg})\\luisa-core.lib;"
            "dbghelp.lib;%(AdditionalDependencies)"
        )

    def _postbuild(cfg: str) -> str:
        return (
            f'xcopy /y "$(CinderBin{cfg})\\cinder.dll" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(CinderBin{cfg})\\d3dcompiler_46.dll" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(CinderBin{cfg})\\libEGL.dll" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(CinderBin{cfg})\\libGLESv2.dll" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(LuisaComputeBin{cfg})\\*.dll" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(LuisaComputeBin{cfg})\\luisa_nvrtc.exe" "$(OutDir)\\" &amp; '
            f'xcopy /y "$(LuisaComputeBin{cfg})\\luisa_embed_device_lib.exe" "$(OutDir)\\"'
        )

    def _src_xml(p: str) -> str:
        return f'    <ClCompile Include="{engine_src}/{p}" />'

    def _hdr_xml(p: str) -> str:
        return f'    <ClInclude Include="{engine_inc}/{p}" />'

    def _local_hdr_xml(p: str) -> str:
        return f'    <ClInclude Include="../include/{p}" />'

    # Build ClCompile / ClInclude sections
    src_items = "\n".join(_src_xml(s) for s in ENGINE_SOURCES)
    rt_src_items = "\n".join(_src_xml(s) for s in RT_SOURCES)
    hdr_items = "\n".join(_hdr_xml(h) for h in ENGINE_HEADERS)
    rt_hdr_items = "\n".join(_hdr_xml(h) for h in RT_HEADERS)
    local_hdr_items = "\n".join(_local_hdr_xml(h) for h in LOCAL_HEADERS)

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
  <!-- Debug_Runtime -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug_Runtime|x64'">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <AdditionalOptions>/Zc:__cplusplus /utf-8 /Zc:preprocessor %(AdditionalOptions)</AdditionalOptions>
      <PreprocessorDefinitions>{PREPROC_RT}</PreprocessorDefinitions>
      <BasicRuntimeChecks>EnableFastChecks</BasicRuntimeChecks>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <PrecompiledHeader></PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <DisableSpecificWarnings>4244;4267;4068;5105</DisableSpecificWarnings>
    </ClCompile>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{_deps("Debug")}</AdditionalDependencies>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Windows</SubSystem>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{_postbuild("Debug")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <!-- Debug -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <AdditionalOptions>/Zc:__cplusplus /utf-8 /Zc:preprocessor %(AdditionalOptions)</AdditionalOptions>
      <PreprocessorDefinitions>{PREPROC_DEBUG}</PreprocessorDefinitions>
      <BasicRuntimeChecks>EnableFastChecks</BasicRuntimeChecks>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <PrecompiledHeader></PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <DisableSpecificWarnings>4244;4267;4068;5105</DisableSpecificWarnings>
    </ClCompile>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{_deps("Debug")}</AdditionalDependencies>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Windows</SubSystem>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{_postbuild("Debug")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <!-- Release -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>{include_dirs}</AdditionalIncludeDirectories>
      <AdditionalOptions>/Zc:__cplusplus /utf-8 /Zc:preprocessor %(AdditionalOptions)</AdditionalOptions>
      <PreprocessorDefinitions>{PREPROC_RELEASE}</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <PrecompiledHeader></PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <DisableSpecificWarnings>4244;4267;4068;5105</DisableSpecificWarnings>
    </ClCompile>
    <ProjectReference>
      <LinkLibraryDependencies>true</LinkLibraryDependencies>
    </ProjectReference>
    <ResourceCompile>
      <AdditionalIncludeDirectories>{res_include}</AdditionalIncludeDirectories>
    </ResourceCompile>
    <Link>
      <AdditionalDependencies>{_deps("Release")}</AdditionalDependencies>
      <GenerateDebugInformation>false</GenerateDebugInformation>
      <GenerateMapFile>true</GenerateMapFile>
      <SubSystem>Windows</SubSystem>
      <OptimizeReferences>true</OptimizeReferences>
      <EnableCOMDATFolding></EnableCOMDATFolding>
      <RandomizedBaseAddress>false</RandomizedBaseAddress>
      <DataExecutionPrevention></DataExecutionPrevention>
    </Link>
    <PostBuildEvent>
      <Command>{_postbuild("Release")}</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ResourceCompile Include="Resources.rc" />
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="$(CinderRoot)\\src\\glad\\glad.c" />
    <ClCompile Include="../src/{class_name}.cpp" />
{src_items}
  </ItemGroup>
  <ItemGroup Condition="'$(Configuration)'=='Debug_Runtime'">
{rt_src_items}
  </ItemGroup>
  <ItemGroup>
{hdr_items}
{local_hdr_items}
  </ItemGroup>
  <ItemGroup Condition="'$(Configuration)'=='Debug_Runtime'">
{rt_hdr_items}
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionSettings"></ImportGroup>
</Project>"""


def generate_filters(
    project_name: str,
    class_name: str,
    engine_root: Path,
    vcx_dir: Path,
) -> str:
    """Generate .vcxproj.filters with full virtual folder hierarchy."""
    engine_inc = rel_path(vcx_dir, engine_root / "include")
    engine_src = rel_path(vcx_dir, engine_root / "src")

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new NewTypeEngine project from Template.cpp"
    )
    parser.add_argument("--path", required=True, help="Parent directory for the new project")
    parser.add_argument("--name", required=True, help="Project name (e.g. MyDemo)")
    args = parser.parse_args()

    project_name = args.name
    class_name = args.name + "App"
    project_root = Path(args.path).resolve() / project_name
    vcx_dir = project_root / "vc2022"

    engine_root = Path(__file__).resolve().parent.parent
    template_path = engine_root / "src" / "Template.cpp"

    if not template_path.exists():
        print(f"Error: Template.cpp not found at {template_path}")
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
        vcx_dir / f"{project_name}.vcxproj": generate_vcxproj(
            project_name, class_name, project_guid, engine_root, vcx_dir
        ),
        vcx_dir / f"{project_name}.vcxproj.filters": generate_filters(
            project_name, class_name, engine_root, vcx_dir
        ),
        vcx_dir / f"{project_name}.props": generate_props(engine_root),
        vcx_dir / f"{project_name}.sln": generate_sln(project_name, project_guid),
        vcx_dir / "Resources.rc": generate_rc(),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8", newline="\r\n")
        print(f"  Created: {path}")

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
    print(f"  Engine: {engine_root}")
    print(f"\nOpen {vcx_dir / (project_name + '.sln')} in Visual Studio 2022")
    return 0


if __name__ == "__main__":
    exit(main())
