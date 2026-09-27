#pragma once

/**
 * @brief CustomMaterialShader DLL Export Interface
 *
 * The DLL exports registration functions that receive function pointers
 * from the host application. This avoids shared singleton issues across
 * DLL boundaries.
 */

#include <string>
#include <any>
#include <cstdint>

#ifdef CUSTOM_MATERIAL_EXPORTS
#define CUSTOM_MATERIAL_API __declspec(dllexport)
#elif !defined(RT_RUNTIME)
#define CUSTOM_MATERIAL_API
#else
#define CUSTOM_MATERIAL_API __declspec(dllimport)
#endif

/// Function pointer: register a callable by name, returns type ID.
/// Host provides this — the DLL calls it for each custom callable.
using CallableRegisterFn = std::uint32_t(*)(const char* name, std::any resolve);

/// Function pointer: clear all custom callables.
using CallableClearFn = void(*)();

// --- ABI v2: runtime tuning params (docs/resolver_params_abi_plan.md) --------

/// Bindless slot of the params buffer inside the texture bindless array the
/// resolver receives as `tex` (engine side: MaterialPool::kResolverParamsBindlessSlot).
/// Must stay in lockstep with the engine constant.
#define CUSTOM_MATERIAL_PARAMS_BINDLESS_SLOT 0u

/// Bindless slot of the PER-INSTANCE params buffer (engine side:
/// MaterialPool::kInstanceParamsBindlessSlot — rows of 4 float4 keyed by
/// TLAS instance row, read via instance_params(tex, s.instance_index, i)).
/// Must stay in lockstep with the engine constant.
#define CUSTOM_MATERIAL_INSTANCE_PARAMS_BINDLESS_SLOT 1u

/// One named scalar tuning param. min/max/default drive the engine's UI slider.
struct ResolverParamDesc {
    const char* name;
    float       min_v;
    float       max_v;
    float       def_v;
};

/// Function pointer: register a callable's params, returns the callable's
/// float4 base in the params buffer (read shader-side via
/// resolver_params(tex, base, i) — see Shading.h). Returns ~0u on overflow.
using ParamRegisterFn = std::uint32_t(*)(const char* callable,
                                         const ResolverParamDesc* descs,
                                         std::uint32_t count);

extern "C" {

/**
 * @brief Register custom material callables via host-provided function pointers.
 *
 * @param registerFn  Host function to register a callable (returns type ID)
 * @param clearFn     Host function to clear all custom callables
 */
CUSTOM_MATERIAL_API void registerMaterialCallables(
    CallableRegisterFn registerFn, CallableClearFn clearFn);

/**
 * @brief ABI v2: registration with runtime tuning params.
 *
 * Same callable semantics as v1, plus per-callable param descriptors: the
 * engine uploads defaults into a params buffer the resolver reads via
 * resolver_params(tex, base, i); editing values host-side (UI sliders) never
 * recompiles shaders. Callables may be registered without params (call
 * paramFn only for those that need it). The loader prefers this export and
 * falls back to registerMaterialCallables when absent.
 */
CUSTOM_MATERIAL_API void registerMaterialCallables2(
    CallableRegisterFn registerFn, ParamRegisterFn paramFn, CallableClearFn clearFn);

/**
 * @brief Clear custom callables (called before DLL unload).
 *
 * @param clearFn Host function to clear all custom callables
 */
CUSTOM_MATERIAL_API void unregisterMaterialCallables(CallableClearFn clearFn);

/**
 * @brief Layout-ABI version the DLL was built against (track B2).
 *
 * The engine compares this against newtype::render::kCallableAbiVersion and
 * force-rebuilds the DLL on mismatch or absence — SurfaceData / MaterialData
 * layouts cross the exe<->DLL boundary by header inclusion, and the
 * hot-reload watch only sees this .cpp's timestamp. Implemented at the
 * bottom of CustomMaterialShader.cpp; every callable DLL should export the
 * same one-liner.
 */
CUSTOM_MATERIAL_API std::uint32_t ntCallableAbiVersion(void);

} // extern "C"
