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
 * @brief Clear custom callables (called before DLL unload).
 *
 * @param clearFn Host function to clear all custom callables
 */
CUSTOM_MATERIAL_API void unregisterMaterialCallables(CallableClearFn clearFn);

} // extern "C"
