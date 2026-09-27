#pragma once

#include <cstdint>

namespace newtype::render {

/// Layout version for everything that crosses the material-callable DLL
/// boundary by header inclusion: SurfaceData, MaterialData, and the
/// SurfaceResolveFn signature. The hot-reloader rebuilds a DLL only when its
/// SOURCE .cpp timestamp changes — a header-only layout change would leave a
/// stale-layout DLL loaded (silent UB). Every callable DLL exports
/// ntCallableAbiVersion() returning the version it was built against;
/// CallableDLLLoader::loadDLL force-rebuilds on mismatch or absence
/// (docs/resolver_params_abi_plan.md, track B2).
///
/// BUMP THIS whenever SurfaceData / MaterialData / the resolver signature
/// changes layout. History:
///   1 — initial guard, introduced with SurfaceData.instance_index (track B2).
inline constexpr std::uint32_t kCallableAbiVersion = 1u;

} // namespace newtype::render
