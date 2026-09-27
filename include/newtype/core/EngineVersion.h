#pragma once

// ============================================================================
// NewTypeEngine version + ABI fingerprint (prebuilt-library drift guard)
// ============================================================================
// The prebuilt NewTypeEngine.lib is compiled against one frozen set of the
// layout/behavior macros below. Every TU (library and consumer) computes the
// same fingerprint from its own macro state; ShaderManager compares the
// consumer-side value against the library's and fails loudly on drift.
//
// Policy (docs/prebuilt_dist.md): prebuilt consumers must NOT /D any NT_*
// macro, EA_DLL, FMT_HEADER_ONLY, MARL_DLL, or switch the CRT flavor — the
// values ship frozen in the distro's EngineCommon.props. Source-mode
// projects compile engine + app together, so the compare always matches.
//
// RT_RUNTIME is deliberately NOT fingerprinted: plain-Debug and
// Debug_Runtime consumers intentionally share one Debug library and differ
// only in behavior defaults (loadShader hot-reload default, isRuntimeMode).
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstdio>

// Stamped by tools/package_dist.py when assembling a distro; "source" for
// dev checkouts and copy-tree generated projects (version recorded there
// only for diagnostics — source mode always matches itself).
#ifndef NT_ENGINE_VERSION
#define NT_ENGINE_VERSION "source"
#endif

// --- Fingerprint inputs -----------------------------------------------------
// Layout-affecting engine macros (Config.h is included by everything that
// includes this header transitively; include it here so the values below
// are always the effective ones, even if Config.h arrives later elsewhere).
#include "newtype/core/Config.h"

#ifdef _DEBUG
#define NT_ABI_DEBUG 1
#else
#define NT_ABI_DEBUG 0
#endif

#ifdef EA_DLL
#define NT_ABI_EA_DLL 1
#else
#define NT_ABI_EA_DLL 0
#endif

#ifdef FMT_HEADER_ONLY
#define NT_ABI_FMT_HEADER_ONLY 1
#else
#define NT_ABI_FMT_HEADER_ONLY 0
#endif

#ifdef MARL_DLL
#define NT_ABI_MARL_DLL 1
#else
#define NT_ABI_MARL_DLL 0
#endif

#define NT_ABI_STR_(x) #x
#define NT_ABI_STR(x)  NT_ABI_STR_(x)

// --- Build-time ABI enforcement (primary mechanism) ------------------------
// MSVC detect_mismatch embeds a record in every object file; the LINKER
// fails with LNK2038/LNK1319 when objects carrying different values for
// the same key are linked together. This catches header/lib drift at
// BUILD time and is immune to /OPT:REF dead-stripping (which can drop the
// runtime-only guards below). Constraints learned the hard way: MSVC does
// not macro-expand plain #pragma arguments (the __pragma-in-macro trick
// below is required), and each value must be a SINGLE string literal —
// concatenated literals are silently ignored — hence one record per input
// rather than one record for a combined string.
#define NT_DM_I(key, val) __pragma(detect_mismatch(key, val))
#define NT_DM(key, val)   NT_DM_I(key, val)
// NOTE: the version string is deliberately NOT a detect_mismatch key — the
// library is built against the repo header ("source") while the distro's
// headers are stamped with the release version, so a version key would
// fail every legitimate prebuilt link. Macro drift is what link-time
// enforcement is for; the version stays a diagnostic (runtime message).
NT_DM("NTABI_dbg", NT_ABI_STR(NT_ABI_DEBUG))
NT_DM("NTABI_NT_DEBUG_VIZ", NT_ABI_STR(NT_DEBUG_VIZ))
NT_DM("NTABI_NT_ALLOW_RASTER_FEATURES", NT_ABI_STR(NT_ALLOW_RASTER_FEATURES))
NT_DM("NTABI_NT_ENABLE_PROCEDURAL", NT_ABI_STR(NT_ENABLE_PROCEDURAL))
NT_DM("NTABI_NT_ENABLE_SHARC", NT_ABI_STR(NT_ENABLE_SHARC))
NT_DM("NTABI_NT_SHARC_COMPACT", NT_ABI_STR(NT_SHARC_COMPACT))
NT_DM("NTABI_NT_SHARC_64_BIT_ATOMICS", NT_ABI_STR(NT_SHARC_64_BIT_ATOMICS))
NT_DM("NTABI_NT_ENABLE_VIDEO_RECORDER", NT_ABI_STR(NT_ENABLE_VIDEO_RECORDER))
NT_DM("NTABI_NT_ENABLE_EDITOR", NT_ABI_STR(NT_ENABLE_EDITOR))
NT_DM("NTABI_EA_DLL", NT_ABI_STR(NT_ABI_EA_DLL))
NT_DM("NTABI_FMT_HEADER_ONLY", NT_ABI_STR(NT_ABI_FMT_HEADER_ONLY))
NT_DM("NTABI_MARL_DLL", NT_ABI_STR(NT_ABI_MARL_DLL))

// One string over every fingerprint input (runtime fingerprint below).
// Bump the leading tag when the input set changes so old/new guards never
// collide silently.
#define NT_ABI_STRING                                                                \
    "ntabi2;"                                                                        \
    "dbg=" NT_ABI_STR(NT_ABI_DEBUG) ";"                                              \
    "NT_DEBUG_VIZ=" NT_ABI_STR(NT_DEBUG_VIZ) ";"                                     \
    "NT_ALLOW_RASTER_FEATURES=" NT_ABI_STR(NT_ALLOW_RASTER_FEATURES) ";"             \
    "NT_ENABLE_PROCEDURAL=" NT_ABI_STR(NT_ENABLE_PROCEDURAL) ";"                     \
    "NT_ENABLE_SHARC=" NT_ABI_STR(NT_ENABLE_SHARC) ";"                               \
    "NT_SHARC_COMPACT=" NT_ABI_STR(NT_SHARC_COMPACT) ";"                             \
    "NT_SHARC_64_BIT_ATOMICS=" NT_ABI_STR(NT_SHARC_64_BIT_ATOMICS) ";"               \
    "NT_ENABLE_VIDEO_RECORDER=" NT_ABI_STR(NT_ENABLE_VIDEO_RECORDER) ";"             \
    "NT_ENABLE_EDITOR=" NT_ABI_STR(NT_ENABLE_EDITOR) ";"                             \
    "EA_DLL=" NT_ABI_STR(NT_ABI_EA_DLL) ";"                                          \
    "FMT_HEADER_ONLY=" NT_ABI_STR(NT_ABI_FMT_HEADER_ONLY) ";"                        \
    "MARL_DLL=" NT_ABI_STR(NT_ABI_MARL_DLL)

namespace newtype::detail {

/// FNV-1a 64 over the ABI string — computed at compile time in every TU.
constexpr std::uint64_t engineAbiFingerprint() noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (char c : NT_ABI_STRING) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= 1099511628211ull;
    }
    return h;
}

// Defined in EngineVersion.cpp (compiled into the library / source-mode exe):
// the values the binary was actually built with.
const char* engineLibVersion() noexcept;
std::uint64_t engineLibAbiFingerprint() noexcept;

/// Compare this TU's fingerprint against the linked binary's; abort with a
/// loud message on mismatch. NOT memoized — each calling TU must observe its
/// own macro state (a lib-side TU always matches the lib; a drifted consumer
/// TU must fail even if another TU already ran a passing check).
/// (newtype::detail — NOT nt::detail — because NewType.h aliases nt.)
inline void engineAbiCheckRaw() noexcept {
    if (engineAbiFingerprint() == engineLibAbiFingerprint())
        return;
    std::fprintf(
        stderr,
        "[NewTypeEngine] FATAL: engine header/lib ABI mismatch.\n"
        "  header fingerprint: %016llx (%s)\n"
        "  library fingerprint: %016llx (%s)\n"
        "  The engine headers this TU was compiled with disagree with the\n"
        "  linked NewTypeEngine library on a frozen macro (NT_*, EA_DLL,\n"
        "  FMT_HEADER_ONLY, MARL_DLL, or the CRT flavor). Rebuild against\n"
        "  the matching library, or stop overriding the frozen macros\n"
        "  (see docs/prebuilt_dist.md).\n",
        static_cast<unsigned long long>(engineAbiFingerprint()),
        NT_ENGINE_VERSION,
        static_cast<unsigned long long>(engineLibAbiFingerprint()),
        engineLibVersion());
    std::abort();
}

/// Memoized variant for hot-path backstops (ShaderManager dispatch): the
/// compare runs once per process; TU coverage comes from the eager
/// _engineAbiTUGuard below, which always calls the raw check.
inline void engineAbiCheck() noexcept {
    static const bool ok = (engineAbiCheckRaw(), true);
    (void)ok;
}

} // namespace newtype::detail

// Eager per-TU guard: every TU that includes this header (the consumer's
// App.cpp, via ShaderManager.h / NewType.h) runs the RAW check at static
// initialization, so header/lib drift aborts loudly BEFORE main() even when
// the app never calls a checking inline (the dispatch templates above are
// only a backstop). Raw, not memoized: a passing lib-side TU must not
// mask a drifted consumer TU. Note: TUs that must not link
// EngineVersion.cpp (the runtime shader DLLs) do not include this
// header — keep it that way.
namespace newtype::detail {
[[maybe_unused]] const bool _engineAbiTUGuard = (engineAbiCheckRaw(), true);
} // namespace newtype::detail
