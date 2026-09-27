#include "newtype/core/EngineVersion.h"

// Compiled into the library (and into source-mode exes, where the "library"
// is the consumer's own build — the guard then trivially matches). The
// values below freeze the macro state this binary was built with; the
// header-side engineAbiFingerprint() computed in consumer TUs must agree.

namespace newtype::detail {

const char* engineLibVersion() noexcept { return NT_ENGINE_VERSION; }

std::uint64_t engineLibAbiFingerprint() noexcept { return engineAbiFingerprint(); }

} // namespace newtype::detail
