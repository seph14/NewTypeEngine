#pragma once

// Resource defines are appended here by the Bundler (D:\Projects\Bundler) as
// `#define NAME CINDER_RESOURCE( ../resources/, <asset path>, <id>, <TYPE> )`.
// Resources.rc includes this file too; under RC_INVOKED each define becomes an
// embed statement, so the .rc and the C++ side always stay in sync.
// The list is intentionally empty: nothing is baked into the exe — the engine
// lib must stay free of embedded resources, and dev scenes load assets from
// the assets/ directory at runtime (loadAsset literal idiom).
#include "cinder/CinderResources.h"
