#include "cinder/Log.h"

namespace newtype::render {

//==============================================================================
// Built-in Material Names
//==============================================================================

/// Names for built-in material types (index = MaterialType value)
const char* kBuiltinNames[] = {
    "Null",           // 0
    "Diffuse",        // 1
    "Conductor",      // 2
    "Dielectric",     // 3
    "Plastic",        // 4
    "Emissive",       // 5
    "Subsurface",     // 6
    "Clearcoat",      // 7
    "Sheen",          // 8
    "Anisotropy",     // 9
    "Iridescence",    // 10
    "ThinDielectric", // 11
    "Unlit",          // 12
    "Fabric"          // 13
};

} // namespace newtype::render
