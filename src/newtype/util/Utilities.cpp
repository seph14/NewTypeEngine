#include "newtype/util/Utilities.h"

using namespace std;

namespace newtype {
namespace util {
void setHDRColorTint(luisa::float3& hdr, const luisa::float3& c) {
    float intensity = luisa::max(.0001f, luisa::max(luisa::max(hdr.x, hdr.y), hdr.z));
    auto  oldTint = hdr / intensity;
    float oldLum = luisa::max(.01f, luisa::max(luisa::max(oldTint.x, oldTint.y), oldTint.z));
    float newLum = luisa::max(.01f, luisa::max(luisa::max(c.x, c.y), c.z));
    hdr = c * oldLum / newLum * intensity;
}

void setHDRColorIntensity(luisa::float3& hdr, float intensity) {
    float prevint = luisa::max(.0001f, luisa::max(
                        luisa::max(hdr.x, hdr.y), hdr.z));
    auto  tint = luisa::clamp(hdr / prevint, 0.f, 1.f);
    hdr = tint * intensity;
}
}
}