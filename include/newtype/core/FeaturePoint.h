#pragma once

namespace newtype::core {

enum class FeaturePoint {
    PreUpdate,         // Before geometry update — user GPU sim (VATMesh, particles, fluids)
    AfterGBuffer,      // G-buffer available (depth, vis, normals, motion vectors)
    AfterShade,        // Raw shade output available (noisy, pre-denoiser)
    AfterDenoiser,     // Denoised HDR, pre OIT composite + glass tint
    AfterGlassTint,    // Denoised HDR with OIT particles + glass tint applied (pre-tonemap)
    AfterToneMap,      // Final LDR output available (pre-present)
    Count
};

} // namespace newtype::core
