#pragma once
#include <luisa/luisa-compute.h>
#include "newtype/core/FrameContext.h"

namespace newtype {
namespace core {

class Pipeline;

struct FeatureContext {
    // Read-only access to pipeline resources (G-buffer, scene, accum buffers, camera)
    const FrameContext& frame;

    // Write target for this injection point:
    //   AfterGBuffer:  Pipeline render target (typically unused at this stage)
    //   AfterShade:    Pipeline render target (available for custom output)
    //   AfterDenoiser: Pipeline render target (raw denoised HDR, pre OIT + glass tint)
    //   AfterGlassTint: Pipeline render target (denoised HDR with OIT particles + glass tint)
    //   AfterToneMap:  Display target (tone-mapped LDR)
    luisa::compute::Image<float>& renderTarget;

    // Device reference for on-demand resource creation
    luisa::compute::Device& device;

    // Pipeline reference for temp image pool access
    Pipeline& pipeline;

    // Convenience accessors
    [[nodiscard]] uint width()       const noexcept { return frame.width; }
    [[nodiscard]] uint height()      const noexcept { return frame.height; }
    [[nodiscard]] uint frameCount()  const noexcept { return frame.frameCount; }
};

} // namespace core
} // namespace newtype
