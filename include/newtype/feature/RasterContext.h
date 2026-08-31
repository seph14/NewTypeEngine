#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/backends/ext/raster_ext.hpp>

namespace newtype::core {
class Pipeline;
struct FrameContext;
struct FeatureContext;
}

namespace newtype::feature {

class RasterBase;

/// Owns one set of raster resources (depth buffer, RTVs, stream, merge shader)
/// and orchestrates the batch lifecycle for all RasterBase features:
///   one clear → opaque features draw → merge → transparent features draw into OIT.
///
/// Held by Pipeline. RasterBase subclasses access resources via the
/// RasterContext& passed to onRasterExecute().
class RasterContext {
public:
    explicit RasterContext(luisa::compute::Device& device);

    /// Execute the full raster batch for the given features.
    /// 1. All features voxelize on computeStream
    /// 2. Signal -> raster stream waits
    /// 3. Opaque pass: clear RTVs, draw opaque features, merge into G-buffer
    /// 4. Transparent pass: clear OIT buffers, draw transparent features
    /// 5. Signal done + featureGbufEvent
    void dispatchBatch(luisa::compute::Stream& computeStream,
                       const core::FeatureContext& ctx,
                       luisa::span<RasterBase*> features);

    /// Resize raster images. No-op if dimensions unchanged.
    void resize(luisa::compute::Device& device, luisa::uint width, luisa::uint height);

    // Accessors for subclass draw commands — opaque pass
    [[nodiscard]] luisa::compute::DepthBuffer&  depthBuffer() noexcept { return _depthBuffer; }
    [[nodiscard]] luisa::compute::Image<float>& rasterDepth()  noexcept { return _rasterDepth; }
    [[nodiscard]] luisa::compute::Image<float>& rasterVis()    noexcept { return _rasterVis; }
    [[nodiscard]] luisa::compute::Image<float>& rasterBary()   noexcept { return _rasterBary; }

    // Accessors for subclass draw commands — transparent pass
    [[nodiscard]] luisa::compute::Image<float>& oitAccum()     noexcept { return _oitAccum; }
    [[nodiscard]] luisa::compute::Image<float>& oitLogReveal() noexcept { return _oitLogReveal; }

    /// Whether raster images have been allocated (i.e., dispatchBatch ran at least once).
    [[nodiscard]] bool hasResources() const noexcept { return _width != 0u; }
    /// Whether OIT buffers are allocated (i.e., transparent features exist).
    [[nodiscard]] bool hasOIT() const noexcept { return _hasOIT; }
    [[nodiscard]] bool requireOITBlit() const noexcept { return _hasOIT && _hasTransparency; }

    /// Access the raster stream for post-transparent work (e.g., voxel grid update).
    [[nodiscard]] luisa::compute::Stream& rasterStream() noexcept { return _rasterStream; }

private:
    void _compileMergeShader(luisa::compute::Device& device);

    static bool _needsOpaquePass(luisa::span<RasterBase*> features);
    static bool _needsTransparentPass(luisa::span<RasterBase*> features);

    luisa::compute::Device& _device;

    // Opaque raster resources (shared across all RasterBase features)
    luisa::compute::DepthBuffer   _depthBuffer;
    luisa::compute::Image<float>  _rasterDepth;    // FLOAT2 RTV: linear depth (R) + packed motion (G)
    luisa::compute::Image<float>  _rasterVis;      // FLOAT4 RTV: vis encoding
    luisa::compute::Image<float>  _rasterBary;     // HALF4  RTV: barycentrics + oct normal

    // OIT resources (allocated only when transparent features exist)
    luisa::compute::Image<float>  _oitAccum;       // HALF4 RTV: RGB = α·c·w, A = α·w
    luisa::compute::Image<float>  _oitLogReveal;   // FLOAT RTV: Σ log(1-α) — additive accumulation
    bool _hasOIT = false, _hasTransparency = false;

    luisa::compute::Stream        _rasterStream;

    // GPU-side sync events
    luisa::compute::TimelineEvent _computeDoneEvent;
    luisa::compute::TimelineEvent _rasterDoneEvent;
    uint64_t _timelineFrame = 0u;

    // Merge shader: raster RTVs -> G-buffer
    luisa::compute::Shader<2,
        luisa::compute::Image<float>,  // gbuf_depth (in/out)
        luisa::compute::Image<uint>,   // gbuf_vis (in/out)
        luisa::compute::Image<float>,  // gbuf_bary_motion (in/out)
        luisa::compute::Image<float>,  // raster_depth (read)
        luisa::compute::Image<float>,  // raster_vis (read)
        luisa::compute::Image<float>   // raster_bary (read)
    > _mergeShader;

    luisa::uint _width = 0u, _height = 0u;
};

} // namespace newtype::feature
