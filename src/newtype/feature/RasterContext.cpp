#include "newtype/feature/RasterContext.h"
#include "newtype/feature/RasterBase.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/FeatureContext.h"

namespace newtype::feature {

using namespace luisa;
using namespace luisa::compute;

RasterContext::RasterContext(Device& device)
    : _device(device) {
    _rasterStream = device.create_stream(StreamTag::GRAPHICS);
    _computeDoneEvent = device.create_timeline_event();
    _rasterDoneEvent  = device.create_timeline_event();
    _compileMergeShader(device);
}

void RasterContext::_compileMergeShader(Device& device) {
    _mergeShader = device.compile<2>(
        [&](ImageFloat gbuf_depth, ImageUInt gbuf_vis, ImageFloat gbuf_bary,
            ImageFloat raster_depth, ImageFloat raster_vis, ImageFloat raster_bary) noexcept {
        set_name("raster_merge");
        UInt2 coord = dispatch_id().xy();

        Float raster_d = raster_depth.read(coord).x;
        Float existing_depth = gbuf_depth.read(coord).x;

        $if(Expr{ raster_d > 0.0f & raster_d < existing_depth }) {
            gbuf_depth.write(coord, make_float4(raster_d));

            Float4 vis_f = raster_vis.read(coord);
            gbuf_vis.write(coord, make_uint4(
                as<uint>(vis_f.x), as<uint>(vis_f.y), 0u, 0u));

            Float4 bary = raster_bary.read(coord);
            gbuf_bary.write(coord, bary);
        };
    });
}

bool RasterContext::_needsOpaquePass(span<RasterBase*> features) {
    for (auto* f : features)
        if (!f->isTransparent() && f->shouldExecute()) return true;
    return false;
}

bool RasterContext::_needsTransparentPass(span<RasterBase*> features) {
    for (auto* f : features)
        if (f->isTransparent() && f->shouldExecute()) return true;
    return false;
}

void RasterContext::resize(Device& device, uint width, uint height) {
    if (_width == width && _height == height) return;
    _width  = width;
    _height = height;

    _depthBuffer = device.create_depth_buffer(DepthFormat::D32, luisa::make_uint2(width, height));
    _rasterDepth = device.create_image<float>(PixelStorage::FLOAT2, width, height, 1u, false, true);
    _rasterVis   = device.create_image<float>(PixelStorage::FLOAT4, width, height, 1u, false, true);
    _rasterBary  = device.create_image<float>(PixelStorage::HALF4,  width, height, 1u, false, true);

    // OIT images will be re-allocated lazily when first transparent pass runs
    _hasOIT = false;
    _oitAccum = {};
    _oitLogReveal = {};
}

void RasterContext::dispatchBatch(Stream& computeStream,
                                   const core::FeatureContext& ctx,
                                   span<RasterBase*> features) {
    uint w = ctx.width(), h = ctx.height();
    resize(_device, w, h);

    _hasTransparency = _needsTransparentPass(features);

    // Allocate OIT images on demand
    if (_hasTransparency && !_hasOIT) {
        _oitAccum     = _device.create_image<float>(PixelStorage::HALF4, w, h, 1u, false, true);
        _oitLogReveal = _device.create_image<float>(PixelStorage::FLOAT1, w, h, 1u, false, true);
        _hasOIT = true;
    }

    uint64_t fence = ++_timelineFrame;

    // 1. All features voxelize on compute stream (both opaque + transparent)
    for (auto* feat : features)
        feat->onVoxelize(computeStream, ctx);

    // 2. Signal compute done -> raster stream can start
    computeStream << _computeDoneEvent.signal(fence);
    _rasterStream << _computeDoneEvent.wait(fence);

    // 2b. Wait for Renderer::stream() to finish G-buffer + presample so
    // shared buffers (env CDF, material) are not accessed concurrently.
    // Without this, the transparent pass on _rasterStream races with
    // the presample pass on Renderer::stream() on the same buffers.
    if (ctx.pipeline.renderReadyFence() != 0u)
        _rasterStream << ctx.pipeline.renderReadyEvent().wait(ctx.pipeline.renderReadyFence());

    // 3. Opaque pass (guarded — skip if no opaque features)
    if (_needsOpaquePass(features)) {
        auto* rasterExt = _device.extension<RasterExt>();
        _rasterStream << rasterExt->clear_render_target(_rasterDepth.view(), make_float4(0.0f));
        _rasterStream << rasterExt->clear_render_target(_rasterVis.view(),   make_float4(0.0f));
        _rasterStream << rasterExt->clear_render_target(_rasterBary.view(),  make_float4(0.0f));
        _rasterStream << _depthBuffer.clear(1.0f);

        bool anyDrawn = false;
        for (auto* feat : features) {
            if (!feat->isTransparent())
                anyDrawn |= feat->onRasterExecute(_rasterStream, *this, ctx);
        }

        if (anyDrawn) {
            _rasterStream << _mergeShader(
                ctx.frame.gbufDepth,
                ctx.frame.gbufVis,
                ctx.frame.gbufBaryMotion,
                _rasterDepth,
                _rasterVis,
                _rasterBary
            ).dispatch(w, h);
        }
    }

    // 4. Transparent pass (guarded — skip if no transparent features)
    if (_hasTransparency && _hasOIT) {
        auto* rasterExt = _device.extension<RasterExt>();
        _rasterStream << rasterExt->clear_render_target(_oitAccum.view(),     make_float4(0.0f));
        _rasterStream << rasterExt->clear_render_target(_oitLogReveal.view(), make_float4(0.0f));

        for (auto* feat : features) {
            if (feat->isTransparent())
                feat->onRasterExecuteTransparent(_rasterStream, *this, ctx);
        }
    }

    // 5. Signal raster done. _featureGbufEvent signal is deferred to the
    // caller so post-transparent work (voxel grid dilate/summary) can run
    // on _rasterStream before the main stream is unblocked.
    _rasterStream << _rasterDoneEvent.signal(fence);
    ctx.pipeline.setFeatureGbufFence(fence);
}

} // namespace newtype::feature
