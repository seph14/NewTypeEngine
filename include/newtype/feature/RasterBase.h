#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include "newtype/core/IFeature.h"
#include "newtype/core/FeatureContext.h"

namespace newtype::feature {

class RasterContext;

/// Pixel output shared by all raster-into-G-buffer features.
struct PointPixelOut {
    luisa::float4 depth_out;  // MRT 0: linear depth
    luisa::float4 vis_out;    // MRT 1: vis encoding (bitcast uint→float)
    luisa::float4 bary_out;   // MRT 2: barycentrics (xy) + oct-encoded normal (zw)
};

// Transparent output: McGuire weighted blended OIT (2 MRT targets)
struct TransparentPixelOut {
    luisa::float4 accum;       // MRT 0: RGB = α·lit·w(z), A = α·w(z)
    luisa::float4 logReveal;   // MRT 1: R = log(1-α), GBA = 0 (additive blend on both)
};

} // namespace newtype::feature

LUISA_STRUCT(newtype::feature::PointPixelOut, depth_out, vis_out, bary_out) {};
LUISA_STRUCT(newtype::feature::TransparentPixelOut, accum, logReveal) {};

namespace newtype::feature {

/// Abstract base for features that rasterize into the G-buffer.
/// RasterContext (owned by Pipeline) holds the shared raster infrastructure
/// and orchestrates the batch lifecycle: one clear, all features draw, one merge.
///
/// Subclasses implement:
///   onRasterInit()   — compile shaders, create mesh format, init raster state
///   onRasterExecute() — issue draw calls on the raster stream
///   onVoxelize()     — voxelize geometry into the shared VoxelGrid
class RasterBase : public core::IFeature {
    friend class RasterContext;
public:
    enum class BlendMode : uint8_t { Opaque, Transparent };

    [[nodiscard]] core::FeaturePoint point() const final {
        return core::FeaturePoint::AfterGBuffer;
    }

    void onInit(luisa::compute::Device& device) final {
        onRasterInit(device);
        _compiled = true;
    }

    void onExecute(luisa::compute::Stream& stream, const core::FeatureContext& ctx) final {
        // No-op: RasterContext handles the batch lifecycle.
    }

    /// Hardware rasterization rides the fixed-function perspective clip
    /// space (view_proj) — cannot express non-linear projections.
    [[nodiscard]] bool requiresPerspectiveProjection() const noexcept final {
        return true;
    }

    /// Whether voxelize + raster should run this frame. Default: enabled() && compiled.
    /// Subclasses can override to add particle-count checks etc.
    [[nodiscard]] virtual bool shouldExecute() const { return enabled() && _compiled; }

    void setBlendMode(BlendMode mode) { _blendMode = mode; }
    [[nodiscard]] BlendMode blendMode() const { return _blendMode; }
    [[nodiscard]] bool isTransparent() const { return _blendMode == BlendMode::Transparent; }

    void setReceiveShadow(bool v) { _receiveShadow = v; }
    [[nodiscard]] bool receiveShadow() const { return _receiveShadow; }
    void setCastShadow(bool v) { _castShadow = v; }
    [[nodiscard]] bool castShadow() const { return _castShadow; }

protected:
    luisa::compute::Device& _device;

    explicit RasterBase(luisa::compute::Device& device) : _device(device) {}

    /// Called once during onInit(). Compile raster shaders, set up mesh format + raster state.
    virtual void onRasterInit(luisa::compute::Device& device) = 0;

    /// Called every frame on the shared raster stream after RTV clear.
    /// Return true if draw commands were issued (merge shader will run).
    /// Return false to skip contribution (e.g., debug visualization path).
    virtual bool onRasterExecute(luisa::compute::Stream& rasterStream,
                                  RasterContext& rc,
                                  const core::FeatureContext& ctx) = 0;

    /// Called every frame on the compute stream before the raster pass.
    /// Use for voxelization into the shared VoxelGrid.
    virtual void onVoxelize(luisa::compute::Stream& computeStream,
                            const core::FeatureContext& ctx) = 0;

    /// Called for transparent features during the transparent pass.
    /// Default returns false (no draw issued). Subclasses override with
    /// their forward-lit + OIT shader.
    virtual bool onRasterExecuteTransparent(luisa::compute::Stream& rasterStream,
                                             RasterContext& rc,
                                             const core::FeatureContext& ctx) { return false; }

    bool _compiled = false;
    BlendMode _blendMode = BlendMode::Opaque;
    bool _receiveShadow = true;
    bool _castShadow = true;
};

} // namespace newtype::feature
