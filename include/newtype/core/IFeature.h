#pragma once
#include "newtype/core/FeaturePoint.h"
#include "cinder/Json.h"

namespace luisa::compute {
    class Device;
    class Stream;
    template<typename T> class Image;
}

namespace newtype::core {

struct FeatureContext;
class Pipeline;

class IFeature {
public:
    virtual ~IFeature() = default;

    // Which injection point this feature runs at
    [[nodiscard]] virtual FeaturePoint point() const = 0;

    // Called once when addFeature() is called — compile shaders, create images
    virtual void onInit(luisa::compute::Device& device) {}

    // Called with the current render dims once at registration (addFeature)
    // and again on window resize — (re)create feature-owned images. Features
    // must NOT rely on a resize event for their first allocation.
    virtual void onResize(luisa::compute::Device& device, uint width, uint height) {}

    // Called every frame at the registered injection point
    virtual void onExecute(luisa::compute::Stream& stream, const FeatureContext& ctx) = 0;

    // Enable/disable without removing from pipeline
    void setEnabled(bool enabled) noexcept { _enabled = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return _enabled; }

    // Features that cannot express non-linear projections (hardware
    // rasterization through the fixed-function clip space, view_proj-based
    // overlays) are skipped while a non-perspective camera projection is
    // active (docs/non_perspective_camera_report.md §3 raster guard).
    [[nodiscard]] virtual bool requiresPerspectiveProjection() const noexcept {
        return false;
    }

    // perf R2 item 13: when true (and this is the ONLY enabled feature at
    // its injection point), the pipeline routes the tonemap output into the
    // persistent "fxaa_temp" BYTE4 image instead of the display target —
    // the feature then reads the temp and writes the display target itself,
    // skipping its full-screen input copy. Must reflect whether the feature
    // will actually dispatch this frame (the pipeline consults it BEFORE
    // tonemap; onExecute must skip exactly when this returns false, or the
    // display target would stay unwritten on routed frames). Base: enabled().
    [[nodiscard]] virtual bool wantsTonemapTempRoute(const Pipeline&) const noexcept {
        return enabled();
    }

    // ImGui panel. Default: no UI. Subclasses open a CollapsingHeader named
    // after the feature so the UI tree stays consistent with DI/GI/Denoiser.
    virtual void drawUi() {}

    virtual void load(const nlohmann::json& file) {}
    [[nodiscard]] virtual nlohmann::json toJson() const = 0;

private:
    bool _enabled = true;
};

} // namespace newtype::core
