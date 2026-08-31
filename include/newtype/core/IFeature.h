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

class IFeature {
public:
    virtual ~IFeature() = default;

    // Which injection point this feature runs at
    [[nodiscard]] virtual FeaturePoint point() const = 0;

    // Called once when addFeature() is called — compile shaders, create images
    virtual void onInit(luisa::compute::Device& device) {}

    // Called on window resize — recreate feature-owned images
    virtual void onResize(luisa::compute::Device& device, uint width, uint height) {}

    // Called every frame at the registered injection point
    virtual void onExecute(luisa::compute::Stream& stream, const FeatureContext& ctx) = 0;

    // Enable/disable without removing from pipeline
    void setEnabled(bool enabled) noexcept { _enabled = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return _enabled; }

    // ImGui panel. Default: no UI. Subclasses open a CollapsingHeader named
    // after the feature so the UI tree stays consistent with DI/GI/Denoiser.
    virtual void drawUi() {}

    virtual void load(const nlohmann::json& file) {}
    [[nodiscard]] virtual nlohmann::json toJson() const = 0;

private:
    bool _enabled = true;
};

} // namespace newtype::core
