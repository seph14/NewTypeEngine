#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <cinder/Camera.h>
#include "cinder/CameraUi.h"
#include "cinder/Json.h"

/**
 * @brief Camera structure for LuisaCompute DSL path tracing
 *
 * Defined at global scope for LUISA_STRUCT macro compatibility.
 * The macro generates DSL code that must be at global/nested namespace scope.
 *
 * Mirrors the Camera struct from LuisaCompute's test_path_tracing_camera.cpp.
 */

namespace newtype::util {
struct CameraData {
    ::luisa::float3 position;       // Camera eye point
    ::luisa::float3 prev_position;  // Previous frame camera eye point (for virtual motion)
    ::luisa::float3 front;          // View direction (normalized, pointing forward)
    ::luisa::float3 up;             // Up vector (normalized)
    ::luisa::float3 right;          // Right vector (normalized)
    ::luisa::float3 prev_front;     // Previous frame view direction
    ::luisa::float3 prev_up;        // Previous frame up vector
    ::luisa::float3 prev_right;     // Previous frame right vector
    float           fov;            // Vertical field of view in degrees
    float           aspect;         // Width / height aspect ratio
    float           near_clip;      // Near clip plane distance
    float           far_clip;       // Far clip plane distance
    ::luisa::float4x4 view_proj;       // Current frame projection * view matrix
    ::luisa::float4x4 prev_view_proj;  // Previous frame projection * view matrix
    ::luisa::float2 jitter;             // Sub-pixel offset in NDC space (TAA)
    ::luisa::float2 prev_jitter;        // Previous frame jitter (jitter-aware motion vectors)
};

} // namespace newtype::util

// Register CameraData with LuisaCompute DSL (must be outside namespace)
LUISA_STRUCT(newtype::util::CameraData, position, prev_position, front, up, right, prev_front, prev_up, prev_right, fov, aspect, near_clip, far_clip, view_proj, prev_view_proj, jitter, prev_jitter) {
    // Generate ray from camera through normalized pixel coordinate [-1, 1]
    // Applies jitter offset for TAA anti-aliasing
    [[nodiscard]] auto generate_ray(::luisa::compute::Expr<::luisa::float2> p) const noexcept {
        auto pj = p + jitter;  // apply sub-pixel jitter
        auto fov_radians = radians(fov);
        auto half_tan = tan(0.5f * fov_radians);
        auto wi_local = make_float3(pj.x * aspect * half_tan, pj.y * half_tan, -1.0f);
        auto wi_world = normalize(wi_local.x * right + wi_local.y * up - wi_local.z * front);
        return ::luisa::compute::make_ray(position, wi_world);
    }

    // Project world position to NDC [-1, 1] using current frame matrix
    [[nodiscard]] auto project(::luisa::compute::Expr<::luisa::float3> world_pos) const noexcept {
        auto clip = view_proj * make_float4(world_pos, 1.0f);
        return clip.xy() / clip.w;
    }

    // Project world position to NDC [-1, 1] using previous frame matrix
    [[nodiscard]] auto project_prev(::luisa::compute::Expr<::luisa::float3> world_pos) const noexcept {
        auto clip = prev_view_proj * make_float4(world_pos, 1.0f);
        return clip.xy() / clip.w;
    }
};

namespace newtype::util {

class Camera;
typedef std::unique_ptr<Camera> CamPtr;

/**
 * @brief Helper class to convert ci::Camera to LuisaCamera
 *
 * Converts between ci's camera representation and LuisaCompute's
 * path tracing camera structure.
 */
class Camera {
protected:
    CameraData      _cam;
    CameraData      _prev_cam; // Previous frame snapshot for motion vectors
    ci::CameraPersp _ciCam;
    ci::CameraUi    _camUi;
    ci::app::WindowRef _window;
    bool            _dirty;
    uint            _frameIndex = 0u; // For Halton jitter sequence

    Camera(const ci::app::WindowRef& window);
public:
    static CamPtr create(const ci::app::WindowRef& window) { return CamPtr(new Camera(window)); }

    [[nodiscard]] ci::CameraPersp& ciCam() noexcept { return _ciCam; }
    [[nodiscard]] ci::CameraUi&    camUi() noexcept { return _camUi; }
    [[nodiscard]] CameraData&      cam()   noexcept { return _cam; }

    void mouseDown (ci::app::MouseEvent& event);
    void mouseDrag (ci::app::MouseEvent& event);
    void mouseWheel(ci::app::MouseEvent& event);

    [[nodiscard]] const nlohmann::json toJson();
    void load(const nlohmann::json& file);

    void update();

private:
    /**
     * @brief Convert a ci::Camera to LuisaCamera for path tracing
     *
     * @param ci_camera The ci camera (typically CameraPersp)
     * @param resolution The render resolution for aspect ratio calculation
     * @return LuisaCamera Struct suitable for passing to DSL shaders
     */
    [[nodiscard]] static CameraData fromCiCamera(
        const ci::Camera& ci_camera);

    /**
     * @brief Create a default path tracing camera
     *
     * @param position Camera position
     * @param target Look-at target
     * @param up Up vector
     * @param fov Vertical FOV in degrees
     * @return LuisaCamera
     */
    [[nodiscard]] static CameraData createDefault(
        const ::luisa::float3& position = {0.0f, 0.2f, 2.0f},
        const ::luisa::float3& target   = {0.0f, 0.0f, 0.0f},
        const ::luisa::float3& up       = {0.0f, 1.0f, 0.0f},
        float fov = 45.0f,
        float aspect = 16.0f / 9.0f);

    /**
     * @brief Build orthonormal basis from front vector and up vector
     */
    [[nodiscard]] static CameraData buildCameraBasis(
        const ::luisa::float3& position,
        const ::luisa::float3& front,
        const ::luisa::float3& world_up,
        float fov,
        float aspect,
        float near_clip = 0.1f,
        float far_clip = 100.0f);
};

} // namespace newtype
