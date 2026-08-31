#include "newtype/util/Camera.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;

namespace newtype::util {

    // Halton sequence for deterministic low-discrepancy sampling
    static float halton(uint index, uint base) {
        float f = 1.0f, result = 0.0f;
        while (index > 0) {
            f /= static_cast<float>(base);
            result += f * static_cast<float>(index % base);
            index /= base;
        }
        return result;
    }

    Camera::Camera(const ci::app::WindowRef& window)
        : _window(window) {
        _ciCam  = ci::CameraPersp(window->getWidth(), window->getHeight(),
            45.0f, 0.1f, 100.0f);
        _ciCam.setEyePoint(ci::vec3(0.0f, 0.2f, 2.0f));
        _ciCam.lookAt(ci::vec3(0.0f, 0.0f, 0.0f));
        _camUi          = ci::CameraUi(&_ciCam);// , window);
        _cam            = fromCiCamera(_ciCam);
        _cam.view_proj  = tolc(_ciCam.getProjectionMatrix() * _ciCam.getViewMatrix());
        _cam.prev_view_proj = _cam.view_proj;
        _cam.prev_position  = _cam.position;
        _cam.jitter         = ::luisa::float2{0.0f};
        _cam.prev_jitter    = ::luisa::float2{0.0f};
        _prev_cam       = _cam;
        _dirty          = false;
    }

    const nlohmann::json Camera::toJson() {
        nlohmann::json file;
        file["eye"] = _ciCam.getEyePoint();
        file["orient"] = _ciCam.getOrientation();
        file["fov"] = _ciCam.getFov();
        file["near"] = _ciCam.getNearClip();
        file["far"] = _ciCam.getFarClip();
        return file;
    }

    void Camera::load(const nlohmann::json& file) {
        _ciCam.setFov(file["fov"]);
        _ciCam.setEyePoint(file["eye"]);
        _ciCam.setOrientation(file["orient"]);
        _ciCam.setNearClip(file["near"]);
        _ciCam.setFarClip(file["far"]);
        _dirty = true;
    }

    newtype::util::CameraData Camera::fromCiCamera(
        const ci::Camera& ci_camera)
    {
        // Get eye position from ci
        ci::vec3 eye      = ci_camera.getEyePoint();
        // Get view direction (note: ci's mW is NEGATIVE view direction)
        ci::vec3 view_dir = ci_camera.getViewDirection();
        // Get world up
        ci::vec3 world_up = ci_camera.getWorldUp();

        // Get FOV and aspect ratio
        float fov    = ci_camera.getFov();
        float aspect = ci_camera.getAspectRatio();
        float near_clip = ci_camera.getNearClip();
        float far_clip  = ci_camera.getFarClip();

        return buildCameraBasis(
            ::luisa::float3{ eye.x, eye.y, eye.z },
            ::luisa::float3{ view_dir.x, view_dir.y, view_dir.z },
            ::luisa::float3{ world_up.x, world_up.y, world_up.z },
            fov,
            aspect,
            near_clip,
            far_clip
        );
    }

    newtype::util::CameraData Camera::createDefault(
        const ::luisa::float3& position,
        const ::luisa::float3& target,
        const ::luisa::float3& up,
        float fov,
        float aspect)
    {
        ::luisa::float3 front = normalize(target - position);
        return buildCameraBasis(position, front, up, fov, aspect, 0.1f, 100.0f);
    }

    newtype::util::CameraData Camera::buildCameraBasis(
        const ::luisa::float3& position,
        const ::luisa::float3& front,
        const ::luisa::float3& world_up,
        float fov,
        float aspect,
        float near_clip,
        float far_clip)
    {
        newtype::util::CameraData camera;
        camera.position = position;
        camera.prev_position = position;
        camera.front    = front;
        camera.prev_front = front;
        camera.fov      = fov;
        camera.aspect   = aspect;
        camera.near_clip = near_clip;
        camera.far_clip  = far_clip;

        // Build orthonormal basis
        // Right = normalize(front × world_up)
        camera.right = normalize(cross(front, world_up));
        // Recalculate up to ensure orthonormality
        // Up = normalize(right × front)
        camera.up    = normalize(cross(camera.right, front));
        camera.prev_right = camera.right;
        camera.prev_up    = camera.up;

        return camera;
    }

    void Camera::mouseDown(ci::app::MouseEvent& event) {
        _camUi.mouseDown(event);
        _dirty = true;
    }

    void Camera::mouseDrag(ci::app::MouseEvent& event) {
        _camUi.mouseDrag(event);
        _dirty = true;
    }

    void Camera::mouseWheel(ci::app::MouseEvent& event) {
        _camUi.mouseWheel(event);
        _dirty = true;
    }

    void Camera::update() {
        if (_dirty || _camUi.isEnabled()) {
            // Snapshot previous frame's view_proj for motion vectors
            _prev_cam           = _cam;

            ci::vec3 eye        = _ciCam.getEyePoint();
            ci::vec3 view_dir   = _ciCam.getViewDirection();
            ci::vec3 up, right;
            _ciCam.getBillboardVectors(&right, &up);

            _cam.position = ::luisa::float3{ eye.x, eye.y, eye.z };
            _cam.front    = ::luisa::float3{ view_dir.x, view_dir.y, view_dir.z };
            _cam.fov      = _ciCam.getFov();
            _cam.aspect   = _ciCam.getAspectRatio();
            _cam.near_clip = _ciCam.getNearClip();
            _cam.far_clip  = _ciCam.getFarClip();
            _cam.right    = ::luisa::float3{ right.x, right.y, right.z };
            _cam.up       = ::luisa::float3{ up.x, up.y, up.z };
        
            // Update current frame view_proj
            _cam.view_proj      = tolc(_ciCam.getProjectionMatrix() * _ciCam.getViewMatrix());
            // prev_view_proj comes from the snapshot above
            _cam.prev_view_proj = _prev_cam.view_proj;
            _cam.prev_position  = _prev_cam.position;
            _cam.prev_front     = _prev_cam.front;
            _cam.prev_right     = _prev_cam.right;
            _cam.prev_up        = _prev_cam.up;
        }

        // Snapshot prev jitter unconditionally: the _prev_cam snapshot above is
        // gated on camera motion, but jitter advances every frame.
        _cam.prev_jitter = _cam.jitter;

        // Compute Halton(2,3) jitter for TAA (sub-pixel offset in NDC space)
        _cam.jitter = ::luisa::float2{
            (halton(_frameIndex, 2u) - 0.5f) * 2.0f / static_cast<float>(_window->getWidth()),
            (halton(_frameIndex, 3u) - 0.5f) * 2.0f / static_cast<float>(_window->getHeight())
        };
        _frameIndex++;
    }
}
