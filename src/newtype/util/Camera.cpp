#include "newtype/util/Camera.h"
#include "newtype/util/TypeConv.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>

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
        _cam.projection     = 0u;
        _cam.face_count     = 0u;
        _cam.fisheye_fov    = 180.0f;
        for (auto& f : _cam.faces) f = RoomFaceData{};
        _prev_cam       = _cam;
        _dirty          = false;
    }

    void Camera::setProjection(CameraProjection p) noexcept {
        if (_projection == p) return;
        _projection = p;
        _cam.projection = static_cast<uint32_t>(p);
        _dirty = true;
    }

    void Camera::setFisheyeFov(float fovDeg) noexcept {
        _fisheyeFov = std::clamp(fovDeg, 1.0f, 359.0f);
        _cam.fisheye_fov = _fisheyeFov;
    }

    void Camera::setRoomRig(const RoomRigConfig& rig) {
        _rig = rig;
        if (_rig.faces.size() > 6u)
            _rig.faces.resize(6u);
        // Face table content is rebuilt from the live camera basis every
        // update() (rig faces follow the viewport camera pose); here we only
        // fix the face count and make the projection consistent.
        _projection = CameraProjection::RoomRig;
        _cam.projection = static_cast<uint32_t>(CameraProjection::RoomRig);
        _cam.face_count = static_cast<uint32_t>(_rig.faces.size());
        _dirty = true;
    }

    void Camera::_buildRoomFace(const RoomRigFaceConfig& cfg, uint32_t faceIdx,
                                uint32_t cols, uint32_t rows, uint32_t col, uint32_t row) {
        // Rotate the camera basis by yaw/pitch/roll around its own up/right/
        // front axes. The triad stays orthonormal for any angle combination
        // (no gimbal degeneracy at pitch = ±90°).
        const float yaw   = glm::radians(cfg.yaw);
        const float pitch = glm::radians(cfg.pitch);
        const float roll  = glm::radians(cfg.roll);
        const glm::vec3 right(_cam.right.x, _cam.right.y, _cam.right.z);
        const glm::vec3 up   (_cam.up.x,    _cam.up.y,    _cam.up.z);
        const glm::vec3 front(_cam.front.x, _cam.front.y, _cam.front.z);
        // Positive pitch looks up (rotation of front around right moves it
        // toward up), positive yaw turns left; matches CameraUi's orbit feel.
        glm::vec3 fFront = glm::rotate(front, pitch, right);
        glm::vec3 axisYaw = glm::normalize(glm::cross(right, fFront));
        fFront = glm::rotate(fFront, yaw, axisYaw);
        glm::vec3 fRight = glm::rotate(glm::rotate(right, pitch, right), yaw, axisYaw);
        glm::vec3 fUp    = glm::rotate(glm::rotate(up,    pitch, right), yaw, axisYaw);
        fRight = glm::rotate(fRight, roll, fFront);
        fUp    = glm::rotate(fUp,    roll, fFront);

        const float halfTan = std::tan(glm::radians(cfg.fov) * 0.5f);
        const float faceAspect = static_cast<float>(_rig.faceResX) /
                                 static_cast<float>(_rig.faceResY);

        RoomFaceData face;
        face.right = ::luisa::float3{fRight.x, fRight.y, fRight.z};
        face.up    = ::luisa::float3{fUp.x, fUp.y, fUp.z};
        face.front = ::luisa::float3{fFront.x, fFront.y, fFront.z};
        face.origin = cfg.origin.value_or(_cam.position);
        face.halfTan = halfTan;
        face.faceAspect = faceAspect;

        // Atlas rectangle in normalized UV; V = 1 at image top (NDC y = +1).
        const float u0 = static_cast<float>(col) / static_cast<float>(cols);
        const float u1 = static_cast<float>(col + 1u) / static_cast<float>(cols);
        const float v1 = 1.0f - static_cast<float>(row) / static_cast<float>(rows);
        const float v0 = 1.0f - static_cast<float>(row + 1u) / static_cast<float>(rows);
        face.atlas = ::luisa::float4{u0, v0, u1, v1};

        _cam.faces[faceIdx] = face;
    }

    const nlohmann::json Camera::toJson() {
        nlohmann::json file;
        file["eye"] = _ciCam.getEyePoint();
        file["orient"] = _ciCam.getOrientation();
        file["fov"] = _ciCam.getFov();
        file["near"] = _ciCam.getNearClip();
        file["far"] = _ciCam.getFarClip();
        file["projection"] = camera_projection_name(_projection);
        if (_projection == CameraProjection::Fisheye)
            file["fisheye_fov"] = _fisheyeFov;
        if (_projection == CameraProjection::RoomRig) {
            nlohmann::json rig;
            rig["layout"] = _rig.layout;
            if (_rig.layout == "grid") {
                rig["grid"] = { _rig.gridCols, _rig.gridRows };
            }
            rig["face_res"] = { _rig.faceResX, _rig.faceResY };
            nlohmann::json faces = nlohmann::json::array();
            for (const auto& f : _rig.faces) {
                nlohmann::json jf;
                jf["yaw"] = f.yaw;
                jf["pitch"] = f.pitch;
                jf["roll"] = f.roll;
                jf["fov"] = f.fov;
                if (f.origin)
                    jf["origin"] = { f.origin->x, f.origin->y, f.origin->z };
                faces.push_back(std::move(jf));
            }
            rig["faces"] = std::move(faces);
            file["room_rig"] = std::move(rig);
        }
        return file;
    }

    void Camera::load(const nlohmann::json& file) {
        _ciCam.setFov(file["fov"]);
        _ciCam.setEyePoint(file["eye"]);
        _ciCam.setOrientation(file["orient"]);
        _ciCam.setNearClip(file["near"]);
        _ciCam.setFarClip(file["far"]);

        const auto projName = file.value("projection", "perspective");
        const auto proj = camera_projection_from_string(projName).value_or(CameraProjection::Perspective);
        _fisheyeFov = file.value("fisheye_fov", 180.0f);
        _projection = proj;
        _cam.projection = static_cast<uint32_t>(proj);
        _cam.fisheye_fov = std::clamp(_fisheyeFov, 1.0f, 359.0f);
        _cam.face_count = 0u;
        if (proj == CameraProjection::RoomRig && file.contains("room_rig")) {
            const auto& jrig = file["room_rig"];
            RoomRigConfig rig;
            rig.layout = jrig.value("layout", "strip");
            const auto idxU32 = [](const nlohmann::json& arr, size_t i,
                                   uint32_t def) -> uint32_t {
                return arr.is_array() && arr.size() > i && arr[i].is_number()
                    ? arr[i].get<uint32_t>() : def;
            };
            if (jrig.contains("grid")) {
                rig.gridCols = idxU32(jrig["grid"], 0, 0u);
                rig.gridRows = idxU32(jrig["grid"], 1, 0u);
            }
            if (jrig.contains("face_res")) {
                rig.faceResX = idxU32(jrig["face_res"], 0, 1024u);
                rig.faceResY = idxU32(jrig["face_res"], 1, 1024u);
            }
            if (jrig.contains("faces")) {
                for (const auto& jf : jrig["faces"]) {
                    RoomRigFaceConfig f;
                    f.yaw = jf.value("yaw", 0.0f);
                    f.pitch = jf.value("pitch", 0.0f);
                    f.roll = jf.value("roll", 0.0f);
                    f.fov = jf.value("fov", 90.0f);
                    if (jf.contains("origin") && jf["origin"].is_array() &&
                        jf["origin"].size() >= 3) {
                        f.origin = ::luisa::float3{
                            jf["origin"][0].get<float>(),
                            jf["origin"][1].get<float>(),
                            jf["origin"][2].get<float>() };
                    }
                    rig.faces.push_back(f);
                    if (rig.faces.size() >= 6u) break;
                }
            }
            setRoomRig(rig); // fixes _cam.face_count
        }
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

    bool Camera::update() {
        bool poseChanged = false;
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
            _cam.far_clip = _ciCam.getFarClip();
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

            // Mirror projection state (rig faces follow the live camera basis)
            _cam.projection  = static_cast<uint32_t>(_projection);
            _cam.fisheye_fov = std::clamp(_fisheyeFov, 1.0f, 359.0f);
            if (_projection == CameraProjection::RoomRig && !_rig.faces.empty()) {
                const uint32_t n = static_cast<uint32_t>(_rig.faces.size());
                _cam.face_count = n;
                const uint32_t cols = (_rig.layout == "grid" && _rig.gridCols > 0u && _rig.gridRows > 0u)
                    ? _rig.gridCols : n;
                const uint32_t rows = (_rig.layout == "grid" && _rig.gridCols > 0u && _rig.gridRows > 0u)
                    ? _rig.gridRows : 1u;
                for (uint32_t i = 0u; i < n; ++i)
                    _buildRoomFace(_rig.faces[i], i, cols, rows, i % cols, i / cols);
            } else {
                _cam.face_count = 0u;
            }

            // Exact compare is sufficient: a still camera copies identical
            // floats every frame; any interaction changes them.
            const auto neq = [](const ::luisa::float3& a, const ::luisa::float3& b) noexcept {
                return a.x != b.x || a.y != b.y || a.z != b.z;
            };
            poseChanged = neq(_cam.position, _prev_cam.position) ||
                          neq(_cam.front,    _prev_cam.front)    ||
                          neq(_cam.right,    _prev_cam.right)    ||
                          neq(_cam.up,       _prev_cam.up);
        }

        // Snapshot prev jitter unconditionally: the _prev_cam snapshot above is
        // gated on camera motion, but jitter advances every frame.
        _cam.prev_jitter = _cam.jitter;

        // Jitter amplitude is one render pixel: with a decoupled render
        // resolution (panoramas) the window grid is the wrong reference.
        const float jitterW = static_cast<float>(
            _jitterResolution ? _jitterResolution->x : _window->getWidth());
        const float jitterH = static_cast<float>(
            _jitterResolution ? _jitterResolution->y : _window->getHeight());

        // Compute Halton(2,3) jitter for TAA (sub-pixel offset in NDC space)
        _cam.jitter = /*float2(0.f);*/::luisa::float2{
            (halton(_frameIndex, 2u) - 0.5f) * 2.0f / jitterW,
            (halton(_frameIndex, 3u) - 0.5f) * 2.0f / jitterH
        };
        _frameIndex++;
        return poseChanged;
    }
}
