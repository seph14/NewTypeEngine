#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <cinder/Camera.h>
#include "cinder/CameraUi.h"
#include "cinder/Json.h"
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Camera structure for LuisaCompute DSL path tracing
 *
 * Defined at global scope for LUISA_STRUCT macro compatibility.
 * The macro generates DSL code that must be at global/nested namespace scope.
 *
 * Mirrors the Camera struct from LuisaCompute's test_path_tracing_camera.cpp.
 */

namespace newtype::util {

// Camera projection model for generate_ray. Values are GPU-facing (uint32 in
// CameraData) — shaders switch on them; 0 = the classic pinhole path that must
// stay byte-compatible with the pre-projection codegen.
enum class CameraProjection : uint32_t {
    Perspective = 0u,   // pinhole (ci::CameraPersp matrices remain authoritative)
    Equirect    = 1u,   // 360° panorama, 2:1 lat/long, seam behind the camera
    Cylindrical = 2u,   // 360° horizontal strip; vertical span via tan(fov/2) like perspective
    Fisheye     = 3u,   // equidistant fisheye, full-circle FOV = fisheye_fov degrees
    RoomRig     = 4u,   // atlas of perspective faces (4 walls + optional floor/ceiling, CAVE-style)
};

[[nodiscard]] inline constexpr const char* camera_projection_name(CameraProjection p) noexcept {
    switch (p) {
        case CameraProjection::Perspective: return "perspective";
        case CameraProjection::Equirect:    return "equirect";
        case CameraProjection::Cylindrical: return "cylindrical";
        case CameraProjection::Fisheye:     return "fisheye";
        case CameraProjection::RoomRig:     return "roomrig";
    }
    return "perspective";
}

[[nodiscard]] inline std::optional<CameraProjection> camera_projection_from_string(std::string_view s) noexcept {
    if (s == "perspective") return CameraProjection::Perspective;
    if (s == "equirect")    return CameraProjection::Equirect;
    if (s == "cylindrical") return CameraProjection::Cylindrical;
    if (s == "fisheye")     return CameraProjection::Fisheye;
    if (s == "roomrig" || s == "room_rig" || s == "room") return CameraProjection::RoomRig;
    return std::nullopt;
}

// One perspective face of a room rig. The GPU mapping uses the raw orthonormal
// basis plus halfTan/faceAspect (dir = front + right*(fx*aspect*halfTan) +
// up*(fy*halfTan)); keeping the raw basis makes the projection invertible for
// project()/project_prev (point → face-local NDC). atlas is the face's
// rectangle in normalized atlas UV: (minU, minV, maxU, maxV), V = 1 at top.
struct RoomFaceData {
    ::luisa::float3 right;
    ::luisa::float3 up;
    ::luisa::float3 front;
    ::luisa::float3 origin;
    ::luisa::float4 atlas;
    float           halfTan;
    float           faceAspect;
};

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
    // --- Non-perspective projection state (docs/non_perspective_camera_report.md) ---
    uint32_t        projection = 0u;   // CameraProjection (0 = perspective)
    uint32_t        face_count = 0u;   // active room-rig faces (max 6)
    float           fisheye_fov = 180.0f; // fisheye full-circle FOV in degrees
    float           projection_pad = 0.0f;
    RoomFaceData    faces[6];          // room-rig face table (first face_count entries)
};

} // namespace newtype::util

// Register RoomFaceData with LuisaCompute DSL (must be outside namespace,
// before CameraData which embeds it by value).
LUISA_STRUCT(newtype::util::RoomFaceData, right, up, front, origin, atlas, halfTan, faceAspect) {};

// Register CameraData with LuisaCompute DSL (must be outside namespace)
LUISA_STRUCT(newtype::util::CameraData, position, prev_position, front, up, right, prev_front, prev_up, prev_right, fov, aspect, near_clip, far_clip, view_proj, prev_view_proj, jitter, prev_jitter, projection, face_count, fisheye_fov, projection_pad, faces) {
    // Generate ray from camera through normalized pixel coordinate [-1, 1]
    // Applies jitter offset for TAA anti-aliasing
    //
    // The projection dispatch is a warp-coherent uniform branch: `projection`
    // is a kernel-argument struct member, identical for every thread, and the
    // perspective body (enum 0) is the original formula verbatim so the
    // generated instruction sequence for Perspective is unchanged. The
    // non-perspective bodies only ever execute for non-perspective cameras.
    [[nodiscard]] auto generate_ray(::luisa::compute::Expr<::luisa::float2> p) const noexcept {
        // Function-local: this struct is also compiled in runtime-shader DLLs
        // that carry no using-directive for luisa::compute (the historical
        // body resolved everything through ADL alone).
        using namespace ::luisa::compute;
        auto pj = p + jitter;  // apply sub-pixel jitter
        Float3 wi_world = def(make_float3(0.0f, 0.0f, 1.0f));
        Float3 ray_origin = def(position);
        $if (projection == 0u) {
            // Perspective (pinhole) — original formula, kept verbatim
            auto fov_radians = radians(fov);
            auto half_tan = tan(0.5f * fov_radians);
            auto wi_local = make_float3(pj.x * aspect * half_tan, pj.y * half_tan, -1.0f);
            wi_world = normalize(wi_local.x * right + wi_local.y * up - wi_local.z * front);
        } $else {
            $switch (projection) {
                // Equirectangular lat/long, standard convention (exact inverse
                // of project's equirect case): NDC.x = lon/pi — image center
                // looks along `front`, +x turns toward `right`, the lon seam
                // sits at the left/right image edges (behind the camera);
                // NDC.y = lat/(pi/2) spans 180° of latitude.
                $case (1u) {
                    Float lat = pj.y * (3.14159265359f * 0.5f);
                    Float lon = pj.x * 3.14159265359f;
                    Float cl = cos(lat);
                    Float3 local = make_float3(cl * sin(lon), sin(lat), cl * cos(lon));
                    wi_world = normalize(local.x * right + local.y * up + local.z * front);
                };
                // Cylindrical panorama: NDC.x = lon/pi (center = front, +x
                // toward right, 360° seam at the image edges); v maps linearly
                // to tan(elevation), matching the perspective vertical extent
                // for a given fov.
                $case (2u) {
                    Float lon = pj.x * 3.14159265359f;
                    Float tan_v = pj.y * tan(0.5f * radians(fov));
                    wi_world = normalize(front * cos(lon) + right * sin(lon) + up * tan_v);
                };
                // Equidistant fisheye: pixel radius maps linearly to the ray
                // angle from `front`; the full circle spans fisheye_fov.
                $case (3u) {
                    Float r = min(length(pj), 1.0f);
                    Float phi = atan2(pj.y, pj.x);
                    Float ang = r * radians(fisheye_fov) * 0.5f;
                    Float sa = sin(ang);
                    wi_world = normalize(right * (sa * cos(phi)) + up * (sa * sin(phi)) + front * cos(ang));
                };
                // Room rig atlas: locate the atlas rectangle containing the
                // pixel, then use that face's perspective formula. Pixels not
                // covered by any face (layout gaps) keep the fallback ray.
                $case (4u) {
                    Float2 uv = (pj + 1.0f) * 0.5f;
                    $for (fi, 6u) {
                        $if (fi < face_count) {
                            Float4 rect = faces[fi]->atlas;
                            $if (uv.x >= rect.x & uv.y >= rect.y &
                                 uv.x <  rect.z & uv.y <  rect.w) {
                                Float2 fndc = (uv - rect.xy()) / (rect.zw() - rect.xy()) * 2.0f - 1.0f;
                                wi_world = normalize(faces[fi]->front +
                                                     faces[fi]->right * (fndc.x * faces[fi]->faceAspect * faces[fi]->halfTan) +
                                                     faces[fi]->up * (fndc.y * faces[fi]->halfTan));
                                ray_origin = faces[fi]->origin;
                            };
                        };
                    };
                };
            };
        };
        return ::luisa::compute::make_ray(ray_origin, wi_world);
    }

    // Project world position to NDC [-1, 1]. Perspective uses the view-proj
    // matrix verbatim; the point-origin projections (equirect / cylindrical /
    // fisheye / room rig) map the camera-relative direction analytically —
    // the exact inverse of generate_ray, so motion vectors
    // (project_prev − project) stay consistent with the ray mapping.
    [[nodiscard]] auto project(::luisa::compute::Expr<::luisa::float3> world_pos) const noexcept {
        using namespace ::luisa::compute;
        Float2 ndc = def(make_float2(0.0f, 0.0f));
        $if (projection == 0u) {
            auto clip = view_proj * make_float4(world_pos, 1.0f);
            ndc = clip.xy() / clip.w;
        } $else {
            Float3 rel = world_pos - position;
            Float3 dir = normalize(rel);
            Float x = dot(dir, right);
            Float y = dot(dir, up);
            Float z = dot(dir, front);
            $switch (projection) {
                // Inverse of the lat/long mapping: lon = atan2(x, z)
                // (local xz = cl*sin(lon), cl*cos(lon)), lat = asin(y)
                $case (1u) {
                    ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f),
                                      asin(clamp(y, -1.0f, 1.0f)) * (2.0f / 3.14159265359f));
                };
                // lon = atan2(x, z); v linear in tan(elevation)/tan(fov/2)
                $case (2u) {
                    Float horiz = sqrt(max(1.0f - y * y, 1e-9f));
                    Float tanV = y / horiz / max(tan(0.5f * radians(fov)), 1e-4f);
                    ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f), tanV);
                };
                // r = angle from front / (fov/2); phi = atan2(y, x)
                $case (3u) {
                    Float ang = acos(clamp(z, -1.0f, 1.0f));
                    Float r = ang / max(radians(fisheye_fov) * 0.5f, 1e-4f);
                    Float phi = atan2(y, x);
                    ndc = make_float2(r * cos(phi), r * sin(phi));
                };
                // Face-local perspective from the face's OWN origin —
                // generate_ray fires face rays from faces[fi]->origin (which
                // may override the rig origin), so the inverse must use it
                // too (mirrors project_prev's room-rig case).
                $case (4u) {
                    $for (fi, 6u) {
                        $if (fi < face_count) {
                            Float3 relF = normalize(world_pos - faces[fi]->origin);
                            Float fz = dot(relF, faces[fi]->front);
                            $if (fz > 1e-4f) {
                                Float fx = dot(relF, faces[fi]->right) / fz;
                                Float fy = dot(relF, faces[fi]->up) / fz;
                                Float2 fndc = make_float2(
                                    fx / (faces[fi]->faceAspect * faces[fi]->halfTan),
                                    fy / faces[fi]->halfTan);
                                Float4 rect = faces[fi]->atlas;
                                Float2 uv = (fndc + 1.0f) * 0.5f;
                                // first face in front of the point wins
                                $if (all(uv >= rect.xy()) & all(uv < rect.zw())) {
                                    ndc = uv * 2.0f - 1.0f;
                                };
                            };
                        };
                    };
                };
            };
        };
        return ndc;
    }

    // Project world position to NDC [-1, 1] using previous frame camera
    [[nodiscard]] auto project_prev(::luisa::compute::Expr<::luisa::float3> world_pos) const noexcept {
        using namespace ::luisa::compute;
        Float2 ndc = def(make_float2(0.0f, 0.0f));
        $if (projection == 0u) {
            auto clip = prev_view_proj * make_float4(world_pos, 1.0f);
            ndc = clip.xy() / clip.w;
        } $else {
            Float3 rel = world_pos - prev_position;
            Float3 dir = normalize(rel);
            Float x = dot(dir, prev_right);
            Float y = dot(dir, prev_up);
            Float z = dot(dir, prev_front);
            $switch (projection) {
                $case (1u) {
                    ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f),
                                      asin(clamp(y, -1.0f, 1.0f)) * (2.0f / 3.14159265359f));
                };
                $case (2u) {
                    Float horiz = sqrt(max(1.0f - y * y, 1e-9f));
                    Float tanV = y / horiz / max(tan(0.5f * radians(fov)), 1e-4f);
                    ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f), tanV);
                };
                $case (3u) {
                    Float ang = acos(clamp(z, -1.0f, 1.0f));
                    Float r = ang / max(radians(fisheye_fov) * 0.5f, 1e-4f);
                    Float phi = atan2(y, x);
                    ndc = make_float2(r * cos(phi), r * sin(phi));
                };
                // Room rig: faces are rigid with the camera (the rig follows
                // the pose), so the CURRENT face table applies to the prev
                // pose too — only the origin differs. KNOWN LIMITATION: under
                // pose rotation the previous frame's face table differed, so
                // these MVs understate rig rotation; the room rig is
                // offline-only this phase (accumulation resets on motion) and
                // nothing consumes those MVs today.
                $case (4u) {
                    $for (fi, 6u) {
                        $if (fi < face_count) {
                            Float3 org = faces[fi]->origin;
                            Float3 relF = world_pos - org;
                            Float3 dirF = normalize(relF);
                            Float fz = dot(dirF, faces[fi]->front);
                            $if (fz > 1e-4f) {
                                Float fx = dot(dirF, faces[fi]->right) / fz;
                                Float fy = dot(dirF, faces[fi]->up) / fz;
                                Float2 fndc = make_float2(
                                    fx / (faces[fi]->faceAspect * faces[fi]->halfTan),
                                    fy / faces[fi]->halfTan);
                                Float4 rect = faces[fi]->atlas;
                                Float2 uv = (fndc + 1.0f) * 0.5f;
                                $if (all(uv >= rect.xy()) & all(uv < rect.zw())) {
                                    ndc = uv * 2.0f - 1.0f;
                                };
                            };
                        };
                    };
                };
            };
        };
        return ndc;
    }
};

namespace newtype::util {

//==============================================================================
// Shared point-origin projection helpers (docs/non_perspective_camera_report.md
// §5.4 — analytic direction evaluation). These mirror the non-perspective
// branch of CameraData::generate_ray / project and are called by the denoiser
// reconstruction sites with the RAW camera basis hoisted from RelaxConstants.
// Perspective (projection == 0) callers never reach them: every call site
// keeps the original frustum formula inside the projection == 0 branch.
//==============================================================================

// NDC [-1,1] -> unit primary-ray direction for equirect / cylindrical /
// fisheye (the room rig stays on per-face frustum constants and is not
// denoised in this phase).
[[nodiscard]] inline ::luisa::compute::Float3 eval_point_origin_direction(
    ::luisa::compute::Float2 pj,
    ::luisa::compute::Float3 right, ::luisa::compute::Float3 up,
    ::luisa::compute::Float3 front,
    ::luisa::compute::UInt projection,
    ::luisa::compute::Float fisheyeHalfFovRad,
    ::luisa::compute::Float cylHalfTan) noexcept {
    using namespace ::luisa::compute;
    // Convention: mirrors generate_ray's equirect/cylindrical branches
    // exactly (NDC.x = lon/pi, center = front, seam at the edges) — the
    // denoiser's absolute uvReprojPrev must live in the same space as the
    // motion vectors CameraData::project produces.
    Float3 dir = def(front);
    $switch (projection) {
        // Equirect lat/long (image center = front, seam at the edges)
        $case (1u) {
            Float lat = pj.y * (3.14159265359f * 0.5f);
            Float lon = pj.x * 3.14159265359f;
            Float cl = cos(lat);
            Float3 local = make_float3(cl * sin(lon), sin(lat), cl * cos(lon));
            dir = normalize(local.x * right + local.y * up + local.z * front);
        };
        // Cylindrical: 360° azimuth; v linear in tan(elevation)
        $case (2u) {
            Float lon = pj.x * 3.14159265359f;
            dir = normalize(front * cos(lon) + right * sin(lon) + up * (pj.y * cylHalfTan));
        };
        // Equidistant fisheye
        $case (3u) {
            Float r = min(length(pj), 1.0f);
            Float phi = atan2(pj.y, pj.x);
            Float ang = r * fisheyeHalfFovRad;
            Float sa = sin(ang);
            dir = normalize(right * (sa * cos(phi)) + up * (sa * sin(phi)) + front * cos(ang));
        };
    };
    return dir;
}

// Camera-relative unit direction -> NDC [-1,1] (the inverse of the above),
// used to reproject world points under point-origin cameras.
[[nodiscard]] inline ::luisa::compute::Float2 project_point_origin_direction(
    ::luisa::compute::Float3 dir,
    ::luisa::compute::Float3 right, ::luisa::compute::Float3 up,
    ::luisa::compute::Float3 front,
    ::luisa::compute::UInt projection,
    ::luisa::compute::Float fisheyeHalfFovRad,
    ::luisa::compute::Float cylHalfTan) noexcept {
    using namespace ::luisa::compute;
    Float x = dot(dir, right);
    Float y = dot(dir, up);
    Float z = dot(dir, front);
    Float2 ndc = def(make_float2(0.0f, 0.0f));
    $switch (projection) {
        // lon = atan2(x, z), lat = asin(y) — inverse of the lat/long case above
        $case (1u) {
            ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f),
                              asin(clamp(y, -1.0f, 1.0f)) * (2.0f / 3.14159265359f));
        };
        $case (2u) {
            Float horiz = sqrt(max(1.0f - y * y, 1e-9f));
            ndc = make_float2(atan2(x, z) * (1.0f / 3.14159265359f),
                              (y / horiz) / max(cylHalfTan, 1e-4f));
        };
        $case (3u) {
            Float ang = acos(clamp(z, -1.0f, 1.0f));
            Float r = ang / max(fisheyeHalfFovRad, 1e-4f);
            Float phi = atan2(y, x);
            ndc = make_float2(r * cos(phi), r * sin(phi));
        };
    };
    return ndc;
}

//==============================================================================
// CPU-side camera class
//==============================================================================

class Camera;
typedef std::unique_ptr<Camera> CamPtr;

// CPU-side description of one room-rig face (angles in degrees, relative to
// the camera basis: yaw rotates around up, pitch around right (positive =
// up), roll around front). `origin` defaults to the shared rig origin (the
// camera eye) when unset.
struct RoomRigFaceConfig {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float fov = 90.0f;
    std::optional<::luisa::float3> origin;
};

// Room-rig layout: faces are packed into an atlas either as a horizontal
// strip (1 row) or a grid (cols x rows), all faces at faceRes. Presets are
// just prefilled face lists (docs/non_perspective_camera_report.md §5.2).
struct RoomRigConfig {
    std::vector<RoomRigFaceConfig> faces;
    std::string layout = "strip";   // "strip" | "grid"
    uint32_t gridCols = 0u;         // used when layout == "grid"
    uint32_t gridRows = 0u;
    uint32_t faceResX = 1024u;
    uint32_t faceResY = 1024u;

    // Atlas pixel size for the current face list + layout
    [[nodiscard]] ::luisa::uint2 atlasSize() const noexcept {
        if (layout == "grid" && gridCols > 0u && gridRows > 0u)
            return { gridCols * faceResX, gridRows * faceResY };
        const auto n = static_cast<uint32_t>(std::max<size_t>(faces.size(), 1u));
        return { n * faceResX, faceResY };
    }
};

// Convenience presets (arbitrary JSON face lists remain fully supported)
[[nodiscard]] inline RoomRigConfig roomRigPreset(std::string_view name) {
    RoomRigConfig rig;
    if (name == "4wall_floor" || name == "4wall+floor") {
        rig.layout = "strip";
        rig.faces  = { {0.f, 0.f, 0.f}, {90.f, 0.f, 0.f}, {180.f, 0.f, 0.f},
                       {270.f, 0.f, 0.f}, {0.f, -90.f, 0.f} };
        for (auto& f : rig.faces) f.fov = 90.0f;
    } else if (name == "cube") {
        rig.layout = "grid";
        rig.gridCols = 3u;
        rig.gridRows = 2u;
        rig.faces  = { {0.f, 0.f, 0.f}, {90.f, 0.f, 0.f}, {180.f, 0.f, 0.f},
                       {270.f, 0.f, 0.f}, {0.f, 90.f, 0.f}, {0.f, -90.f, 0.f} };
        for (auto& f : rig.faces) f.fov = 90.0f;
    } else { // "4wall" (default)
        rig.layout = "strip";
        rig.faces  = { {0.f, 0.f, 0.f}, {90.f, 0.f, 0.f}, {180.f, 0.f, 0.f},
                       {270.f, 0.f, 0.f} };
        for (auto& f : rig.faces) f.fov = 90.0f;
    }
    return rig;
}

/**
 * @brief Helper class to convert ci::Camera to LuisaCamera
 *
 * Converts between ci's camera representation and LuisaCompute's
 * path tracing camera structure. The ci::CameraPersp remains the pose
 * authority for every projection; non-perspective projections reinterpret
 * the pixel->ray mapping in CameraData::generate_ray only.
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

    // Non-perspective projection config (mirrored into _cam every update())
    CameraProjection _projection = CameraProjection::Perspective;
    float            _fisheyeFov = 180.0f;
    RoomRigConfig    _rig;

    // Jitter amplitude resolution: defaults to the window, but decoupled
    // render resolutions (panoramas) must jitter at the render pixel grid.
    std::optional<::luisa::uint2> _jitterResolution;

    Camera(const ci::app::WindowRef& window);
public:
    static CamPtr create(const ci::app::WindowRef& window) { return CamPtr(new Camera(window)); }

    [[nodiscard]] ci::CameraPersp& ciCam() noexcept { return _ciCam; }
    [[nodiscard]] ci::CameraUi&    camUi() noexcept { return _camUi; }
    [[nodiscard]] CameraData&      cam()   noexcept { return _cam; }

    [[nodiscard]] CameraProjection projection() const noexcept { return _projection; }
    [[nodiscard]] const RoomRigConfig& roomRig() const noexcept { return _rig; }
    [[nodiscard]] float fisheyeFov() const noexcept { return _fisheyeFov; }

    void setProjection(CameraProjection p) noexcept;
    void setFisheyeFov(float fovDeg) noexcept;
    // Rebuilds the GPU face table from the config (call after editing rig).
    void setRoomRig(const RoomRigConfig& rig);
    void setJitterResolution(std::optional<::luisa::uint2> res) noexcept { _jitterResolution = res; }

    void mouseDown (ci::app::MouseEvent& event);
    void mouseDrag (ci::app::MouseEvent& event);
    void mouseWheel(ci::app::MouseEvent& event);

    [[nodiscard]] const nlohmann::json toJson();
    void load(const nlohmann::json& file);

    // Returns whether the pose changed this frame (exact float compare of
    // position/basis against the previous snapshot — consumers use it to
    // reset progressive accumulation).
    [[nodiscard]] bool update();

private:
    // Rotates the camera basis by yaw/pitch/roll (degrees) around the
    // camera's up/right/front axes and fills the GPU face entry.
    void _buildRoomFace(const RoomRigFaceConfig& cfg, uint32_t faceIdx,
                        uint32_t cols, uint32_t rows, uint32_t col, uint32_t row);

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
