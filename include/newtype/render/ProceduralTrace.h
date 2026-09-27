#pragma once

#include <newtype/core/Config.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#if NT_ENABLE_PROCEDURAL

#include <newtype/scene/ProceduralGeometry.h>
#include <newtype/render/ProcBindlessSlots.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Quaternion utilities (LuisaCompute has no built-in quaternion support)
//==============================================================================

/// Direct quaternion rotation — fewer registers, optimal for 1 vector.
/// q = (x,y,z,w) unit quaternion. v' = v + 2*w*(u×v) + 2*(u × (u×v))
[[nodiscard]] inline Float3 quat_rotate(Float4 q, Float3 v) noexcept {
    Float3 u = make_float3(q.x, q.y, q.z);
    Float3 t = 2.0f * cross(u, v);
    return v + q.w * t + cross(u, t);
}

/// Quaternion → 3x3 rotation matrix — better for rotating 2+ vectors.
[[nodiscard]] inline Float3x3 quat_to_mat(Float4 q) noexcept {
    Float x = q.x, y = q.y, z = q.z, w = q.w;
    Float xx = x * x, yy = y * y, zz = z * z;
    Float xy = x * y, xz = x * z, yz = y * z;
    Float wx = w * x, wy = w * y, wz = w * z;
    return make_float3x3(
        1.f - 2.f * (yy + zz), 2.f * (xy - wz),     2.f * (xz + wy),
        2.f * (xy + wz),       1.f - 2.f * (xx + zz), 2.f * (yz - wx),
        2.f * (xz - wy),       2.f * (yz + wx),       1.f - 2.f * (xx + yy));
}

/// Inverse rotation via conjugate (valid for unit quaternions).
[[nodiscard]] inline Float3 quat_inv_rotate(Float4 q, Float3 v) noexcept {
    return quat_rotate(make_float4(-q.x, -q.y, -q.z, q.w), v);
}

//==============================================================================
// 16-bit offset unpacking (CPU packs via pack_proc_offsets)
//==============================================================================

[[nodiscard]] inline UInt unpack_index_offset(UInt packed) noexcept {
    return packed >> 16u;
}

[[nodiscard]] inline UInt unpack_position_offset(UInt packed) noexcept {
    return packed & 0xFFFFu;
}

//==============================================================================
// ProcHitInfo — shared result from procedural intersection
//==============================================================================

/// Miss sentinel returned as .t by every intersector below. Also caps the
/// candidate acceptance interval: rays launched with a larger t_max (camera
/// rays use FLT_MAX) must not accept the sentinel as a hit.
inline constexpr float kProcMissT = 1e10f;

struct ProcHitInfo {
    Float t;
    UInt  prim;     // VAT: local triangle; Cube: face_id; Sphere: 0
    Float2 bary;    // VAT: barycentric; Cube: face UV; Sphere: zero
};

//==============================================================================
// Shared procedural intersection helpers
//==============================================================================

/// Intersect triangle mesh (VAT or deformable) with ray - single-frame active buffer.
/// After pre-interpolation, both VAT and deform use the same code path.
/// early_exit: if true, returns on first valid hit (for occlusion tests)
[[nodiscard]] inline ProcHitInfo intersect_triangles(
    Var<scene::ProcInstanceData> inst,
    Float3 ray_orig,
    Float3 ray_dir,
    const BindlessVar& proc_bindless,
    Bool early_exit = false) noexcept {

    Float local_t = def(1e10f);
    UInt local_tri = def(~0u);
    Float2 local_bary = def(make_float2(0.0f));

    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    UInt pos_base = unpack_position_offset(inst.packed_offsets);

    $for(t, inst.tri_count) {
        auto tri = proc_bindless.buffer<compute::Triangle>(kSlot_ProcIndices).read(idx_base + t);
        Float3 p0 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i0).xyz();
        Float3 p1 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i1).xyz();
        Float3 p2 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i2).xyz();

        Float3 e1 = p1 - p0;
        Float3 e2 = p2 - p0;
        Float3 h_vec = cross(ray_dir, e2);
        Float det = dot(e1, h_vec);
        $if(abs(det) > 1e-8f) {
            Float inv_det = 1.0f / det;
            Float3 s_vec = ray_orig - p0;
            Float u = dot(s_vec, h_vec) * inv_det;
            $if(u >= 0.0f) {
                $if(u <= 1.0f) {
                    Float3 q_vec = cross(s_vec, e1);
                    Float v = dot(ray_dir, q_vec) * inv_det;
                    $if(v >= 0.0f) {
                        $if(u + v <= 1.0f) {
                            Float t_hit = dot(e2, q_vec) * inv_det;
                            $if(t_hit > 1e-4f) {
                                $if(t_hit < local_t) {
                                    local_t = t_hit;
                                    local_tri = t;
                                    local_bary = make_float2(u, v);
                                    $if(early_exit) {
                                        $break;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
    };

    ProcHitInfo result;
    result.t = local_t;
    result.prim = local_tri;
    result.bary = local_bary;
    return result;
}

/// Intersect sphere with ray - returns hit info
[[nodiscard]] inline ProcHitInfo intersect_sphere(
    Float3 center,
    Float radius,
    Float3 ray_orig,
    Float3 ray_dir) noexcept {

    Float3 L = center - ray_orig;
    Float3 dir = ray_dir;
    Float cos_theta = dot(dir, normalize(L));

    Float dist = def(1e10f);
    $if(cos_theta > 0.0f) {
        Float d_oc = length(L);
        Float tc = d_oc * cos_theta;
        Float d = sqrt(max(d_oc * d_oc - tc * tc, 0.0f));
        $if(d <= radius) {
            Float t1c = sqrt(max(radius * radius - d * d, 0.0f));
            dist = tc - t1c;
        };
    };

    ProcHitInfo result;
    result.t = ite(dist > 1e-4f, dist, 1e10f);
    result.prim = 0u;
    result.bary = make_float2(0.0f);
    return result;
}

/// Intersect cube with ray - returns hit info
[[nodiscard]] inline ProcHitInfo intersect_cube(
    Float3 center,
    Float half_ext,
    Float4 rotation,
    Float3 ray_orig,
    Float3 ray_dir) noexcept {

    // Transform ray into cube's local space
    // For unit quaternions: q^-1 = conjugate(q) = (-x, -y, -z, w)
    Float3x3 inv_rot = quat_to_mat(make_float4(
        -rotation.x, -rotation.y, -rotation.z, rotation.w
    ));
    Float3 local_orig = inv_rot * (ray_orig - center);
    Float3 local_dir = inv_rot * ray_dir;

    // Slab method on [-h, h]^3
    Float3 box_min = make_float3(-half_ext);
    Float3 box_max = make_float3(half_ext);

    Float t_min = def(-1e10f);
    Float t_max = def(1e10f);
    UInt face_min = def(0u);

    // X slab
    $if(abs(local_dir.x) > 1e-8f) {
        Float inv_d = 1.0f / local_dir.x;
        Float t1 = (box_min.x - local_orig.x) * inv_d;
        Float t2 = (box_max.x - local_orig.x) * inv_d;
        UInt f1 = 0u; // -X face
        UInt f2 = 1u; // +X face
        $if(t1 > t2) {
            Float tmp = t1; t1 = t2; t2 = tmp;
            UInt ft = f1; f1 = f2; f2 = ft;
        };
        $if(t1 > t_min) { t_min = t1; face_min = f1; };
        t_max = min(t_max, t2);
    }
    $else {
        $if(local_orig.x < box_min.x | local_orig.x > box_max.x) {
            t_max = -1e10f;
        };
    };

    // Y slab
    $if(abs(local_dir.y) > 1e-8f) {
        Float inv_d = 1.0f / local_dir.y;
        Float t1 = (box_min.y - local_orig.y) * inv_d;
        Float t2 = (box_max.y - local_orig.y) * inv_d;
        UInt f1 = 2u; // -Y
        UInt f2 = 3u; // +Y
        $if(t1 > t2) {
            Float tmp = t1; t1 = t2; t2 = tmp;
            UInt ft = f1; f1 = f2; f2 = ft;
        };
        $if(t1 > t_min) { t_min = t1; face_min = f1; };
        t_max = min(t_max, t2);
    }
    $else {
        $if(local_orig.y < box_min.y | local_orig.y > box_max.y) {
            t_max = -1e10f;
        };
    };

    // Z slab
    $if(abs(local_dir.z) > 1e-8f) {
        Float inv_d = 1.0f / local_dir.z;
        Float t1 = (box_min.z - local_orig.z) * inv_d;
        Float t2 = (box_max.z - local_orig.z) * inv_d;
        UInt f1 = 4u; // -Z
        UInt f2 = 5u; // +Z
        $if(t1 > t2) {
            Float tmp = t1; t1 = t2; t2 = tmp;
            UInt ft = f1; f1 = f2; f2 = ft;
        };
        $if(t1 > t_min) { t_min = t1; face_min = f1; };
        t_max = min(t_max, t2);
    }
    $else {
        $if(local_orig.z < box_min.z | local_orig.z > box_max.z) {
            t_max = -1e10f;
        };
    };

    ProcHitInfo result;
    // Branch ordering: check t_min > epsilon first (cheap, filters ray-origin hits)
    result.t = ite(t_min > 1e-4f & t_max >= t_min, t_min, 1e10f);
    result.prim = face_min;
    result.bary = make_float2(0.0f);
    return result;
}

//==============================================================================
// trace_closest: full traversal with procedural intersection
//==============================================================================

/// Result of a closest-hit trace.
struct TraceResult {
    Float committed_ray_t;
    UInt  inst;
    UInt  prim;
    Float2 bary;
    UInt  hit_type;        // 0=miss, 1=surface, 2=procedural
    Bool  is_procedural;
    UInt  local_tri;       // VAT: triangle index; Cube: face_id; Sphere: 0
    Float2 local_bary;     // VAT: barycentric; Cube: face UV; Sphere: zero

    [[nodiscard]] auto operator->() const noexcept { return this; }

    [[nodiscard]] auto miss() const noexcept {
        return hit_type == 0u;
    }
};

[[nodiscard]] inline TraceResult trace_closest(
    const AccelVar& accel,
    Var<compute::Ray> ray,
    const BindlessVar& proc_bindless) noexcept {

    Float best_t = def(1e10f);
    UInt best_tri = def(~0u);
    Float2 best_bary = def(make_float2(0.0f));
    UInt best_prim = def(~0u); // track winning AABB index manually

    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) noexcept {
            c.commit();
        })
        .on_procedural_candidate([&](ProceduralCandidate& candidate) noexcept {
            Var<ProceduralHit> h = candidate.hit();
            Var<scene::ProcInstanceData> inst = proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(h.prim);

            auto c_ray = candidate.ray();
            Float3 ray_orig = c_ray->origin();
            Float3 ray_dir = c_ray->direction();
            // Candidate-ray acceptance interval. t_max() is the query ray's
            // upper bound, so a commit survives iff t < t_hi — the same bound
            // _CommitProcedural enforces; t_lo additionally rejects hits
            // behind RayTMin. The cap at kProcMissT is NOT optional: every
            // intersector returns t == 1e10f as its MISS sentinel, and camera
            // rays are created with t_max = FLT_MAX (make_ray 2-arg default),
            // so a miss would pass `t < t_hi` and commit a phantom hit at
            // 1e10 wherever the ray crosses an AABB without hitting a
            // triangle (the env-background bbox artifact).
            // The manual best_* bookkeeping below must agree with this
            // interval, otherwise it records hits the hardware rejected
            // (e.g. a procedural surface just beyond a shadow ray's t_max).
            Float t_lo = c_ray->t_min();
            Float t_hi = min(c_ray->t_max(), kProcMissT);

            $if((inst.type & 0xFu) == 0u) { // VAT mesh — pre-interpolated, single-frame
                ProcHitInfo hit_info = intersect_triangles(inst, ray_orig, ray_dir, proc_bindless);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    $if(hit_info.t < best_t) {
                        best_t = hit_info.t;
                        best_tri = hit_info.prim;
                        best_bary = hit_info.bary;
                        best_prim = h.prim;
                    };
                };
            }
            $elif((inst.type & 0xFu) == 1u) { // Sphere
                auto aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(h.prim);
                Float3 center = (aabb->min() + aabb->max()) * 0.5f;
                Float3 extent = aabb->max() - aabb->min();
                Float radius = extent.x * 0.5f;

                ProcHitInfo hit_info = intersect_sphere(center, radius, ray_orig, ray_dir);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    $if(hit_info.t < best_t) {
                        best_t = hit_info.t;
                        best_tri = 0u;
                        best_bary = make_float2(0.0f);
                        best_prim = h.prim;
                    };
                };
            }
            $elif((inst.type & 0xFu) == 2u) { // Cube
                auto aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(h.prim);
                Float3 center = (aabb->min() + aabb->max()) * 0.5f;
                Float half_ext = inst.param;

                ProcHitInfo hit_info = intersect_cube(center, half_ext, inst.rotation, ray_orig, ray_dir);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    $if(hit_info.t < best_t) {
                        best_t = hit_info.t;
                        best_tri = hit_info.prim;
                        best_bary = make_float2(0.0f);
                        best_prim = h.prim;
                    };
                };
            }
            $else { // Deformable static (type == 3u) — same code path as VAT
                ProcHitInfo hit_info = intersect_triangles(inst, ray_orig, ray_dir, proc_bindless);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    $if(hit_info.t < best_t) {
                        best_t = hit_info.t;
                        best_tri = hit_info.prim;
                        best_bary = hit_info.bary;
                        best_prim = h.prim;
                    };
                };
            };
        })
        .trace();

    TraceResult r;
    r.committed_ray_t = hit.committed_ray_t;
    r.inst = hit.inst;
    // Always use manually-tracked best_prim for procedural hits.
    // CommittedHit.prim may return TLAS instance index instead of AABB index.
    r.prim = ite(hit.hit_type == 2u & best_prim != ~0u, best_prim, hit.prim);
    r.bary = hit.bary;
    r.hit_type = hit.hit_type;
    r.is_procedural = (hit.hit_type == 2u);
    r.local_tri = best_tri;
    r.local_bary = best_bary;
    return r;
}

//==============================================================================
// trace_occluded: shadow/occlusion trace — returns true if ray hits anything
//==============================================================================

/// Occlusion is derived from the query's CommittedHit, never from a flag set
/// inside the candidate handlers:
///  - Engine meshes are built opaque (D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE /
///    VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR), so opaque triangles
///    auto-commit WITHOUT invoking on_surface_candidate — the generated loop
///    only reports CANDIDATE_NON_OPAQUE_TRIANGLE there. A handler-side flag
///    would miss mesh blockers entirely and shadow rays would see through all
///    mesh geometry.
///  - Procedural analytic hits must land inside the candidate-ray interval
///    [RayTMin, CommittedRayT) — the same acceptance the hardware enforces on
///    CommitProceduralPrimitiveHit. The analytic intersectors themselves are
///    unbounded (they test the infinite ray), and visibility rays deliberately
///    stop `dist - offset` short of their target: without the t_hi bound, a
///    procedural target surface sitting just beyond t_max would falsely
///    occlude its own visibility ray.
[[nodiscard]] inline Bool trace_occluded(
    const AccelVar& accel,
    Var<compute::Ray> ray,
    const BindlessVar& proc_bindless) noexcept {

    auto hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) noexcept {
            c.commit();
            c.terminate();
        })
        .on_procedural_candidate([&](ProceduralCandidate& candidate) noexcept {
            Var<ProceduralHit> h = candidate.hit();
            Var<scene::ProcInstanceData> inst = proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(h.prim);

            auto c_ray = candidate.ray();
            Float3 ray_orig = c_ray->origin();
            Float3 ray_dir = c_ray->direction();
            // Same acceptance contract as trace_closest: [t_min, t_max) capped
            // at the intersectors' 1e10f miss sentinel — without the cap a
            // miss (t == 1e10f) passes on rays with t_max > 1e10 (camera rays
            // use FLT_MAX) and falsely reports occlusion.
            Float t_lo = c_ray->t_min();
            Float t_hi = min(c_ray->t_max(), kProcMissT);

            $if((inst.type & 0xFu) == 0u) { // VAT mesh
                ProcHitInfo hit_info = intersect_triangles(inst, ray_orig, ray_dir, proc_bindless, true);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    candidate.terminate();
                };
            }
            $elif((inst.type & 0xFu) == 1u) { // Sphere
                auto aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(h.prim);
                Float3 center = (aabb->min() + aabb->max()) * 0.5f;
                Float3 extent = aabb->max() - aabb->min();
                Float radius = extent.x * 0.5f;

                ProcHitInfo hit_info = intersect_sphere(center, radius, ray_orig, ray_dir);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    candidate.terminate();
                };
            }
            $elif((inst.type & 0xFu) == 2u) { // Cube
                auto aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(h.prim);
                Float3 center = (aabb->min() + aabb->max()) * 0.5f;
                Float half_ext = inst.param;

                ProcHitInfo hit_info = intersect_cube(center, half_ext, inst.rotation, ray_orig, ray_dir);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    candidate.terminate();
                };
            }
            $else { // Deformable static (type == 3u)
                ProcHitInfo hit_info = intersect_triangles(inst, ray_orig, ray_dir, proc_bindless, true);
                $if(hit_info.t >= t_lo & hit_info.t < t_hi) {
                    candidate.commit(hit_info.t);
                    candidate.terminate();
                };
            };
        })
        .trace();

    return !hit->miss();
}

//==============================================================================
// Builtin normal reconstruction — called from Shading.h / shade passes
//==============================================================================

/// Compute world-space normal for a sphere hit.
/// hit_pos and center are in world space; rotation is applied to the local normal.
[[nodiscard]] inline Float3 sphere_normal(
    Float3 hit_pos, Float3 center, Float4 rotation) noexcept {
    Float3 local_n = normalize(hit_pos - center);
    return quat_rotate(rotation, local_n);
}

/// Compute world-space normal for a cube hit from face_id.
/// face_id: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
[[nodiscard]] inline Float3 cube_normal(UInt face_id, Float4 rotation) noexcept {
    Float3 local_n = make_float3(0.0f);
    // Map face_id to axis
    $if(face_id == 0u) { local_n = def(luisa::make_float3(-1.0f, 0.0f, 0.0f)); }
    $elif(face_id == 1u) { local_n = def(luisa::make_float3(1.0f, 0.0f, 0.0f)); }
    $elif(face_id == 2u) { local_n = def(luisa::make_float3(0.0f, -1.0f, 0.0f)); }
    $elif(face_id == 3u) { local_n = def(luisa::make_float3(0.0f, 1.0f, 0.0f)); }
    $elif(face_id == 4u) { local_n = def(luisa::make_float3(0.0f, 0.0f, -1.0f)); }
    $else { local_n = def(luisa::make_float3(0.0f, 0.0f, 1.0f)); };
    return quat_rotate(rotation, local_n);
}

} // namespace newtype::render

#else // !NT_ENABLE_PROCEDURAL

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] inline auto trace_closest(
    const AccelVar& accel, Var<Ray> ray) noexcept {
    return accel->intersect(ray, {});
}

[[nodiscard]] inline Bool trace_occluded(
    const AccelVar& accel, Var<Ray> ray) noexcept {
    return accel->intersect_any(ray, {});
}

} // namespace newtype::render

#endif // NT_ENABLE_PROCEDURAL
