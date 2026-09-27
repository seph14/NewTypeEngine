#include "newtype/scene/ProceduralGeometry.h"
#include "newtype/scene/VATLoader.h"
#if !NT_ENABLE_PROCEDURAL
// ProceduralGeometry not enabled — empty translation unit
#else

#include "cinder/Log.h"
#include "cinder/app/App.h"
#include "newtype/util/Profiler.h"
#include "newtype/core/ShaderManager.h"
#include <filesystem>

namespace newtype::scene {
using namespace luisa::compute;

// Helper: apply permutation to a vector
template<typename T>
static luisa::vector<T> permute(const luisa::vector<T>& src, const luisa::vector<uint>& perm) {
    luisa::vector<T> dst(src.size());
    for (uint i = 0u; i < static_cast<uint>(perm.size()); ++i)
        dst[i] = src[perm[i]];
    return dst;
}

// Helper: compute world-space AABB for a rotated cube
static AABB compute_cube_aabb(luisa::float3 center, float half_extent,
                               luisa::float4 rotation) noexcept {
    // 8 corners of [-h,h]^3 in local space
    float h = half_extent;
    luisa::float3 corners[8] = {
        {-h,-h,-h}, {h,-h,-h}, {-h,h,-h}, {h,h,-h},
        {-h,-h, h}, {h,-h, h}, {-h,h, h}, {h,h, h}
    };
    // Rotation matrix from quaternion
    float x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w;
    float xx=x*x, yy=y*y, zz=z*z;
    float xy=x*y, xz=x*z, yz=y*z;
    float wx=w*x, wy=w*y, wz=w*z;
    float m[3][3] = {
        {1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)},
        {2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)},
        {2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)}
    };
    luisa::float3 bmin = {1e10f, 1e10f, 1e10f};
    luisa::float3 bmax = {-1e10f, -1e10f, -1e10f};
    for (int i = 0; i < 8; ++i) {
        auto& c = corners[i];
        luisa::float3 r = {
            m[0][0]*c.x + m[0][1]*c.y + m[0][2]*c.z,
            m[1][0]*c.x + m[1][1]*c.y + m[1][2]*c.z,
            m[2][0]*c.x + m[2][1]*c.y + m[2][2]*c.z
        };
        luisa::float3 world = center + r;
        bmin = luisa::min(bmin, world);
        bmax = luisa::max(bmax, world);
    }
    AABB aabb;
    aabb.packed_min = {bmin.x, bmin.y, bmin.z};
    aabb.packed_max = {bmax.x, bmax.y, bmax.z};
    return aabb;
}

ProceduralGeometry::ProceduralGeometry(Device& device) noexcept
    : _device(device), _deformShaderBlock(256u) {
    // VAT interpolation shader: 2D dispatch — x = instance index, y = vertex stride (0..255).
    // One block per instance (set_block_size(1, 256, 1)), shared memory AABB reduction.
    _vatInterpolateShader = device.compile<2>(
        [](BufferVar<uint> dispatch_indices,
           BufferVar<luisa::float4> active_pos,
           BufferVar<luisa::float4> active_norm,
           BufferVar<AABB> aabbs,
           BufferVar<ProcInstanceData> instances,
           BufferVar<luisa::float4> all_pos,
           BufferVar<luisa::float4> all_norm,
           BufferVar<ProcMeshMeta> mesh_meta) noexcept {
            set_block_size(1u, 256u, 1u);

            UInt instance_idx = dispatch_indices.read(dispatch_id().x);
            UInt tid = thread_id().y;
            Var<ProcInstanceData> inst = instances.read(instance_idx);

            Var<ProcMeshMeta> meta = mesh_meta.read(inst.mesh_id);
            UInt frame_a = clamp(cast<UInt>(floor(inst.param)),
                0u, inst.frame_count - 1u);
            UInt frame_b = min(frame_a + 1u, inst.frame_count - 1u);
            Float frac = fract(inst.param);
            UInt active_base = inst.packed_offsets & 0xFFFFu;
            UInt all_base = meta.all_data_base;

            Float3 local_min = make_float3(1e10f);
            Float3 local_max = make_float3(-1e10f);

            // Each thread processes vertices [tid, tid+256, tid+512, ...]
            UInt max_iter = (meta.vertex_count + 255u) / 256u;
            $for(iter, max_iter) {
                UInt v = tid + iter * 256u;
                $if(v < meta.vertex_count) {
                    UInt src_a = all_base + frame_a * meta.vertex_count + v;
                    UInt src_b = all_base + frame_b * meta.vertex_count + v;

                    Float4 pa = all_pos.read(src_a);
                    Float4 pb = all_pos.read(src_b);
                    Float4 p = lerp(pa, pb, frac);
                    active_pos.write(active_base + v, p);
                    local_min = min(local_min, p.xyz());
                    local_max = max(local_max, p.xyz());

                    Float4 na = all_norm.read(src_a);
                    Float4 nb = all_norm.read(src_b);
                    Float4 n = make_float4(normalize(lerp(na.xyz(), nb.xyz(), frac)),
                                           lerp(na.w, nb.w, frac));
                    active_norm.write(active_base + v, n);
                };
            };

            // Shared memory AABB reduction (R3 wave/smem pass). The tree stops
            // at 32 partials — exactly one warp — and a single
            // warp_active_min/max pair replaces the five sub-warp tree steps
            // and their barriers (lc_optimize §4.5 two-level pattern; min/max
            // are order-invariant over non-NaN inputs, so the AABB is
            // bit-identical to a full correct tree for these finite VAT
            // positions and 1e10 sentinels).
            // BUG FIX carried by the rewrite: the legacy first step merged
            // smax with stride +64 at the tid<128 gate (copy-paste from the
            // next step; smin used the correct +128). Deterministic outcome:
            // threads 192-255's local maxima never reached the final AABB —
            // the max side under-covered a quarter of the vertices whenever
            // the extreme vertex's tid landed there. smin was unaffected.
            Shared<float3> smin(256u);
            Shared<float3> smax(256u);
            smin.write(tid, local_min);
            smax.write(tid, local_max);
            sync_block();

            $if(tid < 128u) { smin.write(tid, min(smin.read(tid), smin.read(tid + 128u))); smax.write(tid, max(smax.read(tid), smax.read(tid + 128u))); };
            sync_block();
            $if(tid < 64u) { smin.write(tid, min(smin.read(tid), smin.read(tid + 64u))); smax.write(tid, max(smax.read(tid), smax.read(tid + 64u))); };
            sync_block();
            $if(tid < 32u) { smin.write(tid, min(smin.read(tid), smin.read(tid + 32u))); smax.write(tid, max(smax.read(tid), smax.read(tid + 32u))); };
            sync_block();

            // Lanes 0..31 hold the remaining 32 partials; one warp collective
            // broadcasts the block min/max (diverged lanes 32..255 are simply
            // excluded from the reduction).
            $if(tid < 32u) {
                Float3 block_min = warp_active_min(smin.read(tid));
                Float3 block_max = warp_active_max(smax.read(tid));
                $if(tid == 0u) {
                    Var<AABB> aabb;
                    aabb.packed_min = { block_min.x, block_min.y, block_min.z };
                    aabb.packed_max = { block_max.x, block_max.y, block_max.z };
                    aabbs.write(instance_idx, aabb);
                };
            };
        });
}

uint ProceduralGeometry::add_vat_mesh(
    luisa::span<const luisa::float3> positions,
    luisa::span<const luisa::float3> normals,
    luisa::span<const compute::Triangle> indices,
    uint vertex_count, uint frame_count,
    luisa::span<const luisa::float2> uvs) noexcept {

    LUISA_ASSERT(!_built, "Cannot add VAT mesh after build()");
    LUISA_ASSERT(positions.size() >= frame_count * vertex_count, "Position buffer too small");
    LUISA_ASSERT(normals.size() >= frame_count * vertex_count, "Normal buffer too small");

    uint mesh_id = static_cast<uint>(_vatMeshes.size());
    uint tri_count = static_cast<uint>(indices.size());

    // Performance warning for high-poly VAT meshes
    if (tri_count > 1000u) {
        CI_LOG_W("VAT mesh " << mesh_id << " has " << tri_count
            << " triangles - recommended limit is 1000 for performance");
    }

    VATMeshRegistry reg{};
    reg.vertex_count = vertex_count;
    reg.frame_count = frame_count;
    reg.tri_count = tri_count;

    reg.positions_cpu.assign(positions.begin(), positions.end());
    reg.normals_cpu.assign(normals.begin(), normals.end());
    reg.indices_cpu.assign(indices.begin(), indices.end());
    if (!uvs.empty()) {
        reg.uvs_cpu.assign(uvs.begin(), uvs.end());
    }

    _vatMeshes.push_back(std::move(reg));
    return mesh_id;
}

uint ProceduralGeometry::add_instance(uint mesh_id, uint32_t material_layers,
                                      float fps, float time_offset, float speed,
                                      bool double_sided) noexcept {
    LUISA_ASSERT(!_built, "Cannot add instances after build()");
    LUISA_ASSERT(mesh_id < _vatMeshes.size(), "Invalid mesh_id");

    uint idx = _instanceCount;
    _instanceCount++;

    AnimState anim{};
    anim.fps = fps;
    anim.time_offset = time_offset;
    anim.speed = speed;
    _animState.push_back(anim);

    ProcInstanceData inst{};
    inst.type = 0u | (double_sided ? kProcDoubleSided : 0u);
    inst.material_layers = material_layers;
    inst.mesh_id = mesh_id;
    inst.packed_offsets = 0u; // Set in build()
    inst.param = 0.0f; // current_frame, advanced in update()
    inst.vertex_count = _vatMeshes[mesh_id].vertex_count;
    inst.tri_count = _vatMeshes[mesh_id].tri_count;
    inst.frame_count = _vatMeshes[mesh_id].frame_count;
    inst.rotation = luisa::make_float4(0.f, 0.f, 0.f, 1.f);

    _instances_cpu.push_back(inst);
    _builtinAabbs.push_back({}); // placeholder
    return idx;
}

uint ProceduralGeometry::add_sphere(luisa::float3 center, float radius,
                                     uint32_t material_layers,
                                     luisa::float4 rotation,
                                     bool double_sided) noexcept {
    LUISA_ASSERT(!_built, "Cannot add instances after build()");

    uint idx = _instanceCount;
    _instanceCount++;

    AnimState anim{};
    _animState.push_back(anim);

    ProcInstanceData inst{};
    inst.type = 1u | (double_sided ? kProcDoubleSided : 0u);
    inst.material_layers = material_layers;
    inst.mesh_id = ~0u;
    inst.packed_offsets = 0u;
    inst.param = 0.0f;
    inst.vertex_count = 0u;
    inst.tri_count = 0u;
    inst.frame_count = 0u;
    inst.rotation = rotation;

    _instances_cpu.push_back(inst);

    // Compute AABB (sphere — rotation doesn't affect AABB)
    AABB aabb;
    aabb.packed_min = {center.x - radius, center.y - radius, center.z - radius};
    aabb.packed_max = {center.x + radius, center.y + radius, center.z + radius};
    _builtinAabbs.push_back(aabb);
    return idx;
}

uint ProceduralGeometry::add_cube(luisa::float3 center, float half_extent,
                                   uint32_t material_layers,
                                   luisa::float4 rotation,
                                   bool double_sided) noexcept {
    LUISA_ASSERT(!_built, "Cannot add instances after build()");

    uint idx = _instanceCount;
    _instanceCount++;

    AnimState anim{};
    _animState.push_back(anim);

    ProcInstanceData inst{};
    inst.type = 2u | (double_sided ? kProcDoubleSided : 0u);
    inst.material_layers = material_layers;
    inst.mesh_id = ~0u;
    inst.packed_offsets = 0u;
    inst.param = half_extent;
    inst.vertex_count = 0u;
    inst.tri_count = 0u;
    inst.frame_count = 0u;
    inst.rotation = rotation;

    _instances_cpu.push_back(inst);

    // Compute rotated AABB
    _builtinAabbs.push_back(compute_cube_aabb(center, half_extent, rotation));
    return idx;
}

uint ProceduralGeometry::add_static_mesh(
    luisa::span<const luisa::float3> positions,
    luisa::span<const luisa::float3> normals,
    luisa::span<const compute::Triangle> indices,
    uint vertex_count,
    luisa::span<const luisa::float2> uvs) noexcept {

    LUISA_ASSERT(!_built, "Cannot add static mesh after build()");
    LUISA_ASSERT(positions.size() >= vertex_count, "Position buffer too small");
    LUISA_ASSERT(normals.size() >= vertex_count, "Normal buffer too small");

    uint mesh_id = static_cast<uint>(_deformMeshes.size());
    uint tri_count = static_cast<uint>(indices.size());

    DeformMeshRegistry reg{};
    reg.vertex_count = vertex_count;
    reg.tri_count = tri_count;
    reg.positions_cpu.assign(positions.begin(), positions.end());
    reg.normals_cpu.assign(normals.begin(), normals.end());
    reg.indices_cpu.assign(indices.begin(), indices.end());
    if (!uvs.empty()) {
        reg.uvs_cpu.assign(uvs.begin(), uvs.end());
    }

    _deformMeshes.push_back(std::move(reg));
    return mesh_id;
}

uint ProceduralGeometry::add_static_mesh(ci::TriMesh& triMesh) noexcept {
    LUISA_ASSERT(!_built, "Cannot add static mesh after build()");

    const auto& vertexIndices = triMesh.getIndices();
    LUISA_ASSERT(!vertexIndices.empty(), "TriMesh must have indexed vertices");
    if (!triMesh.hasNormals()) triMesh.recalculateNormals();

    const ci::vec3* srcPos = triMesh.getPositions<3>();
    const auto& srcNorm = triMesh.getNormals();
    uint vertexCount = triMesh.getNumVertices();

    luisa::vector<luisa::float3> positions(vertexCount);
    luisa::vector<luisa::float3> normals(vertexCount);
    for (uint i = 0u; i < vertexCount; ++i) {
        positions[i] = luisa::make_float3(srcPos[i].x, srcPos[i].y, srcPos[i].z);
        normals[i]   = luisa::make_float3(srcNorm[i].x, srcNorm[i].y, srcNorm[i].z);
    }

    // Extract UVs if available
    luisa::vector<luisa::float2> uvs;
    if (triMesh.hasTexCoords()) {
        const ci::vec2* srcTexCoords = triMesh.getTexCoords0<2>();
        uvs.resize(vertexCount);
        for (uint i = 0u; i < vertexCount; ++i) {
            uvs[i] = luisa::make_float2(srcTexCoords[i].x, srcTexCoords[i].y);
        }
    }

    uint triCount = static_cast<uint>(vertexIndices.size() / 3u);
    luisa::vector<compute::Triangle> triangles(triCount);
    for (uint t = 0u; t < triCount; ++t) {
        triangles[t] = compute::Triangle{
            static_cast<uint>(vertexIndices[t * 3u + 0u]),
            static_cast<uint>(vertexIndices[t * 3u + 1u]),
            static_cast<uint>(vertexIndices[t * 3u + 2u])};
    }

    return add_static_mesh(positions, normals, triangles, vertexCount,
        uvs.empty() ? luisa::span<const luisa::float2>{} : 
                      luisa::span<const luisa::float2>{uvs.data(), uvs.size()});
}

uint ProceduralGeometry::add_deformable_instances(
    uint mesh_id, uint count,
    uint32_t material_layers,
    luisa::span<const ProcDeformState> initial_states,
    bool double_sided) noexcept {

    LUISA_ASSERT(!_built, "Cannot add instances after build()");
    LUISA_ASSERT(mesh_id < _deformMeshes.size(), "Invalid mesh_id");

    auto& mesh = _deformMeshes[mesh_id];
    uint first_idx = _instanceCount;

    for (uint i = 0u; i < count; ++i) {
        _instanceCount++;
        _animState.push_back({}); // no animation state for deformable

        ProcInstanceData inst{};
        inst.type = 3u | (double_sided ? kProcDoubleSided : 0u);
        inst.material_layers = material_layers;
        inst.mesh_id = mesh_id;
        inst.packed_offsets = 0u; // set in build()
        inst.param = static_cast<float>(i); // local frame index
        inst.vertex_count = mesh.vertex_count;
        inst.tri_count = mesh.tri_count;
        inst.frame_count = count; // number of instances in this batch
        inst.rotation = luisa::make_float4(0.f, 0.f, 0.f, 1.f);

        _instances_cpu.push_back(inst);
        _builtinAabbs.push_back({}); // placeholder, computed by deform shader

        // Deform state
        if (i < initial_states.size()) {
            _deformStateCPU.push_back(initial_states[i]);
        } else {
            _deformStateCPU.push_back(ProcDeformState{});
        }
    }

    _deformInstanceCount += count;
    return first_idx;
}

void ProceduralGeometry::set_deform_state(uint instance_idx, const ProcDeformState& state) noexcept {
    LUISA_ASSERT(instance_idx < _instanceCount, "Invalid instance index");
    uint sorted_idx = _built ? _originalToSorted[instance_idx] : instance_idx;
    LUISA_ASSERT((_instances_cpu[sorted_idx].type & 0xFu) == 3u, "Instance is not deformable");
    uint deform_idx = sorted_idx - _deformStartIdx;
    _deformStateCPU[deform_idx] = state;
    if (_built) {
        _dirtyDeformIndices.push_back(deform_idx);
    }
}

void ProceduralGeometry::set_deform_shader_id(luisa::string_view shaderId, uint blockSize) noexcept {
    _deformShader.assign(shaderId);
    _deformShaderBlock = blockSize;
}

luisa::vector<ProceduralGeometry::VATLoadResult>
ProceduralGeometry::add_vat_from_file(const std::filesystem::path& path) noexcept {
    LUISA_ASSERT(!_built, "Cannot add VAT mesh after build()");

    luisa::vector<VATLoadResult> results;

    // Load VAT file (include texcoords for UV packing); V1 packed files yield
    // one VATData per topology, V0 files yield a single one
    auto topologies = VATLoader::load_all(path, /*skip_texcoords=*/false);
    if (topologies.empty())
        return results;

    results.reserve(topologies.size());
    for (auto& vat_data : topologies) {
        // Convert indices to Triangle array
        uint32_t tri_count = vat_data.index_count / 3u;
        luisa::vector<compute::Triangle> triangles(tri_count);
        for (uint32_t t = 0; t < tri_count; ++t) {
            triangles[t] = compute::Triangle{
                vat_data.indices[t * 3u], vat_data.indices[t * 3u + 1u], vat_data.indices[t * 3u + 2u]};
        }

        // Convert flat texcoords [vertex_count * 2] to float2 array
        luisa::vector<luisa::float2> uvs;
        if (!vat_data.texcoords.empty()) {
            uvs.resize(vat_data.vertex_count);
            for (uint32_t v = 0u; v < vat_data.vertex_count; ++v) {
                uvs[v] = luisa::make_float2(
                    vat_data.texcoords[v * 2u], vat_data.texcoords[v * 2u + 1u]);
            }
        }

        uint mesh_id = add_vat_mesh(
            luisa::span<const luisa::float3>{vat_data.positions.data(), vat_data.positions.size()},
            luisa::span<const luisa::float3>{vat_data.normals.data(), vat_data.normals.size()},
            luisa::span<const compute::Triangle>{triangles.data(), triangles.size()},
            vat_data.vertex_count, vat_data.frame_count,
            uvs.empty() ? luisa::span<const luisa::float2>{} :
                luisa::span<const luisa::float2>{uvs.data(), uvs.size()});

        CI_LOG_D("ProceduralGeometry: Loaded VAT " << path.filename().string()
                 << " verts:" << vat_data.vertex_count << " frames:" << vat_data.frame_count
                 << " tris:" << tri_count);

        results.push_back(VATLoadResult{mesh_id, vat_data.vertex_count, vat_data.frame_count, tri_count});
    }

    return results;
}

void ProceduralGeometry::build(Stream& stream) noexcept {
    if (_instanceCount == 0u) return;

    const uint vat_count = static_cast<uint>(_vatMeshes.size());
    const uint deform_count = static_cast<uint>(_deformMeshes.size());

    // --- Cache flags before sort ---
    _has_vat = false;
    _has_deform = false;
    for (auto& inst : _instances_cpu) {
        if ((inst.type & 0xFu) == 0u) _has_vat = true;
        if ((inst.type & 0xFu) == 3u) _has_deform = true;
    }

    // --- Stable partition: type != 3 first, type == 3 last ---
    if (_has_deform) {
        luisa::vector<uint> perm(_instanceCount);
        uint non_deform = 0u;
        for (uint i = 0u; i < _instanceCount; ++i) {
            if ((_instances_cpu[i].type & 0xFu) != 3u) perm[non_deform++] = i;
        }
        _deformStartIdx = non_deform;
        uint d = non_deform;
        for (uint i = 0u; i < _instanceCount; ++i) {
            if ((_instances_cpu[i].type & 0xFu) == 3u) perm[d++] = i;
        }

        // Build reverse mapping: original idx → sorted idx
        _originalToSorted.resize(_instanceCount);
        for (uint i = 0u; i < _instanceCount; ++i) {
            _originalToSorted[perm[i]] = i;
        }

        // Apply permutation to all parallel arrays
        _instances_cpu = permute(_instances_cpu, perm);
        _animState = permute(_animState, perm);
        _builtinAabbs = permute(_builtinAabbs, perm);
    } else {
        _deformStartIdx = _instanceCount;
        _originalToSorted.resize(_instanceCount);
        for (uint i = 0u; i < _instanceCount; ++i) _originalToSorted[i] = i;
    }

    // --- Remap deform instance mesh_id: deform uses [vat_count..vat_count+deform_count) ---
    for (auto& inst : _instances_cpu) {
        if ((inst.type & 0xFu) == 3u) {
            inst.mesh_id += vat_count;
        }
    }

    // --- Count totals ---
    // Source data (all VAT frames)
    uint total_all_positions = 0u;
    // Active data (per-instance, single frame — VAT + deform)
    uint total_active_positions = 0u;
    uint total_active_normals = 0u;
    uint total_indices = 0u;
    uint total_uvs = 0u;

    // VAT meshes — source data sizing
    for (auto& mesh : _vatMeshes) {
        uint all_pos_count = mesh.frame_count * mesh.vertex_count;
        mesh.all_data_base = total_all_positions;
        mesh.index_base = total_indices;
        mesh.uv_base = total_uvs;
        total_all_positions += all_pos_count;
        total_indices += mesh.tri_count;
        total_uvs += mesh.vertex_count;
    }

    // VAT meshes — count per-mesh instances for active sizing
    luisa::vector<uint> vat_instance_counts(vat_count, 0u);
    for (auto& inst : _instances_cpu) {
        if ((inst.type & 0xFu) == 0u) {
            ++vat_instance_counts[inst.mesh_id];
        }
    }

    // Assign active position offsets for VAT instances
    for (uint m = 0u; m < vat_count; ++m) {
        auto& mesh = _vatMeshes[m];
        mesh.position_base = total_active_positions;
        total_active_positions += vat_instance_counts[m] * mesh.vertex_count;
        total_active_normals += vat_instance_counts[m] * mesh.vertex_count;
    }

    // --- Update VAT instance offsets (point into active buffer, 1 frame per instance) ---
    luisa::vector<uint> vat_local_counters(vat_count, 0u);
    for (auto& inst : _instances_cpu) {
        if ((inst.type & 0xFu) != 0u) continue;
        auto& mesh = _vatMeshes[inst.mesh_id];
        uint local_idx = vat_local_counters[inst.mesh_id]++;
        inst.packed_offsets = pack_proc_offsets(mesh.index_base,
            mesh.position_base + local_idx * mesh.vertex_count);
    }

    // Deformable meshes — count per-topology instances and assign offsets
    if (_has_deform) {
        for (uint mesh_idx = 0u; mesh_idx < _deformMeshes.size(); ++mesh_idx) {
            auto& dm = _deformMeshes[mesh_idx];
            dm.position_base = total_active_positions;
            dm.index_base = total_indices;
            dm.deform_base_pos_offset = total_active_positions;
            dm.uv_base = total_uvs;

            uint instance_count = 0u;
            // Remapped mesh_id = vat_count + mesh_idx
            uint remapped_id = vat_count + mesh_idx;
            for (uint i = _deformStartIdx; i < _instanceCount; ++i) {
                if (_instances_cpu[i].mesh_id == remapped_id) ++instance_count;
            }
            total_active_positions += instance_count * dm.vertex_count;
            total_active_normals += instance_count * dm.vertex_count;
            total_indices += dm.tri_count;
            total_uvs += dm.vertex_count;
        }

        // Set packed_offsets for deformable instances (O(N) single pass)
        luisa::vector<uint> mesh_local_counters(_deformMeshes.size(), 0u);
        for (uint i = _deformStartIdx; i < _instanceCount; ++i) {
            auto& inst = _instances_cpu[i];
            uint local_mesh_idx = inst.mesh_id - vat_count;
            auto& dm = _deformMeshes[local_mesh_idx];
            uint local_idx = mesh_local_counters[local_mesh_idx]++;
            inst.packed_offsets = pack_proc_offsets(dm.index_base,
                dm.position_base + local_idx * dm.vertex_count);
        }
    }

    // --- Allocate source data buffers (all VAT frames) ---
    _vatAllPositions = _device.create_buffer<luisa::float4>(max(1u, total_all_positions));
    _vatAllNormals = _device.create_buffer<luisa::float4>(max(1u, total_all_positions));

    // --- Allocate active data buffers (per-instance, single frame) ---
    _vatPositions = _device.create_buffer<luisa::float4>(max(1u, total_active_positions));
    _vatNormals = _device.create_buffer<luisa::float4>(max(1u, total_active_normals));
    _vatIndices = _device.create_buffer<compute::Triangle>(max(1u, total_indices));
    _procUVs = _device.create_buffer<luisa::float2>(max(1u, total_uvs));

    // --- Build ProcMeshMeta buffer (VAT topologies + deform topologies) ---
    uint total_topologies = vat_count + deform_count;
    _procMeshMeta = _device.create_buffer<ProcMeshMeta>(max(1u, total_topologies));
    if (total_topologies > 0u) {
        luisa::vector<ProcMeshMeta> meta_cpu(total_topologies);
        for (uint m = 0u; m < vat_count; ++m) {
            auto& mesh = _vatMeshes[m];
            meta_cpu[m].all_data_base = mesh.all_data_base;
            meta_cpu[m].vertex_count = mesh.vertex_count;
            meta_cpu[m].uv_base = mesh.uv_base;
        }
        // Deform topologies: all_data_base = 0 (no source data), vertex_count and uv_base set
        for (uint m = 0u; m < deform_count; ++m) {
            auto& dm = _deformMeshes[m];
            meta_cpu[vat_count + m].all_data_base = 0u;
            meta_cpu[vat_count + m].vertex_count = dm.vertex_count;
            meta_cpu[vat_count + m].uv_base = dm.uv_base;
        }
        stream << _procMeshMeta.copy_from(meta_cpu.data());
    }

    // --- Upload source data (all VAT frames to _vatAllPositions/_vatAllNormals) ---
    for (auto& mesh : _vatMeshes) {
        uint pos_count = mesh.frame_count * mesh.vertex_count;

        luisa::vector<luisa::float4> pos4(pos_count);
        for (uint f = 0; f < mesh.frame_count; ++f) {
            for (uint v = 0; v < mesh.vertex_count; ++v) {
                uint idx = f * mesh.vertex_count + v;
                pos4[idx] = luisa::make_float4(
                    mesh.positions_cpu[idx],
                    mesh.uvs_cpu.empty() ? 0.0f : mesh.uvs_cpu[v].x);
            }
        }
        stream << _vatAllPositions.view(mesh.all_data_base, pos_count)
                    .copy_from(pos4.data());

        luisa::vector<luisa::float4> norm4(pos_count);
        for (uint f = 0; f < mesh.frame_count; ++f) {
            for (uint v = 0; v < mesh.vertex_count; ++v) {
                uint idx = f * mesh.vertex_count + v;
                norm4[idx] = luisa::make_float4(
                    mesh.normals_cpu[idx],
                    mesh.uvs_cpu.empty() ? 0.0f : mesh.uvs_cpu[v].y);
            }
        }
        stream << _vatAllNormals.view(mesh.all_data_base, pos_count)
                    .copy_from(norm4.data());

        // Upload indices (shared per topology)
        stream << _vatIndices.view(mesh.index_base, mesh.tri_count)
                    .copy_from(mesh.indices_cpu.data());

        // Upload static UVs
        if (!mesh.uvs_cpu.empty()) {
            stream << _procUVs.view(mesh.uv_base, mesh.vertex_count)
                        .copy_from(mesh.uvs_cpu.data());
        }
    }

    // Upload deform data to active buffers (replicated base meshes, packed into float4 with UVs)
    if (_has_deform) {
        for (uint mesh_idx = 0u; mesh_idx < _deformMeshes.size(); ++mesh_idx) {
            auto& dm = _deformMeshes[mesh_idx];
            uint remapped_id = vat_count + mesh_idx;

            // Upload indices (shared per topology)
            stream << _vatIndices.view(dm.index_base, dm.tri_count)
                        .copy_from(dm.indices_cpu.data());

            // Build replicated float4 positions with uv_u in .w
            luisa::vector<luisa::float4> replicated_pos;
            luisa::vector<luisa::float4> replicated_norm;
            for (uint i = 0u; i < _instanceCount; ++i) {
                if ((_instances_cpu[i].type & 0xFu) == 3u && _instances_cpu[i].mesh_id == remapped_id) {
                    for (uint v = 0u; v < dm.vertex_count; ++v) {
                        replicated_pos.push_back(luisa::make_float4(
                            dm.positions_cpu[v],
                            dm.uvs_cpu.empty() ? 0.0f : dm.uvs_cpu[v].x));
                        replicated_norm.push_back(luisa::make_float4(
                            dm.normals_cpu[v],
                            dm.uvs_cpu.empty() ? 0.0f : dm.uvs_cpu[v].y));
                    }
                }
            }
            uint total_verts = static_cast<uint>(replicated_pos.size());
            if (total_verts > 0u) {
                stream << _vatPositions.view(dm.position_base, total_verts)
                            .copy_from(replicated_pos.data());
                stream << _vatNormals.view(dm.position_base, total_verts)
                            .copy_from(replicated_norm.data());
            }

            // Upload static UVs
            if (!dm.uvs_cpu.empty()) {
                stream << _procUVs.view(dm.uv_base, dm.vertex_count)
                            .copy_from(dm.uvs_cpu.data());
            }
        }

        // Create base positions + normals buffers (one copy per topology)
        uint total_base_verts = 0u;
        for (auto& dm : _deformMeshes) {
            total_base_verts += dm.vertex_count;
        }
        _deformBasePositions = _device.create_buffer<luisa::float3>(max(1u, total_base_verts));
        _deformBaseNormals = _device.create_buffer<luisa::float3>(max(1u, total_base_verts));
        _deformBaseOffsets = _device.create_buffer<uint>(max(1u, static_cast<uint>(_deformMeshes.size())));

        uint base_offset = 0u;
        luisa::vector<uint> offsets_cpu(_deformMeshes.size());
        for (uint m = 0u; m < _deformMeshes.size(); ++m) {
            auto& dm = _deformMeshes[m];
            stream << _deformBasePositions.view(base_offset, dm.vertex_count)
                        .copy_from(dm.positions_cpu.data());
            stream << _deformBaseNormals.view(base_offset, dm.vertex_count)
                        .copy_from(dm.normals_cpu.data());
            offsets_cpu[m] = base_offset;
            base_offset += dm.vertex_count;
        }
        stream << _deformBaseOffsets.copy_from(offsets_cpu.data());

        // Create deform state buffer
        _deformStateBuffer = _device.create_buffer<ProcDeformState>(_deformInstanceCount);
        stream << _deformStateBuffer.copy_from(_deformStateCPU.data());
    }

    // --- Create GPU buffers ---
    _instanceBuffer = _device.create_buffer<ProcInstanceData>(_instanceCount);
    _aabbBuffer = _device.create_buffer<AABB>(_instanceCount);

    if (_deformInstanceCount == 0u) {
        _deformStateBuffer = _device.create_buffer<ProcDeformState>(1u);
    }

    stream << _instanceBuffer.copy_from(_instances_cpu.data());

    // Build CPU AABB staging buffer — builtin AABBs are pre-computed, VAT uses zero initially
    luisa::vector<AABB> cpu_aabbs(_instanceCount);
    for (uint i = 0; i < _instanceCount; ++i) {
        if (_instances_cpu[i].type != 0u) {
            cpu_aabbs[i] = _builtinAabbs[i];
        }
    }
    // No deform shader id: type-3 instances keep the rest-pose base mesh that
    // was uploaded above — give them static rest AABBs (collapsed when the
    // initial state hides the instance, same protocol the shader implements).
    // With a shader id the initial deform dispatch below overwrites these.
    if (_has_deform && !_deformShader.is_set()) {
        luisa::vector<AABB> rest_aabb(_deformMeshes.size());
        for (uint m = 0u; m < _deformMeshes.size(); ++m) {
            const auto& dm = _deformMeshes[m];
            luisa::float3 bmin(1e10f), bmax(-1e10f);
            for (const auto& p : dm.positions_cpu) {
                bmin = luisa::min(bmin, p);
                bmax = luisa::max(bmax, p);
            }
            rest_aabb[m].packed_min = { bmin.x, bmin.y, bmin.z };
            rest_aabb[m].packed_max = { bmax.x, bmax.y, bmax.z };
        }
        for (uint i = _deformStartIdx; i < _instanceCount; ++i) {
            const auto& inst = _instances_cpu[i];
            cpu_aabbs[i] = rest_aabb[inst.mesh_id - vat_count];
            if (_deformStateCPU[i - _deformStartIdx].params[0].w < 0.f) {
                cpu_aabbs[i].packed_min = { 1e30f, 1e30f, 1e30f };
                cpu_aabbs[i].packed_max = { -1e30f, -1e30f, -1e30f };
            }
        }
    }
    stream << _aabbBuffer.copy_from(cpu_aabbs.data());

    // --- VAT interpolation: compute active positions/normals + AABBs ---
    if (_has_vat) {
        // Collect all VAT instance indices for initial dispatch
        luisa::vector<uint> vat_indices;
        for (uint i = 0u; i < _instanceCount; ++i) {
            if ((_instances_cpu[i].type & 0xFu) == 0u) {
                vat_indices.push_back(i);
            }
        }
        uint vat_instance_count = static_cast<uint>(vat_indices.size());

        // Allocate and fill dispatch buffer
        _vatDispatchBuf = _device.create_buffer<uint>(max(1u, vat_instance_count));
        if (vat_instance_count > 0u) {
            stream << _vatDispatchBuf.view(0u, vat_instance_count)
                        .copy_from(vat_indices.data());
        }

        stream << _vatInterpolateShader(
            _vatDispatchBuf,
            _vatPositions, _vatNormals,
            _aabbBuffer, _instanceBuffer,
            _vatAllPositions, _vatAllNormals,
            _procMeshMeta)
            .dispatch(vat_instance_count, 256u);
    } else _vatDispatchBuf = _device.create_buffer<uint>(1u);

    // Run the deform shader for initial deformed positions + AABBs (type=3).
    // Without a shader id the instances simply keep the rest-pose upload above.
    if (_has_deform && _deformShader.is_set()) {
        if (_deformInstanceCount > 0u) {
            _deformStateCPU[0].params[0].x = 0.0f;
            stream << _deformStateBuffer.view(0u, 1u).copy_from(&_deformStateCPU[0]);
        }

        stream << core::ShaderManager::instance().shader(
            _deformShader,
            _vatPositions, _vatNormals,
            _aabbBuffer, _instanceBuffer,
            _deformStateBuffer, _deformBasePositions, _deformBaseNormals,
            _deformBaseOffsets, _vatIndices, _procUVs, _procMeshMeta)
            .dispatch(_deformInstanceCount, _deformShaderBlock);
    } else if (_has_deform) {
        CI_LOG_W("ProceduralGeometry: " << _deformInstanceCount
            << " deformable instance(s) without set_deform_shader_id() - "
               "rendering the rest-pose mesh (no per-frame deformation)");
    }

    // Create ProceduralPrimitive with the AABB buffer
    _blas = _device.create_procedural_primitive(_aabbBuffer.view());
    stream << _blas.build();

    _built = true;
}

void ProceduralGeometry::set_anim_state(uint instance_idx, float fps,
                                         float time_offset, float speed,
                                         bool looping) noexcept {
    uint sorted_idx = _built ? _originalToSorted[instance_idx] : instance_idx;
    LUISA_ASSERT(sorted_idx < _animState.size(), "Invalid instance index");
    _animState[sorted_idx].fps = fps;
    _animState[sorted_idx].time_offset = time_offset;
    _animState[sorted_idx].speed = speed;
    _animState[sorted_idx].looping = looping;
}

bool ProceduralGeometry::update(Stream& stream, float time) noexcept {
    if (!_built || _instanceCount == 0u) return false;

    // Advance animation frames per instance (VAT only), track dirty indices
    bool any_changed = false;
    _dirtyVatIndices.clear();
    for (uint i = 0u; i < _instanceCount; ++i) {
        auto& inst = _instances_cpu[i];
        if ((inst.type & 0xFu) != 0u) continue;
        auto& anim = _animState[i];
        float local_time = (time + anim.time_offset) * anim.speed * anim.fps;
        uint frame_count = inst.frame_count;
        float new_frame = anim.looping
            ? fmod(local_time, static_cast<float>(frame_count))
            : fmin(local_time, static_cast<float>(frame_count - 1u));
        if (new_frame != inst.param) {
            inst.param = new_frame;
            _dirty_proc_indices.push_back(i);
            _dirtyVatIndices.push_back(i);
            any_changed = true;
        }
    }

    // Deformable instances update every frame when a deform shader is set
    // (rest-pose fallback instances are static — no forced refit)
    if (_has_deform && _deformShader.is_set()) {
        any_changed = true;
    }

    // Upload dirty deform states
    if (_has_deform && !_dirtyDeformIndices.empty()) {
        for (uint idx : _dirtyDeformIndices) {
            stream << _deformStateBuffer.view(idx, 1u).copy_from(&_deformStateCPU[idx]);
        }
        _dirtyDeformIndices.clear();
    }

    if (!any_changed) return false;

    auto& profiler = util::Profiler::instance();

    // Upload only changed instance rows (O(K) where K = dirty count)
    for (uint idx : _dirty_proc_indices) {
        stream << _instanceBuffer.view(idx, 1u).copy_from(&_instances_cpu[idx]);
    }
    _dirty_proc_indices.clear();

    // VAT interpolation: recompute active positions/normals + AABBs for dirty instances
    if (_has_vat && !_dirtyVatIndices.empty()) {
        profiler.set_pass("Proc/VATInterp");
        uint dirty_count = static_cast<uint>(_dirtyVatIndices.size());
        // Reallocate dispatch buffer if needed (dirty count may vary per frame)
        if (!_vatDispatchBuf || _vatDispatchBuf.size() < dirty_count) {
            _vatDispatchBuf = _device.create_buffer<uint>(max(1u, dirty_count));
        }
        stream << _vatDispatchBuf.view(0u, dirty_count)
                    .copy_from(_dirtyVatIndices.data());
        stream << _vatInterpolateShader(
            _vatDispatchBuf,
            _vatPositions, _vatNormals,
            _aabbBuffer, _instanceBuffer,
            _vatAllPositions, _vatAllNormals,
            _procMeshMeta)
            .dispatch(dirty_count, 256u);
    }

    // Deform pass: update shared time state, then run deform shader
    if (_has_deform && _deformShader.is_set()) {
        profiler.set_pass("Proc/Deform");

        stream << core::ShaderManager::instance().shader(
                _deformShader,
                _vatPositions, _vatNormals,
                _aabbBuffer, _instanceBuffer,
                _deformStateBuffer, _deformBasePositions, _deformBaseNormals,
                _deformBaseOffsets, _vatIndices, _procUVs, _procMeshMeta)
                .dispatch(_deformInstanceCount, _deformShaderBlock);
    }

    profiler.set_pass("Proc/BLAS");
    stream << _blas.build(compute::AccelBuildRequest::PREFER_UPDATE);
    return true;
}

} // namespace newtype::scene

#endif // NT_ENABLE_PROCEDURAL
