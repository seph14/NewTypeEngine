#include "newtype/scene/VATMesh.h"
#include "newtype/scene/VATLoader.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/TypeConv.h"
#include "cinder/Log.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include <filesystem>

namespace newtype::scene {

using namespace luisa;

//==============================================================================
// Constants
//==============================================================================

static constexpr const char* kShaderName = "vat_interpolate";

// Resolved once on first dispatch; re-resolved automatically after registry
// mutations (registration, clear, DLL hot-reload).
static core::ShaderHandle<1,
    Buffer<MeshShape::GpuVertex>, Buffer<luisa::float3>, Buffer<luisa::float3>,
    uint, uint, float, bool> gInterpolateShader;

//==============================================================================
// Loading
//==============================================================================

namespace {

// Move a parsed VATData into a sequence slot at the given global frame offset
void fill_sequence(VATSequence& seq, util::VATData&& data, uint32_t frame_offset) noexcept {
    seq.vertex_count = data.vertex_count;
    seq.frame_count  = data.frame_count;
    seq.index_count  = data.index_count;
    seq.indices      = std::move(data.indices);
    seq.texcoords    = std::move(data.texcoords);
    seq.positions    = std::move(data.positions);
    seq.normals      = std::move(data.normals);
    seq.frame_offset = frame_offset;
}

} // namespace

bool VATMesh::load_folder(const std::filesystem::path& folder_path,
                           const std::string& base_name) noexcept {
    _sequences.clear();
    _name = base_name;

    // Packed V1 takes precedence: a single <base>.vat holding every topology
    auto packed_path = folder_path / (base_name + ".vat");
    if (std::filesystem::exists(packed_path)) {
        auto topologies = VATLoader::load_all(packed_path);
        if (topologies.empty()) {
            CI_LOG_E("VATMesh: Failed to load packed VAT " << packed_path.string());
            return false;
        }

        _sequences.resize(topologies.size());
        uint32_t frame_offset = 0;
        for (uint32_t i = 0; i < topologies.size(); ++i) {
            fill_sequence(_sequences[i], std::move(topologies[i]), frame_offset);
            frame_offset += _sequences[i].frame_count;
        }

        CI_LOG_I("VATMesh: Loaded " << topologies.size() << " topologies (packed), "
                 << frame_offset << " total frames from " << packed_path.string());
        return true;
    }

    // Count numbered .vat files: base_name + "0.vat", "1.vat", ...
    uint32_t count = 0;
    while (true) {
        auto file_path = folder_path / (base_name + std::to_string(count) + ".vat");
        if (!std::filesystem::exists(file_path))
            break;
        count++;
    }

    if (count == 0) {
        CI_LOG_E("VATMesh: No VAT files found in " << folder_path.string()
                 << " with base name '" << base_name << "'");
        return false;
    }

    _sequences.resize(count);
    uint32_t frame_offset = 0;

    for (uint32_t i = 0; i < count; ++i) {
        auto file_path = folder_path / (base_name + std::to_string(i) + ".vat");
        auto vat_data = VATLoader::load(file_path);
        if (!vat_data.is_valid()) {
            _sequences.clear();
            return false;
        }

        fill_sequence(_sequences[i], std::move(vat_data), frame_offset);
        frame_offset += _sequences[i].frame_count;
    }

    CI_LOG_I("VATMesh: Loaded " << count << " topologies, "
             << frame_offset << " total frames from " << folder_path.string());
    return true;
}

bool VATMesh::load_vat(const std::filesystem::path& file_path) noexcept {
    // V1 packed files append every topology; V0 files append a single one
    auto topologies = VATLoader::load_all(file_path);
    if (topologies.empty())
        return false;

    _name = file_path.filename().stem().string();

    uint32_t frame_offset = 0;
    for (auto& seq : _sequences)
        frame_offset += seq.frame_count;

    for (auto& vat_data : topologies) {
        VATSequence seq;
        fill_sequence(seq, std::move(vat_data), frame_offset);
        frame_offset += seq.frame_count;
        _sequences.push_back(std::move(seq));
    }
    return true;
}

//==============================================================================
// Tangent Computation
//==============================================================================

void VATMesh::_compute_tangents(VATSequence& seq) noexcept {
    auto& verts = seq.initial_verts;
    verts.resize(seq.vertex_count);

    // Fill from first-frame CPU data (indices [0, vertex_count))
    for (uint32_t i = 0; i < seq.vertex_count; ++i) {
        float u = seq.texcoords[i * 2u];
        float v = seq.texcoords[i * 2u + 1u];
        verts[i] = Vertex::encode(seq.positions[i], seq.normals[i],
                                   luisa::make_float4(0.f, 0.f, 1.f, 1.f),
                                   luisa::make_float2(u, v));
    }

    // Compute tangents using standard algorithm
    luisa::vector<luisa::float3> tan1(seq.vertex_count, luisa::float3{});
    luisa::vector<luisa::float3> tan2(seq.vertex_count, luisa::float3{});

    uint32_t tri_count = seq.index_count / 3u;
    for (uint32_t t = 0; t < tri_count; ++t) {
        uint32_t i0 = seq.indices[t * 3u];
        uint32_t i1 = seq.indices[t * 3u + 1u];
        uint32_t i2 = seq.indices[t * 3u + 2u];

        auto& v0 = verts[i0], &v1 = verts[i1], &v2 = verts[i2];

        luisa::float3 e1 = luisa::make_float3(v1.px - v0.px, v1.py - v0.py, v1.pz - v0.pz);
        luisa::float3 e2 = luisa::make_float3(v2.px - v0.px, v2.py - v0.py, v2.pz - v0.pz);

        luisa::float2 duv1 = luisa::make_float2(v1.u - v0.u, v1.v - v0.v);
        luisa::float2 duv2 = luisa::make_float2(v2.u - v0.u, v2.v - v0.v);

        float r = 1.0f / (duv1.x * duv2.y - duv1.y * duv2.x + 1e-8f);

        luisa::float3 tdir = luisa::make_float3(
            (duv2.y * e1.x - duv1.y * e2.x) * r,
            (duv2.y * e1.y - duv1.y * e2.y) * r,
            (duv2.y * e1.z - duv1.y * e2.z) * r
        );
        luisa::float3 bdir = luisa::make_float3(
            (-duv2.x * e1.x + duv1.x * e2.x) * r,
            (-duv2.x * e1.y + duv1.x * e2.y) * r,
            (-duv2.x * e1.z + duv1.x * e2.z) * r
        );

        tan1[i0] += tdir; tan1[i1] += tdir; tan1[i2] += tdir;
        tan2[i0] += bdir; tan2[i1] += bdir; tan2[i2] += bdir;
    }

    // Orthogonalize tangents against normals
    for (uint32_t i = 0; i < seq.vertex_count; ++i) {
        luisa::float3 n = luisa::make_float3(verts[i].nx, verts[i].ny, verts[i].nz);
        luisa::float3 t = tan1[i];

        // Gram-Schmidt orthogonalize
        t = t - n * luisa::dot(n, t);
        float len = luisa::length(t);
        if (len > 1e-8f) {
            t = t / len;
        } else {
            t = luisa::make_float3(1.f, 0.f, 0.f);
        }

        // Handedness
        float handedness = (luisa::dot(luisa::cross(n, t), tan2[i]) < 0.f) ? -1.f : 1.f;

        verts[i].tx = t.x; verts[i].ty = t.y; verts[i].tz = t.z;
        verts[i].tw = handedness;
    }
}

//==============================================================================
// Shader Registration
//==============================================================================

void VATMesh::_ensure_shader_registered(Device& device) noexcept {
    auto& sm = newtype::core::ShaderManager::instance();
    gInterpolateShader.assign(kShaderName);
    if (sm.hasShader(kShaderName))
        return;

    sm.registerShader<1>(kShaderName, [&](
        compute::BufferVar<util::ActiveVertex> out_vertices,
        compute::BufferVar<luisa::float3> pos_frames,
        compute::BufferVar<luisa::float3> norm_frames,
        compute::UInt vertex_count,
        compute::UInt frame_count,
        compute::Float frac_frame,
        compute::Bool update_tangent
    ) noexcept {
        using namespace luisa::compute;

        set_block_size(256u);
        UInt i = dispatch_x();
        
        $if(i < vertex_count) {
            // Compute frame indices
            UInt frame_a = clamp(
                cast<UInt>(floor(frac_frame)), 0u, frame_count - 1u);
            UInt frame_b = min(frame_a + 1u, frame_count - 1u);

            // If at last frame, no interpolation
            //$if(cast<float>(frame_a) >= cast<float>(frame_count) - 1.0f) {
            $if(frame_a >= frame_count - 1u) {
                frame_b = frame_a;
            };

            Float t = fract(frac_frame);

            auto safe_normalize = [](Float3 n) -> Float3 {
                Float3 res;
                Float len = length(n);
                $if(len < .01f) {
                    // Return Y-up default for zero/near-zero normals
                    res = compute::make_float3(0.f, 1.f, 0.f);
                } $else {
                    res = n / len;
                };
                return res;
            };

            // Read positions
            Float3 pos_a = pos_frames.read(frame_a * vertex_count + i);
            Float3 pos_b = pos_frames.read(frame_b * vertex_count + i);

            // Read normals
            Float3 norm_a = norm_frames.read(frame_a * vertex_count + i);
            Float3 norm_b = norm_frames.read(frame_b * vertex_count + i);

            // Interpolate
            Float3 pos  = lerp(pos_a, pos_b, t);
            Float3 norm = safe_normalize(lerp(norm_a, norm_b, t));
          
            // Write to vertex buffer (preserve tangent + UV by reading existing).
            // A2: layout-generic setters re-encode normal/tangent when the
            // active GPU layout packs them (positions are fp32 everywhere).
            using GpuVert = util::ActiveVertex;
            Var<GpuVert> v = out_vertices.read(i);
            v.px = pos.x; v.py = pos.y; v.pz = pos.z;
            util::vertex_set_normal(v, norm);

            $if(update_tangent) {
                // Recompute tangent from interpolated normal
                Float3 up = ite(
                    abs(norm.y) < 0.999f,
                    luisa::compute::make_float3(0.f, 1.f, 0.f),
                    luisa::compute::make_float3(1.f, 0.f, 0.f)
                );
                Float3 tangent = safe_normalize(cross(up, norm));
                util::vertex_set_tangent(v, tangent, 1.0f);
            } 
            $else{
                util::vertex_set_tangent(v, luisa::compute::make_float3(1.f, 0.f, 0.f), 1.0f);
            };

            out_vertices.write(i, v);
        };
    });

    CI_LOG_D("VATMesh: Registered interpolation shader '" << kShaderName << "'");
}

//==============================================================================
// Build
//==============================================================================

void VATMesh::build(newtype::core::Pipeline& pipeline, Stream& stream, uint material_id) noexcept {
    if (_sequences.empty()) {
        CI_LOG_E("VATMesh: No sequences loaded, cannot build");
        return;
    }

    // Register shared shader (idempotent)
    _ensure_shader_registered(_device);

    _shape_ids.clear();
    _shape_ids.reserve(_sequences.size());

    for (uint32_t i = 0; i < _sequences.size(); ++i) {
        auto& seq = _sequences[i];

        // Compute tangents + fill initial vertices from first-frame CPU data
        _compute_tangents(seq);

        // Upload frame data to GPU
        uint32_t total_verts = seq.frame_count * seq.vertex_count;
        seq.positions_gpu = _device.create_buffer<luisa::float3>(total_verts);
        seq.normals_gpu   = _device.create_buffer<luisa::float3>(total_verts);
        stream << seq.positions_gpu.copy_from(seq.positions.data())
               << seq.normals_gpu.copy_from  (seq.normals.data());

        // Convert indices (uint32_t array) to Triangle array
        uint32_t tri_count = seq.index_count / 3u;
        luisa::vector<compute::Triangle> triangles(tri_count);
        for (uint32_t t = 0; t < tri_count; ++t) {
            triangles[t] = compute::Triangle{
                seq.indices[t * 3u],
                seq.indices[t * 3u + 1u],
                seq.indices[t * 3u + 2u]
            };
        }

        // Create DeformableMesh
        auto mesh = DeformableMesh::create(_device, /*requireDoubleBuffer=*/false, material_id);
        mesh->set_data(
            luisa::span<const Vertex>(seq.initial_verts.data(), seq.initial_verts.size()),
            luisa::span<const compute::Triangle>(triangles.data(), triangles.size())
        );
        mesh->set_double_sided(_double_sided);
        mesh->build(stream);

        // Register with Pipeline (static identity transform)
        auto id = pipeline.addShape(std::move(mesh), &_transform);
        _shape_ids.push_back(id);

        CI_LOG_D("VATMesh: Built topology " << i
                 << " (verts:" << seq.vertex_count
                 << " tris:"   << tri_count
                 << " frames:" << seq.frame_count << ")");

        // free cpu data
        seq.positions.clear();
        seq.normals.clear();
        seq.initial_verts.clear();
        seq.indices.clear();
    }

    // Hide all topologies except the first
    for (uint32_t i = 1; i < _shape_ids.size(); ++i) {
        pipeline.setShapeVisibility(_shape_ids[i], false);
    }

    _active = 0;
    _built = true;
}

//==============================================================================
// Update
//==============================================================================

bool VATMesh::_compute_frame(uint& out_topo_idx,
                              float& out_local_frame) const noexcept {
    float total     = static_cast<float>(total_frames());
    float raw_frame = _frame;
   
    // Apply play mode to get a [0, total) frame value
    float global_frame;
    if (total <= 0.f) {
        global_frame = 0.f;
    } else {
        switch (_play_mode) {
        default:
        case VATPlayMode::Clamp:
            global_frame = glm::min(raw_frame, total - 1.0f);
            break;
        case VATPlayMode::Loop:
            global_frame = glm::mod(raw_frame, total);
            if (global_frame < 0.f) global_frame += total;
            break;
        case VATPlayMode::PingPong: {
            // Ping-pong: goes 0→total→0→total...
            float cycle = total * 2.0f - 2.0f; // period of one full ping-pong
            if (cycle <= 0.f) {
                global_frame = 0.f;
            } else {
                float t = glm::mod(raw_frame, cycle);
                if (t < 0.f) t += cycle;
                global_frame = (t < total) ? t : (cycle - t);
            }
            break;
        }
        }
    }

    if (global_frame >= total-1.f) 
        return false;

    // Find active topology
    out_topo_idx = 0;
    for (uint32_t i = 0; i < _sequences.size(); ++i) {
        float seq_end = static_cast<float>(_sequences[i].frame_offset + _sequences[i].frame_count);
        if (global_frame < seq_end || i == _sequences.size() - 1) {
            out_topo_idx = i;
            break;
        }
    }

    // Local frame within active topology, clamped to valid range
    auto& seq = _sequences[out_topo_idx];
    out_local_frame = global_frame - static_cast<float>(seq.frame_offset);
    float max_frame = static_cast<float>(seq.frame_count) - 1.0f;
    out_local_frame = glm::clamp(out_local_frame, .0f, max_frame);

    return true;
}

void VATMesh::update(newtype::core::Pipeline& pipeline, float dt) noexcept {
    if (_transform.is_dirty()) {
        apply_transform(pipeline);
        _transform.clear_dirty();
    }

    if (!_built || _sequences.empty() || !_playing || !_visible)
        return;

    // Accumulate frame
    _frame += dt * _fps * _speed;

    // Determine active topology and fractional frame
    uint topo_idx;
    float local_frame;
    if (!_compute_frame(topo_idx, local_frame)) return;

    // Handle topology switch
    if (topo_idx != _active) {
        pipeline.setShapeVisibility(_shape_ids[_active], false);
        pipeline.setShapeVisibility(_shape_ids[topo_idx], true);
        _active = topo_idx;
    }

    // Get active DeformableMesh
    auto* deformable = pipeline.getDeformable(_shape_ids[_active]);
    if (!deformable) return;

    // Dispatch interpolation shader on compute stream (same queue as BLAS rebuild)
    auto& seq = _sequences[_active];
    auto& sm = newtype::core::ShaderManager::instance();

    pipeline.computeStream() << sm.shader(
        gInterpolateShader,
        deformable->next_vertex_buffer(),
        seq.positions_gpu,
        seq.normals_gpu,
        seq.vertex_count,
        seq.frame_count,
        glm::clamp(local_frame, 0.f, static_cast<float>(seq.frame_count) - 1.f),
        _update_tangent
    ).dispatch(seq.vertex_count);

    // Mark dirty so Geometry::update() rebuilds BLAS
    deformable->mark_buffer_dirty();
}

void VATMesh::reset() noexcept {
    _active = 0;
    _frame = 0.0f;
}

//==============================================================================
// Accessors
//==============================================================================

DeformableMesh* VATMesh::active_deformable(newtype::core::Pipeline& pipeline) const noexcept {
    if (_shape_ids.empty())
        return nullptr;
    return pipeline.getDeformable(_shape_ids[_active]);
}

uint32_t VATMesh::total_frames() const noexcept {
    if (_sequences.empty()) return 0;
    auto& last = _sequences.back();
    return last.frame_offset + last.frame_count;
}

float VATMesh::total_duration() const noexcept {
    return static_cast<float>(total_frames()) / _fps;
}

float VATMesh::current_frame() const noexcept {
    return _frame;
}

//==============================================================================
// Shape ID / Visibility / Material / Transform
//==============================================================================

ShapeId VATMesh::shape_id() const noexcept {
    if (_shape_ids.empty()) return kInvalidShapeId;
    return _shape_ids[_active];
}

ShapeId VATMesh::shape_id(uint topology_index) const noexcept {
    if (topology_index >= _shape_ids.size()) return kInvalidShapeId;
    return _shape_ids[topology_index];
}

const luisa::vector<ShapeId>& VATMesh::shape_ids() const noexcept {
    return _shape_ids;
}

void VATMesh::set_visible(core::Pipeline& pipeline, bool visible) noexcept {
    _visible = visible;
    for (uint32_t i = 0; i < _shape_ids.size(); ++i) {
        pipeline.setShapeVisibility(_shape_ids[i], visible && i == _active);
    }
}

bool VATMesh::is_visible() const noexcept {
    return _visible;
}

void VATMesh::set_material(core::Pipeline& pipeline, uint material_id) noexcept {
    uint32_t layers = 0xFFFFFF00u | (material_id & 0xFFu);
    for (auto id : _shape_ids)
        pipeline.setShapeMaterial(id, layers);
}

void VATMesh::set_material(core::Pipeline& pipeline, uint topology_index, uint material_id) noexcept {
    if (topology_index < _shape_ids.size()) {
        uint32_t layers = 0xFFFFFF00u | (material_id & 0xFFu);
        pipeline.setShapeMaterial(_shape_ids[topology_index], layers);
    }
}

void VATMesh::set_material_layers(core::Pipeline& pipeline, uint32_t layers) noexcept {
    for (auto id : _shape_ids)
        pipeline.setShapeMaterial(id, layers);
}

void VATMesh::set_material_layers(core::Pipeline& pipeline, uint topology_index, uint32_t layers) noexcept {
    if (topology_index < _shape_ids.size())
        pipeline.setShapeMaterial(_shape_ids[topology_index], layers);
}

void VATMesh::set_layer(core::Pipeline& pipeline, uint layer, uint8_t idx) noexcept {
    for (auto id : _shape_ids) {
        auto* shape = pipeline.getShape(id);
        if (!shape) continue;
        uint32_t cur = shape->material_layers();
        cur = (cur & ~(0xFFu << (layer * 8u))) | (static_cast<uint32_t>(idx) << (layer * 8u));
        pipeline.setShapeMaterial(id, cur);
    }
}

void VATMesh::set_layer(core::Pipeline& pipeline, uint topology_index, uint layer, uint8_t idx) noexcept {
    if (topology_index >= _shape_ids.size()) return;
    auto* shape = pipeline.getShape(_shape_ids[topology_index]);
    if (!shape) return;
    uint32_t cur = shape->material_layers();
    cur = (cur & ~(0xFFu << (layer * 8u))) | (static_cast<uint32_t>(idx) << (layer * 8u));
    pipeline.setShapeMaterial(_shape_ids[topology_index], cur);
}

void VATMesh::set_layers(core::Pipeline& pipeline, uint8_t base, uint8_t l1, uint8_t l2, uint8_t l3) noexcept {
    uint32_t layers = static_cast<uint32_t>(base)
                    | (static_cast<uint32_t>(l1) << 8u)
                    | (static_cast<uint32_t>(l2) << 16u)
                    | (static_cast<uint32_t>(l3) << 24u);
    for (auto id : _shape_ids)
        pipeline.setShapeMaterial(id, layers);
}

void VATMesh::set_layers(core::Pipeline& pipeline, uint topology_index,
                         uint8_t base, uint8_t l1, uint8_t l2, uint8_t l3) noexcept {
    if (topology_index >= _shape_ids.size()) return;
    uint32_t layers = static_cast<uint32_t>(base)
                    | (static_cast<uint32_t>(l1) << 8u)
                    | (static_cast<uint32_t>(l2) << 16u)
                    | (static_cast<uint32_t>(l3) << 24u);
    pipeline.setShapeMaterial(_shape_ids[topology_index], layers);
}

void VATMesh::apply_transform(core::Pipeline& pipeline) noexcept {
    auto mat = _transform.matrix();
    auto change = _transform.change();
    for (auto id : _shape_ids)
        pipeline.setShapeTransform(id, mat, change);
}

void VATMesh::register_name(std::unordered_map<std::string, ShapeId>& nameMap,
                              const std::string& name) noexcept {
    if (!_shape_ids.empty())
        nameMap[name] = _shape_ids[_active];
}

//==============================================================================
// UI
//==============================================================================

void VATMesh::drawUi() noexcept {
    if (ImGui::CollapsingHeader(_name.c_str())) {
        std::string scp = "vat_" + _name;
        ImGui::ScopedId scpId(scp.c_str());

        ImGui::Checkbox("Play Aniamtion", &_playing);

        float tf = static_cast<float>(total_frames());
        float dur = total_duration();
        float cf = current_frame();

        // Summary line
        ImGui::Text("Topo %u/%u  |  %.0f frames  |  %.2fs  |  frame %.1f",
            _active, topology_count(), tf, dur, cf);

        ImGui::DragFloat("Speed", &_speed, .1f, .1f, 4.f, "%.1f");
        ImGui::DragFloat("FPS", &_fps, 1.f, 1.f, 60.f, "%.1f");

        // Play mode combo
        const char* mode_names[] = { "Clamp", "Loop", "PingPong" };
        int mode_idx = static_cast<int>(_play_mode);
        if (ImGui::Combo("Play Mode", &mode_idx, mode_names, 3))
            _play_mode = static_cast<VATPlayMode>(mode_idx);

        // Frame scrubber
        ImGui::SliderFloat("Frame", &_frame, 0.f, tf, "%.1f");
        if (ImGui::Button("Reset")) reset();

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Sequences")) {
            uint32_t idx = 0;
            for (auto& seq : _sequences) {
                ImGui::ScopedId seqId(idx);
                uint32_t end_frm = (idx < _sequences.size() - 1u)
                    ? _sequences[idx + 1u].frame_offset : tf;
                ImGui::Text("Seq %u: frames %u-%u  (verts:%u  tris:%u)",
                    idx, seq.frame_offset, end_frm,
                    seq.vertex_count, seq.index_count / 3u);
                idx++;
            }
        }

        ImGui::Checkbox("Update Tangent", &_update_tangent);
    }
}

} // namespace newtype::scene
