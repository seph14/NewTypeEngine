#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/scene/Geometry.h"
#include <unordered_map>

namespace newtype { namespace core { class Pipeline; } }

namespace newtype::scene {

using namespace luisa;
using compute::Buffer;
using compute::Device;
using compute::Stream;

class VATMesh;
typedef luisa::unique_ptr<VATMesh> VATMeshPtr;

/// Animation playback mode
enum class VATPlayMode : uint8_t {
    Clamp    = 0,  // Hold last frame
    Loop     = 1,  // Wrap around
    PingPong = 2,  // Reverse at end, then reverse at start
};

/// Per-topology data loaded from a single .vat file
struct VATSequence {
    uint32_t vertex_count = 0;
    uint32_t frame_count = 0;
    uint32_t index_count = 0;
    uint32_t frame_offset = 0;     // cumulative frame offset in global timeline

    // CPU data (kept for DeformableMesh init + tangent computation)
    luisa::vector<uint32_t>           indices;
    luisa::vector<float>              texcoords;     // flattened vec2
    luisa::vector<luisa::float3>      positions;     // all frames (frame_count * vertex_count)
    luisa::vector<luisa::float3>      normals;       // all frames (frame_count * vertex_count)
    luisa::vector<util::Vertex>       initial_verts; // first frame with tangent + UV

    // GPU frame data (read-only animation source)
    Buffer<luisa::float3> positions_gpu;  // frame_count * vertex_count
    Buffer<luisa::float3> normals_gpu;    // frame_count * vertex_count
};

/**
 * @brief Vertex-Animation-Texture mesh with GPU frame interpolation
 *
 * Loads VAT binary files and plays back baked vertex animation
 * with GPU-based linear interpolation between consecutive frames.
 *
 * Each .vat file represents one topology (unique index/vertex layout).
 * Multiple topologies are handled by owning one DeformableMesh per
 * topology, with visibility toggling for topology changes.
 *
 * The interpolation shader is shared across all VATMesh instances
 * via ShaderManager ("vat_interpolate").
 *
 * Usage:
 *   auto vat = VATMesh::create(Renderer::device());
 *   vat->load_folder(app::getAssetPath("models/character"), "character");
 *   vat->build(*pipeline, Renderer::stream(), materialId);
 *
 *   // Per frame (before Pipeline::update):
 *   vat->update(*pipeline, Renderer::stream(), time);
 *   pipeline->update(time);
 */
class VATMesh {
public:
    using Vertex = util::Vertex;

private:
    Device& _device;

    // Per-topology data
    luisa::vector<VATSequence> _sequences;

    // One DeformableMesh per topology, registered with Pipeline
    luisa::vector<ShapeId> _shape_ids;
    uint _active = 0;
    bool _built  = false;
    bool _visible = true;
    scene::StaticTransform _transform;

    // Animation state
    float _frame         = 0.0f;
    float _fps           = 24.0f;
    float _speed         = 1.0f;
    bool _update_tangent = false;
    bool _playing        = false;
    bool _double_sided   = false;
    std::string _name;
    VATPlayMode _play_mode = VATPlayMode::Clamp;

public:
    explicit VATMesh(Device& device) noexcept : _device(device) {}
    ~VATMesh() = default;

    static VATMeshPtr create(Device& device) noexcept {
        return luisa::make_unique<VATMesh>(device);
    }

    /// Load all .vat files from a folder (model0.vat, model1.vat, ...)
    bool load_folder(const std::filesystem::path& folder_path,
                     const std::string& base_name = "model") noexcept;

    /// Load a single .vat file as one topology
    bool load_vat(const std::filesystem::path& file_path) noexcept;

    /// Create DeformableMesh instances, compute tangents, register with Pipeline.
    /// Registers shared interpolation shader via ShaderManager (once globally).
    void build(newtype::core::Pipeline& pipeline, Stream& stream, uint material_id = 0) noexcept;

    /// Per-frame: interpolate vertices on compute stream, handle topology switch.
    /// Call BEFORE Pipeline::update().
    void update(newtype::core::Pipeline& pipeline, float dt) noexcept;

    /// Reset animation to start
    void reset() noexcept;

    /// Draw ImGui controls
    void drawUi() noexcept;

    // --- Accessors ---
    [[nodiscard]] uint active_topology() const noexcept { return _active; }
    [[nodiscard]] float speed() const noexcept { return _speed; }
    [[nodiscard]] float fps() const noexcept { return _fps; }
    [[nodiscard]] auto& transform() const noexcept { return _transform; }
    [[nodiscard]] VATPlayMode play_mode() const noexcept { return _play_mode; }
    [[nodiscard]] uint topology_count() const noexcept {
        return static_cast<uint>(_sequences.size());
    }
    /// Total frames across all topologies
    [[nodiscard]] uint32_t total_frames() const noexcept;
    /// Total animation duration in seconds (at speed=1)
    [[nodiscard]] float total_duration() const noexcept;
    /// Current frame (global)
    [[nodiscard]] float current_frame() const noexcept;
    [[nodiscard]] DeformableMesh* active_deformable(
        newtype::core::Pipeline& pipeline) const noexcept;

    void set_speed(float s) noexcept { _speed = s; }
    void set_fps(float fps) noexcept { _fps = fps; }
    void set_update_tangent(bool v) noexcept { _update_tangent = v; }
    void set_play_mode(VATPlayMode mode) noexcept { _play_mode = mode; }
    void set_playing(bool v) noexcept { _playing = v; }
    void set_double_sided(bool v) noexcept { _double_sided = v; }

    void set_position(const luisa::float3& pos) noexcept { _transform.set_position(pos); }
    void set_rotation (const luisa::float4& rot) noexcept { _transform.set_rotation(rot); }
    void set_scale(const luisa::float3& scale) noexcept { _transform.set_scale(scale); }
    void set_transform(const luisa::float4x4& trs) noexcept { _transform.set_matrix(trs); }

    // --- Shape ID Access ---
    [[nodiscard]] ShapeId shape_id() const noexcept;
    [[nodiscard]] ShapeId shape_id(uint topology_index) const noexcept;
    [[nodiscard]] const luisa::vector<ShapeId>& shape_ids() const noexcept;

    // --- Visibility ---
    void set_visible(core::Pipeline& pipeline, bool visible) noexcept;
    [[nodiscard]] bool is_visible() const noexcept;

    // --- Material ---
    void set_material(core::Pipeline& pipeline, uint material_id) noexcept;
    void set_material(core::Pipeline& pipeline, uint topology_index, uint material_id) noexcept;

    /// Set full 4-layer packed material on all topologies
    void set_material_layers(core::Pipeline& pipeline, uint32_t layers) noexcept;
    /// Set full 4-layer packed material on a specific topology
    void set_material_layers(core::Pipeline& pipeline, uint topology_index, uint32_t layers) noexcept;
    /// Set material for a specific layer (0-3) on all topologies
    void set_layer(core::Pipeline& pipeline, uint layer, uint8_t idx) noexcept;
    /// Set material for a specific layer (0-3) on a specific topology
    void set_layer(core::Pipeline& pipeline, uint topology_index, uint layer, uint8_t idx) noexcept;
    /// Set multiple layers at once on all topologies (inactive layers default to 0xFF)
    void set_layers(core::Pipeline& pipeline, uint8_t base, uint8_t l1 = 0xFF,
                    uint8_t l2 = 0xFF, uint8_t l3 = 0xFF) noexcept;
    /// Set multiple layers at once on a specific topology
    void set_layers(core::Pipeline& pipeline, uint topology_index,
                    uint8_t base, uint8_t l1 = 0xFF, uint8_t l2 = 0xFF, uint8_t l3 = 0xFF) noexcept;

    // --- Transform sync ---
    void apply_transform(core::Pipeline& pipeline) noexcept;

    // --- Name registration for timeline ---
    void register_name(std::unordered_map<std::string, ShapeId>& nameMap,
                       const std::string& name) noexcept;

private:
    /// Compute tangents from first frame positions + normals + texcoords + indices
    void _compute_tangents(VATSequence& seq) noexcept;

    /// Register the interpolation shader with ShaderManager (if not already)
    static void _ensure_shader_registered(Device& device) noexcept;

    /// Determine active topology and fractional frame from current _frame
    bool _compute_frame(uint& out_topo_idx,
                        float& out_local_frame) const noexcept;
};

} // namespace newtype::scene
