#pragma once

//==============================================================================
// TetCageGeometry — runtime half of the tetrahedral-cage pipeline
// (docs/tetrahedral-cage-prototype.md, Stage 2)
//
// A .tetcage file (built offline by tools/tet_cage) holds a rest-pose mesh
// dissected into per-tet "piece" micro-meshes plus the lattice ("cage") the
// pieces were cut from. This class registers every piece as a MeshShape
// prototype — one static, immutable BLAS, shared by every copy — and creates
// one TLAS instance per tet per copy.
//
// Per frame a user-supplied deform shader (a core::ShaderManager id, see
// set_deform_shader_id) animates the cage vertices on the GPU, and the
// engine derives each tet's rest→animated affine map from its 4 cage
// corners on the DEVICE (TetSolve kernel — no readback):
//
//     M = [a1-a0, a2-a0, a3-a0] · [r1-r0, r2-r0, r3-r0]⁻¹,  t = a0 - M·r0
//
// The kernel writes world = copyWorld · [M|t] straight into the engine
// instance-transform buffer; the TLAS copies the registered rows from that
// buffer during its build (LC fork Accel::set_transform_buffer_on_update),
// so a cage frame is [user wind kernel → TetSolve → TLAS PREFER_UPDATE
// refit], all on the GPU stream. CPU cost per frame is O(dispatch).
// NT_TETCAGE_CPU_PATH=1 forces the original path (wind readback + CPU
// solve + per-instance setShapeTransform) as the validation baseline; row
// registration failures (non-contiguous TLAS rows) fall back to it too.
//
// Neighboring tets share 3 cage verts, so their affine maps agree exactly
// on the shared face — the animated surface stays watertight no matter
// what the deform shader writes. The TLAS refit is the only per-frame
// GPU cost; pieces are never touched.
//
// Usage:
//   auto cage = scene::TetCageGeometry::create(Renderer::device());
//   cage->load(app::getAssetPath("models/ginkgo/ginkgo0.tetcage"));
//   cage->add_copy(worldTransform);           // any number of copies
//   cage->set_deform_shader_id("my_cage_deform", 256);
//   cage->build(*pipeline, Renderer::stream(), materialId);
//
//   // Per frame, BEFORE Pipeline::update():
//   cage->set_deform_state(0, state);         // params[0] = (time, strength, ...)
//   cage->update(*pipeline);
//   pipeline->update(time, dt);
//==============================================================================

#include <luisa/luisa-compute.h>
#include <filesystem>

#include "newtype/scene/Geometry.h"
#include "newtype/core/ShaderManager.h"

namespace newtype::core { class Pipeline; }

// Forward declare struct at global scope for LUISA_STRUCT
namespace newtype::scene {
struct TetCageDeformState;
struct TetCageSolveRow;
struct TetCageRestFrame;
class TetCageGeometry;
using TetCagePtr = luisa::unique_ptr<TetCageGeometry>;
}

/// Per-copy deformation state for the cage deform shader. Opaque to the
/// engine — the shader defines the semantics (the wind sample reads
/// params[0] as (time, strength, freq, unused)).
struct newtype::scene::TetCageDeformState {
    luisa::float4 params[4]; // 64 bytes generic user data
};
LUISA_STRUCT(newtype::scene::TetCageDeformState, params) {};

/// One GPU solve thread per (copy, tet) pair: reads the copy's animated cage
/// verts, derives the tet's affine, and writes the composed world matrix
/// into the engine instance-transform buffers at `tlas_row` (curr+prev).
struct newtype::scene::TetCageSolveRow {
    uint32_t cage_verts[4] = {}; // indices into the copy's anim-cage slice
    uint32_t anim_base = 0u;     // copy × cageVertexCount
    uint32_t rest_idx = 0u;      // index into the rest-frame table
    uint32_t copy_idx = 0u;      // index into the copy-world buffer
    uint32_t tlas_row = 0u;      // engine instance-transform row
};
LUISA_STRUCT(newtype::scene::TetCageSolveRow,
             cage_verts, anim_base, rest_idx, copy_idx, tlas_row) {};

/// Copy-independent half of the affine solve (precomputed by load()).
struct newtype::scene::TetCageRestFrame {
    luisa::float3 r0;        // rest cage vert 0
    luisa::float3 rin[3];    // columns of [r1-r0, r2-r0, r3-r0]⁻¹
};
LUISA_STRUCT(newtype::scene::TetCageRestFrame, r0, rin) {};

namespace newtype::scene {

class TetCageGeometry {
public:
    using Vertex = util::Vertex;

    /// Per-tet cage record, mirroring the .tetcage table (face f is opposite
    /// cage vertex f).
    struct Tet {
        uint32_t cage_verts[4] = {};
        int32_t neighbors[4] = {};      // tet id across face f, -1 = boundary
        uint32_t vert_start = 0, vert_count = 0;
        uint32_t tri_start = 0, tri_count = 0;

        // Precomputed rest-frame affine data (filled by load(), not on disk):
        // r0 = rest cage vert 0, rinv[j] = column j of [r1-r0, r2-r0, r3-r0]⁻¹.
        luisa::float3 r0 = luisa::make_float3(0.f);
        luisa::float3 rinv[3] = {
            luisa::make_float3(1.f, 0.f, 0.f),
            luisa::make_float3(0.f, 1.f, 0.f),
            luisa::make_float3(0.f, 0.f, 1.f)};
    };

    /// Below this |det| an animated tet is considered collapsed; its previous
    /// instance matrix is kept for the frame instead (plan's degenerate guard).
    static constexpr float kMinDet = 1e-9f;

public:
    explicit TetCageGeometry(Device &device) noexcept : _device(device) {}
    ~TetCageGeometry() = default;

    static TetCagePtr create(Device &device) noexcept {
        return luisa::make_unique<TetCageGeometry>(device);
    }

    /// Parse a .tetcage v1 file. Logs and returns false on any mismatch.
    bool load(const std::filesystem::path &path) noexcept;

    /// Queue an animated copy (call before build()). The world transform is
    /// composed in front of every per-tet affine. Returns the copy index.
    uint add_copy(const luisa::float4x4 &world) noexcept;

    /// Set the cage-vertex deform shader: an id previously registered with
    /// core::ShaderManager. Call before build(). Without an id the copies
    /// render the rest pose (matrices flushed once, then idle).
    ///
    /// Cage deform shader definition — register a 2-D compute shader with
    /// EXACTLY these 3 buffer parameters (order matters, it is the dispatch
    /// order):
    ///
    ///     core::ShaderManager::instance().registerShader<2>(shaderId,
    ///         [vertCount](compute::BufferVar<luisa::float3> anim_cage,      // rw: animated cage verts
    ///                     compute::BufferVar<luisa::float3> rest_cage,      // r:  rest-pose cage verts
    ///                     compute::BufferVar<TetCageDeformState> states) {  // r:  per-copy state
    ///             UInt copy = dispatch_id().x;
    ///             Var<TetCageDeformState> st = states.read(copy);
    ///             $for(v, vertCount) {                 // partition by dispatch_id().y
    ///                 Float3 r = rest_cage.read(v);
    ///                 anim_cage.write(copy * vertCount + v, /* deformed r */);
    ///             };
    ///         });
    ///
        /// Dispatch: .dispatch(copyCount, blockSize) — x = copy index, y is free
        /// for the shader (grid-stride over the cage verts; each y lane must write
        /// disjoint verts). The TetSolve kernel consumes anim_cage on the device
        /// and derives every tet's affine from its 4 cage corners, so the
        /// animated surface stays watertight regardless of the deformation.
        /// vertCount == cage_vertex_count() — capture it after load(), before
        /// registration.
    void set_deform_shader_id(luisa::string_view shaderId, uint blockSize) noexcept;

    /// Set one copy's deform state (consumed by the deform shader; opaque to
    /// the engine — the wind sample reads params[0] as (time, strength, freq,
    /// unused)). Pre-build calls seed the initial state; post-build calls
    /// upload the row on the next update().
    void set_deform_state(uint copy, const TetCageDeformState &state) noexcept;

    /// Set every copy's deform state to the same value. Post-build this is
    /// the O(1)-command path for the animated case: one whole-buffer upload
    /// on the next update() instead of one copy_from per row (per-row
    /// commands cost O(copies) CPU per frame — the uniform-wind scene case).
    void set_deform_state_all(const TetCageDeformState &state) noexcept;

    /// Bake the pose defined by the current deform states into the INITIAL
    /// instance matrices (call after the states + shader id, before build()).
    /// The deformed pose then ships with the first instance-buffer upload and
    /// no per-frame transform writes happen — used to separate
    /// affine/serialization errors from upload-churn artifacts
    /// (docs/tlas_wrong_blas_binding.md). Requires a deform shader id;
    /// otherwise the rest pose is baked.
    void bake_initial_pose() noexcept;

    /// Register all pieces as prototypes and create one TLAS instance per tet
    /// per copy. Must run before Pipeline::buildScene().
    void build(core::Pipeline &pipeline, Stream &stream,
               uint material_id, bool double_sided = true) noexcept;

    /// Per-frame deform: dispatch the cage deform shader, derive the per-tet
    /// affines and flush the instance transforms. Call before
    /// Pipeline::update(). Without a shader id this flushes the rest
    /// matrices once, then costs nothing.
    ///
    /// GPU path (default): wind shader + a TetSolve kernel run back to back
    /// on the compute stream; the matrices land in the engine instance-
    /// transform buffer and are copied into the TLAS instance rows by the
    /// device (LC fork API) — no readback, no CPU solve, no per-instance
    /// setShapeTransform. CPU cost per frame is O(dispatch), enabling 10⁵+
    /// animated cage instances. Falls back to the original readback path
    /// when row registration fails (non-contiguous TLAS rows) or when
    /// NT_TETCAGE_CPU_PATH=1 is set (A/B validation baseline).
    void update(core::Pipeline &pipeline) noexcept;

    /// Converge prev ← curr for the GPU-owned rows (one TetSolve dispatch on
    /// the unchanged cage) so motion vectors return to zero after the
    /// animation stops. No-op on the CPU path (Geometry's stop-frame prev
    /// fix covers it) and when idle. Call on the first frame the cage stops
    /// animating (frozen/settled).
    void settle(core::Pipeline &pipeline) noexcept;

    //==========================================================================
    // Procedural comparison reference — REMOVED. TetCageGeometry runs entirely
    // in the TLAS line; A/B comparison against the type-3 procedural deform
    // path is scene-side wiring now (see src/tests/TetCageScene.cpp).
    //==========================================================================

    /// Toggle one copy's instances (all its tets).
    void set_copy_visible(core::Pipeline &pipeline, uint copy, bool visible) noexcept;
    /// Toggle every copy.
    void set_visible(core::Pipeline &pipeline, bool visible) noexcept;
    /// Set the packed 4-layer material on every instance.
    void set_material_layers(core::Pipeline &pipeline, uint32_t layers) noexcept;

    // --- Accessors ---
    [[nodiscard]] bool loaded() const noexcept { return !_tets.empty(); }
    [[nodiscard]] bool built() const noexcept { return _built; }
    [[nodiscard]] uint copy_count() const noexcept {
        return static_cast<uint>(_copyWorld.size());
    }
    [[nodiscard]] uint tet_count() const noexcept {
        return static_cast<uint>(_tets.size());
    }
    /// Tets that produced a piece (non-empty, non-degenerate).
    [[nodiscard]] uint piece_count() const noexcept { return _pieceCount; }
    /// Total piece triangles across the soup (per-copy surface = same count).
    [[nodiscard]] uint piece_triangle_count() const noexcept {
        return static_cast<uint>(_pieceTris.size());
    }
    [[nodiscard]] uint cage_vertex_count() const noexcept {
        return static_cast<uint>(_restCage.size());
    }
    /// Total TLAS instances = copies × pieces.
    [[nodiscard]] uint instance_count() const noexcept {
        return static_cast<uint>(_instanceIds.size());
    }
    /// Rest-pose bounds of the actual piece geometry (not the voxel grid).
    [[nodiscard]] luisa::float3 rest_min() const noexcept { return _restMin; }
    [[nodiscard]] luisa::float3 rest_max() const noexcept { return _restMax; }
    /// Wall-clock CPU time of the last update(). GPU path: dispatch-issue
    /// cost only (wind + solve kernels, no readback); CPU path: dispatch +
    /// readback + cage solve + matrix flush.
    [[nodiscard]] float last_update_ms() const noexcept { return _lastUpdateMs; }
    /// True when the GPU solve path owns this frame's transform writes.
    [[nodiscard]] bool gpu_path_active() const noexcept {
        return _built && _gpuPathActive;
    }

private:
    Device &_device;
    bool _built = false;
    uint _pieceCount = 0u;

    // Cage + piece soup (rest pose)
    luisa::vector<luisa::float3> _restCage;
    luisa::vector<Tet> _tets;
    luisa::vector<Vertex> _pieceVerts;          // global soup; Tet holds per-piece ranges
    luisa::vector<compute::Triangle> _pieceTris;
    luisa::float3 _restMin = luisa::make_float3(0.f);
    luisa::float3 _restMax = luisa::make_float3(0.f);

    // Copies
    luisa::vector<luisa::float4x4> _copyWorld;
    luisa::vector<ShapeId> _instanceIds;        // copy × tetCount + tet; kInvalidShapeId for skipped pieces

    // Deform shader + per-copy state (ShaderManager handle; unset = rest pose)
    core::ShaderHandle<2,
        compute::Buffer<luisa::float3>,
        compute::Buffer<luisa::float3>,
        compute::Buffer<TetCageDeformState>> _deformShader;
    uint _deformShaderBlock = 256u;
    bool _bakeInitialPose = false;              // seed initial matrices with the deformed pose
    luisa::vector<TetCageDeformState> _cageStateCPU;
    luisa::vector<uint> _dirtyCageStates;
    bool _allStatesDirty = false;               // whole-buffer upload pending (set_deform_state_all)
    compute::Buffer<luisa::float3> _restCageBuffer;          // shader input
    compute::Buffer<luisa::float3> _animCageBuffer;          // shader output, [copies × cageVertCount]
    compute::Buffer<TetCageDeformState> _cageStateBuffer;

    // GPU solve path (docs/TLAS-instance-transform-updates.md Part 3): one
    // thread per solve row composes the per-tet affine on the device and
    // writes the engine instance-transform buffers directly; Geometry hands
    // the TLAS build a device-side copy source for those rows.
    bool _gpuPathWanted = false;   // registered at build (or CPU path forced)
    bool _gpuPathActive = false;   // registration alive; solve rows resolved
    uint _gpuTopologyGeneration = ~0u;
    luisa::vector<TetCageSolveRow> _solveRowsCPU;
    luisa::vector<TetCageRestFrame> _restFramesCPU;
    compute::Buffer<TetCageSolveRow> _solveRowBuffer;
    compute::Buffer<TetCageRestFrame> _restFrameBuffer;
    compute::Buffer<luisa::float4x4> _copyWorldBuffer;
    compute::Shader<1,
        compute::Buffer<luisa::float4x4>,  // rw: instance transforms (curr)
        compute::Buffer<luisa::float4x4>,  // rw: instance transforms (prev)
        compute::Buffer<TetCageSolveRow>,  // r:  solve rows
        compute::Buffer<TetCageRestFrame>, // r:  per-tet rest frames
        compute::Buffer<luisa::float3>,    // r:  animated cage verts
        compute::Buffer<luisa::float4x4>   // r:  per-copy world matrices
    > _tetSolveShader;

    /// (Re-)resolve solve rows and register them with the pipeline's
    /// Geometry. False when the rows are not one contiguous TLAS run (the
    /// caller falls back to the CPU path) — also rebuilds the row table
    /// after a topology reshuffle. `stream` carries the row-table upload.
    bool register_solve_rows(core::Pipeline &pipeline, Stream &stream) noexcept;

    /// Dispatch the TetSolve kernel (curr/prev write) on the current
    /// instance-transform buffers and mark the GPU rows dirty.
    void dispatch_tet_solve(core::Pipeline &pipeline) noexcept;

    // Scratch (reused across frames): host copy of _animCageBuffer
    luisa::vector<luisa::float3> _animCage;

    /// Dispatch the deform shader + read the animated cage back into
    /// _animCage. Synchronizes the stream — the CPU affine solve consumes the
    /// verts immediately. No-op without a shader id.
    void dispatch_deform(Stream &stream) noexcept;

    float _lastUpdateMs = 0.f;
};

} // namespace newtype::scene
