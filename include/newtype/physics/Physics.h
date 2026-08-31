#pragma once

// nt::physics::Physics — tighter LCS (LuisaCompute Solver) integration.
//
// Replaces the old PhysicsBridge demo. One Physics instance owns a single
// LCS Newton solver and exposes a per-body API that returns a DeformableMesh*
// the caller registers with the scene via Pipeline::addShape.
//
// Lifecycle: construct -> add_body()* -> prepare() -> step()*. Bodies cannot
// be added after prepare() (LCS would need a full re-init); call restart()
// for scene transitions.
//
// Stream/sync model: Physics owns a dedicated luisa::compute::Stream so
// LCS commits don't stall Renderer::stream() or Pipeline::computeStream().
// Cross-stream sync with the compute stream is via a TimelineEvent signaled
// at the end of step(); callers MUST invoke wait_for_step() on
// pipeline.computeStream() BEFORE Pipeline::update() so Geometry::update
// sees completed vertex writes when it rebuilds deformable BLAS.
//
// Threading: prepare() and step() MUST run on the same thread that
// constructed Pipeline — LCS relies on a bound luisa::fiber::scheduler
// (Pipeline::_fiberScheduler) for parallel kernel JIT in init_solver().
// Calling from another thread silently serializes that compile.

#include <luisa/luisa-compute.h>
#include <array>
#include <cstdint>
#include <string>
#include "cinder/TriMesh.h"
#include "cinder/GeomIo.h"

namespace newtype::scene { class DeformableMesh; }

namespace newtype::physics {

// v1 only routes to NewtonSolver (the only LCS backend). Enum is in place so
// the API stays stable when XPBD/VBD land in LCS.
enum class SolverType : uint8_t { Newton };

// Mirrored from lcs::Material to keep LCS headers out of the public API.
enum class ClothStretchModel : uint8_t { Spring, FEM_BW98 };
enum class ClothBendingModel  : uint8_t { QuadraticBending, DihedralAngle };

// Tetrahedral (soft body) constitutive models. Mirrors
// lcs::Material::ConstitutiveModelTet.
enum class TetModel : uint8_t { Spring, StVK, StableNeoHookean, Corotated, ARAP };

// Rigid (Affine Body Dynamics) constitutive models. Mirrors
// lcs::Material::ConstitutiveModelRigid.
enum class RigidModel : uint8_t { Spring, Orthogonality, ARAP, StableNeoHookean };

// Mirrored subset of lcs::Initializer::FixedPointsType (positional pins).
// Use FixedPinMethod::FromIndices with ClothBodyConfig::pin_indices for
// explicit vertex indices.
enum class FixedPinMethod : uint8_t {
    Left, Right, Front, Back, Up, Down,
    LeftUp, LeftDown, LeftFront, LeftBack,
    RightUp, RightDown, RightFront, RightBack,
    FrontUp, FrontDown, BackUp, BackDown,
    All,
};

struct ClothBodyConfig {
    ClothStretchModel stretch_model             = ClothStretchModel::FEM_BW98;
    ClothBendingModel bending_model             = ClothBendingModel::QuadraticBending;
    float    thickness                          = 1e-3f;
    float    youngs_modulus                     = 1e6f;
    float    poisson_ratio                      = 0.35f;
    float    area_bending_stiffness             = 5e-3f;

    // Placement in LCS world space. Should match the DeformableMesh's initial
    // world transform; Physics does not transform vertices itself.
    luisa::float3 translation                   {0.f, 0.f, 0.f};
    luisa::float3 rotation_euler                {0.f, 0.f, 0.f};   // radians, per-axis
    luisa::float3 scale                         {1.f, 1.f, 1.f};

    // Positional pin methods (Left/Right/Up/...). Applied via LCS's
    // add_fixed_point_from_method using the configured `pin_range`.
    luisa::vector<FixedPinMethod> pin_methods;
    float                         pin_range     = 0.001f;

    // Explicit vertex indices to pin (independent of pin_methods; both apply).
    luisa::vector<uint>           pin_indices;

    std::string name = "cloth";
};

// Volumetric soft body. Mesh is a tetrahedral mesh (4-vertex simplices);
// LCS extracts the surface triangle mesh internally for rendering and
// collision. Constitutive model picks the strain energy.
//
// youngs_modulus / poisson_ratio have the same meaning as for cloth but the
// strains are 3D (volume-preserving for StableNeoHookean, etc.). d_hat /
// friction_mu / contact_offset are IPC collision parameters (also used by
// cloth, but they default-hidden inside ClothBodyConfig because most cloth
// demos leave them at LCS defaults).
struct TetrahedralBodyConfig {
    TetModel model                                = TetModel::StVK;
    float    youngs_modulus                       = 1e6f;
    float    poisson_ratio                        = 0.35f;

    // IPC / shared MaterialBase knobs. d_hat is the contact distance scale;
    // friction_mu is the Coulomb friction coefficient.
    float    mass                                 = 0.f;     // 0 = density * volume
    float    density                              = 1e3f;
    float    d_hat                                = 1e-3f;
    float    contact_offset                       = 0.f;
    float    friction_mu                          = 0.5f;

    // Placement in LCS world space. Should match the DeformableMesh's initial
    // world transform; Physics does not transform vertices itself.
    luisa::float3 translation                     {0.f, 0.f, 0.f};
    luisa::float3 rotation_euler                  {0.f, 0.f, 0.f};
    luisa::float3 scale                           {1.f, 1.f, 1.f};

    // Same positional-pin semantics as cloth: pin_methods picks verts by
    // bounding-box face/edge/corner; pin_indices adds explicit verts.
    luisa::vector<FixedPinMethod> pin_methods;
    float                         pin_range       = 0.001f;
    luisa::vector<uint>           pin_indices;

    std::string name = "tet";
};

// Reduced-coordinate rigid body (Affine Body Dynamics). LCS tracks a per-body
// affine transform (translation + rotation + scale) and writes the
// transformed positions back into the sa_x buffer that get_curr_positions_device
// exposes — so the NT-side update path is the same as cloth/tet (per-vertex
// copy into next_vertex_buffer). The constitutive model is the ABD strain
// energy; Orthogonality is the classical ABD choice.
//
// is_solid=true treats the mesh as a closed solid (uses mesh volume for mass
// and rotational inertia); is_solid=false treats it as a thin shell.
struct RigidBodyConfig {
    RigidModel model                               = RigidModel::Orthogonality;
    bool       is_solid                            = true;
    float      thickness                           = 1e-3f;   // used when is_solid=false
    float      stiffness                           = 1e6f;

    // IPC / shared MaterialBase knobs.
    float      mass                                = 0.f;
    float      density                             = 1e3f;
    float      d_hat                               = 1e-3f;
    float      contact_offset                      = 0.f;
    float      friction_mu                         = 0.5f;

    // Initial placement in LCS world space. Subsequent motion comes from the
    // solver; the only way to "drive" a rigid body is via LCS's per-body
    // animation API, which NT does not currently wrap.
    luisa::float3 translation                      {0.f, 0.f, 0.f};
    luisa::float3 rotation_euler                   {0.f, 0.f, 0.f};
    luisa::float3 scale                            {1.f, 1.f, 1.f};

    std::string name = "rigid";
};

class Physics {
public:
    Physics();   // borrows Renderer::device(); creates dedicated Stream + TimelineEvent
    ~Physics();

    Physics(const Physics&)            = delete;
    Physics& operator=(const Physics&) = delete;

    // --- Body registration (call BEFORE prepare()) ----------------------
    // Returns a DeformableMesh* the caller registers with pipeline->addShape.
    // Physics creates the mesh (caller takes ownership via addShape) and
    // caches a non-owning pointer for per-frame updates.

    // From ci::geom::Source (Teapot, Sphere, etc.)
    [[nodiscard]] scene::DeformableMesh* add_body(
        const ci::geom::Source& geom,
        const ClothBodyConfig&  cfg,
        uint                    material_id);

    // From ci::TriMesh (caller already has the topology).
    [[nodiscard]] scene::DeformableMesh* add_body(
        const ci::TriMesh&      trimesh,
        const ClothBodyConfig&  cfg,
        uint                    material_id);

    // Adopt an existing DeformableMesh. The mesh MUST have CPU data resident
    // (has_cpu_data() == true) since LCS needs the topology for registration.
    // Returns the same pointer.
    [[nodiscard]] scene::DeformableMesh* add_body(
        scene::DeformableMesh*  existing,
        const ClothBodyConfig&  cfg);

    // --- Tetrahedral (soft body) bodies ---------------------------------
    // Tet topology is provided as 4-vertex simplices; LCS extracts the surface
    // triangle mesh internally. The returned DeformableMesh's vertex buffer
    // covers ALL tet verts (interior included) so the per-frame LCS sync stays
    // a single contiguous copy — interior verts are simply unreferenced by
    // surface faces and never rendered. Normals/UVs/tangents are auto-derived
    // by MeshShape::load_from on the NT side (UVs default to (0,0)).

    // Tet mesh from explicit arrays (positions + 4-index tets).
    [[nodiscard]] scene::DeformableMesh* add_body(
        const TetrahedralBodyConfig&            cfg,
        uint                                    material_id,
        luisa::span<const std::array<float, 3>> tet_vertices,
        luisa::span<const std::array<uint, 4>>  tet_elements);

    // Tet mesh from file (.t or .vtk — LCS handles IO).
    [[nodiscard]] scene::DeformableMesh* add_body(
        const TetrahedralBodyConfig&            cfg,
        uint                                    material_id,
        const std::string&                      tet_mesh_path);

    // --- Rigid (ABD) bodies ---------------------------------------------
    // Surface triangle mesh only. LCS tracks the affine state; per-frame
    // positions are written back into the LCS sa_x buffer that
    // get_curr_positions_device exposes, so the existing copy_positions
    // kernel handles rigid bodies unchanged.
    [[nodiscard]] scene::DeformableMesh* add_body(
        const ci::geom::Source&   geom,
        const RigidBodyConfig&    cfg,
        uint                      material_id);

    [[nodiscard]] scene::DeformableMesh* add_body(
        const ci::TriMesh&        trimesh,
        const RigidBodyConfig&    cfg,
        uint                      material_id);

    [[nodiscard]] scene::DeformableMesh* add_body(
        scene::DeformableMesh*    existing,
        const RigidBodyConfig&    cfg);

    // --- Lifecycle ------------------------------------------------------
    // Compiles LCS kernels, allocates GPU buffers, sorts world_data.
    // Multi-second JIT compile — call once at scene load, NOT per frame.
    void prepare();

    // Returns to the un-prepared state; clears all registered bodies.
    // Use for scene transitions. (Per-body removal is not supported by LCS.)
    void restart() noexcept;

    [[nodiscard]] bool prepared()   const noexcept;
    [[nodiscard]] uint body_count() const noexcept;

    // --- Per-frame ------------------------------------------------------
    // Advance LCS by one step; copy positions into each body's
    // next_vertex_buffer on the physics stream. Marks all bodies dirty.
    //
    // NOTE: step() blocks the host for the duration of physics_step_GPU;
    // LCS calls stream << synchronize() internally. async_readback only
    // eliminates the CPU vertex re-upload, not the solve itself.
    void step();

    // Insert a wait for the most recent step() on a downstream stream.
    // MUST be called on pipeline.computeStream() BEFORE Pipeline::update
    // if any body is registered, so Geometry::update sees completed vertex
    // writes when it rebuilds deformable BLAS.
    void wait_for_step(luisa::compute::Stream& downstream) const;

    // --- Tunables / debug ----------------------------------------------
    void set_async_readback(bool enable) noexcept;   // default: true
    [[nodiscard]] bool async_readback() const noexcept;
    void dump_profile() const;                        // LCS Profiler::print_tree
    void drawUi();                                    // ImGui panel (caller hosts window)

private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;
};

} // namespace newtype::physics
