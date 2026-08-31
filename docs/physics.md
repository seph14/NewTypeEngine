# Physics — Cloth, Soft Body, Rigid Body

LuisaComputeSimulator (LCS) wrapper for GPU physics. One `physics::Physics` instance owns a single LCS Newton solver and exposes per-body registration that returns a `DeformableMesh*` for the rendering pipeline. Three body types are supported:

| Body | Config struct | LCS material | Models |
|---|---|---|---|
| **Cloth** | `ClothBodyConfig` | `ClothMaterial` | FEM_BW98 / Spring stretch + Quadratic / Dihedral bending |
| **Soft body** | `TetrahedralBodyConfig` | `TetMaterial` | Spring, StVK, StableNeoHookean, Corotated, ARAP |
| **Rigid (ABD)** | `RigidBodyConfig` | `RigidMaterial` | Spring, Orthogonality, ARAP, StableNeoHookean |

All three share the same lifecycle: construct → `add_body()` × N → `prepare()` → `step()` each frame. Bodies cannot be added after `prepare()`; call `restart()` for scene transitions.

---

## Quick Start — Cloth

```cpp
#include "newtype/physics/Physics.h"
#include "newtype/core/Pipeline.h"

using namespace newtype;
using namespace luisa;

// 1. Create Physics (uses Renderer::device(), owns a dedicated stream)
mPhysics = luisa::make_unique<physics::Physics>();

// 2. Register a material for the cloth
auto clothMatIdx = mPipeline->addMaterial(
    "physics_cloth", render::make_diffuse(luisa::float3(0.8f, 0.2f, 0.2f)));

// 3. Build cloth topology — any ci::geom::Source or ci::TriMesh works
ci::TriMesh clothTri(ci::geom::Plane()
                         .size(glm::vec2(1.f))
                         .subdivisions(glm::vec2(31.f, 31.f)));

// 4. Register as a cloth body
auto* clothMesh = mPhysics->add_body(
    clothTri,
    physics::ClothBodyConfig{
        .stretch_model          = physics::ClothStretchModel::FEM_BW98,
        .bending_model          = physics::ClothBendingModel::QuadraticBending,
        .thickness              = 0.001f,
        .youngs_modulus         = 1e5f,
        .poisson_ratio          = 0.3f,
        .translation            = luisa::float3(0.f, 3.8f, 0.25f),
        .pin_methods            = {physics::FixedPinMethod::LeftBack,
                                   physics::FixedPinMethod::RightBack},
        .name                   = "physics_cloth",
    },
    clothMatIdx);

// 5. Hand DeformableMesh to the scene — Pipeline takes ownership
scene::StaticTransform identity;
mPipeline->addShape(
    luisa::unique_ptr<scene::DeformableMesh>(clothMesh),
    &identity);

// 6. Compile LCS kernels (multi-second JIT — call once at scene load)
mPhysics->prepare();
```

---

## Quick Start — Soft Body (Tetrahedral)

Volumetric deformation. Tet topology is authoritative (4-vertex simplices); LCS extracts the surface triangle mesh internally. The returned `DeformableMesh`'s vertex buffer covers ALL tet verts (interior included) so per-frame LCS sync stays a single contiguous copy — interior verts are simply unreferenced by surface faces and never rendered.

### From explicit arrays

```cpp
// tet_vertices : std::vector<std::array<float, 3>> — tet mesh vertices
// tet_elements : std::vector<std::array<uint, 4>>  — 4-index tets

auto* squishy = mPhysics->add_body(
    physics::TetrahedralBodyConfig{
        .model          = physics::TetModel::StVK,
        .youngs_modulus = 5e5f,
        .poisson_ratio  = 0.35f,
        .density        = 1e3f,
        .translation    = luisa::float3(0.f, 2.f, 0.f),
        .name           = "squishy_ball",
    },
    squishyMatIdx,
    luisa::span{tet_vertices},
    luisa::span{tet_elements});
```

### From file (.t or .vtk — LCS handles IO)

```cpp
auto* squishy = mPhysics->add_body(
    physics::TetrahedralBodyConfig{
        .model          = physics::TetModel::StableNeoHookean,
        .youngs_modulus = 5e5f,
        .poisson_ratio  = 0.45f,
        .translation    = luisa::float3(0.f, 2.f, 0.f),
    },
    squishyMatIdx,
    "assets/tetmeshes/squishy_ball.vtk");
```

### Notes

- The tet mesh file format LCS reads is `.t` (try first) or `.vtk`. Both are common outputs from meshing tools like TetGen or Quartet.
- Tetrahedral bodies use the same positional-pin API as cloth (`pin_methods` / `pin_indices`) — useful for hanging a soft body from a fixed region.

---

## Quick Start — Rigid Body (Affine Body Dynamics)

Reduced-coordinate rigid bodies. Surface triangle mesh only; LCS reduces to affine DOF (translation + rotation + scale) internally. The NT-side `DeformableMesh`'s vertex buffer is updated each frame from LCS's world-space positions, so the rendering pipeline needs no special handling.

```cpp
auto* cube = mPhysics->add_body(
    ci::geom::Cube().size(glm::vec3(0.5f)),
    physics::RigidBodyConfig{
        .model        = physics::RigidModel::Orthogonality,
        .is_solid     = true,
        .stiffness    = 1e6f,
        .translation  = luisa::float3(0.f, 3.f, 0.f),
        .name         = "rigid_cube",
    },
    cubeMatIdx);
```

### Notes

- `is_solid=true` treats the mesh as a closed solid (uses mesh volume for mass + rotational inertia). `false` treats it as a thin shell.
- All three overloads from Cloth are mirrored: `ci::geom::Source`, `ci::TriMesh`, and existing `DeformableMesh*`.

---

## Per-frame Integration

```cpp
// In your update() / tick():

// 1. Advance physics on its dedicated stream (host blocks for the solve)
mPhysics->step();

// 2. Insert a wait on Pipeline's compute stream BEFORE Pipeline::update()
//    so Geometry::update sees completed vertex writes when it rebuilds
//    deformable BLAS.
mPhysics->wait_for_step(mPipeline->computeStream());
```

If you skip `wait_for_step`, deformable BLAS refits may read partially-written vertex data and produce flickering geometry.

---

## Scene-Global Solver Knobs

These live on the LCS `SceneParams` and apply to all bodies. They're exposed in `Physics::drawUi()`:

| Knob | LCS field | Effect |
|---|---|---|
| **Self-collision (IPC)** | `use_self_collision` | Enables IPC log-barrier collision between triangles of the same body. Off = cloth folds through itself, rigid bodies tunnel. Higher compute cost. |
| **CCD line search** | `use_ccd_linesearch` | Continuous Collision Detection in the Newton line search — prevents tunneling at large timesteps. Requires self-collision to be meaningful. |
| **Gravity** | `gravity` (float3, m/s²) | World gravity vector. Default `{0, -9.8, 0}`. |

Ground floor (`use_floor` + `floor.y`) is currently hardcoded off — add a checkbox to `Physics::drawUi` if needed.

---

## Materials — All Config Fields

```cpp
struct ClothBodyConfig {
    ClothStretchModel stretch_model = FEM_BW98;
    ClothBendingModel bending_model = QuadraticBending;
    float thickness, youngs_modulus, poisson_ratio, area_bending_stiffness;
    luisa::float3 translation, rotation_euler, scale;     // placement in LCS world space
    luisa::vector<FixedPinMethod> pin_methods;            // bbox-face positional pins
    float pin_range;
    luisa::vector<uint> pin_indices;                      // explicit vertex indices
    std::string name;
};

struct TetrahedralBodyConfig {
    TetModel model = StVK;
    float youngs_modulus, poisson_ratio;
    // IPC / shared MaterialBase knobs:
    float mass, density, d_hat, contact_offset, friction_mu;
    luisa::float3 translation, rotation_euler, scale;
    luisa::vector<FixedPinMethod> pin_methods;
    float pin_range;
    luisa::vector<uint> pin_indices;
    std::string name;
};

struct RigidBodyConfig {
    RigidModel model = Orthogonality;
    bool is_solid;
    float thickness, stiffness;                           // thickness used when is_solid=false
    // IPC / shared MaterialBase knobs:
    float mass, density, d_hat, contact_offset, friction_mu;
    luisa::float3 translation, rotation_euler, scale;
    std::string name;
};
```

`d_hat` is the IPC contact distance scale (smaller = stiffer contact). `friction_mu` is the Coulomb coefficient. `mass = 0` means derive from `density × volume`.

---

## Debugging

```cpp
mPhysics->dump_profile();   // LCS Profiler::print_tree — per-phase breakdown of last step
mPhysics->drawUi();         // ImGui panel: iters, PCG tol, self-collision, CCD, gravity, etc.
mPhysics->set_async_readback(false);  // also copy positions to host for inspection
```

---

## Caveats

- **TetMaterial designated-init limitation**: `lcs::Material::TetMaterial` has a user-provided constructor (sets `is_shell = false`) and is not a C++ aggregate. Inside NT code that constructs LCS materials directly, use default-construct + member-assign — designated initializers won't compile. (The `physics::Physics` API hides this; the limitation only matters if you bypass the wrapper.)
- **Rigid body motion**: the wrapper trusts LCS to write affine-transformed positions back into the buffer that `get_curr_positions_device` exposes. If a rigid body ever fails to move at runtime, this is the first suspect.
- **No per-body removal**: LCS doesn't support removing a single body. Use `restart()` to clear the whole scene.
- **`prepare()` is expensive**: LCS JIT-compiles all kernels on first call (multi-second). Don't call per-frame.
