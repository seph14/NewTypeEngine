#include "newtype/physics/Physics.h"

#include "newtype/core/Renderer.h"
#include "newtype/scene/DeformableMesh.h"
#include "newtype/util/Profiler.h"
#include "newtype/util/Vertex.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"

// LCS headers — only this TU may include them.
#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/world_data.h"
#include "SimulationCore/physical_material.h"
#include "Utils/profiler.h"

namespace newtype::physics {

using namespace luisa;
using scene::DeformableMesh;
using util::Vertex;

namespace {

// ---- enum mirrors -> lcs::Material / lcs::Initializer ----------------------

lcs::Material::ConstitutiveStretchModelCloth to_lcs(ClothStretchModel m) {
    switch (m) {
        case ClothStretchModel::Spring:   return lcs::Material::ConstitutiveStretchModelCloth::Spring;
        case ClothStretchModel::FEM_BW98: return lcs::Material::ConstitutiveStretchModelCloth::FEM_BW98;
    }
    return lcs::Material::ConstitutiveStretchModelCloth::FEM_BW98;
}

lcs::Material::ConstitutiveBendingModelCloth to_lcs(ClothBendingModel m) {
    switch (m) {
        case ClothBendingModel::QuadraticBending: return lcs::Material::ConstitutiveBendingModelCloth::QuadraticBending;
        case ClothBendingModel::DihedralAngle:    return lcs::Material::ConstitutiveBendingModelCloth::DihedralAngle;
    }
    return lcs::Material::ConstitutiveBendingModelCloth::QuadraticBending;
}

lcs::Initializer::FixedPointsType to_lcs(FixedPinMethod m) {
    switch (m) {
        case FixedPinMethod::Left:       return lcs::Initializer::FixedPointsType::Left;
        case FixedPinMethod::Right:      return lcs::Initializer::FixedPointsType::Right;
        case FixedPinMethod::Front:      return lcs::Initializer::FixedPointsType::Front;
        case FixedPinMethod::Back:       return lcs::Initializer::FixedPointsType::Back;
        case FixedPinMethod::Up:         return lcs::Initializer::FixedPointsType::Up;
        case FixedPinMethod::Down:       return lcs::Initializer::FixedPointsType::Down;
        case FixedPinMethod::LeftUp:     return lcs::Initializer::FixedPointsType::LeftUp;
        case FixedPinMethod::LeftDown:   return lcs::Initializer::FixedPointsType::LeftDown;
        case FixedPinMethod::LeftFront:  return lcs::Initializer::FixedPointsType::LeftFront;
        case FixedPinMethod::LeftBack:   return lcs::Initializer::FixedPointsType::LeftBack;
        case FixedPinMethod::RightUp:    return lcs::Initializer::FixedPointsType::RightUp;
        case FixedPinMethod::RightDown:  return lcs::Initializer::FixedPointsType::RightDown;
        case FixedPinMethod::RightFront: return lcs::Initializer::FixedPointsType::RightFront;
        case FixedPinMethod::RightBack:  return lcs::Initializer::FixedPointsType::RightBack;
        case FixedPinMethod::FrontUp:    return lcs::Initializer::FixedPointsType::FrontUp;
        case FixedPinMethod::FrontDown:  return lcs::Initializer::FixedPointsType::FrontDown;
        case FixedPinMethod::BackUp:     return lcs::Initializer::FixedPointsType::BackUp;
        case FixedPinMethod::BackDown:   return lcs::Initializer::FixedPointsType::BackDown;
        case FixedPinMethod::All:        return lcs::Initializer::FixedPointsType::All;
    }
    return lcs::Initializer::FixedPointsType::All;
}

lcs::Material::ConstitutiveModelTet to_lcs(TetModel m) {
    switch (m) {
        case TetModel::Spring:            return lcs::Material::ConstitutiveModelTet::Spring;
        case TetModel::StVK:              return lcs::Material::ConstitutiveModelTet::StVK;
        case TetModel::StableNeoHookean:  return lcs::Material::ConstitutiveModelTet::StableNeoHookean;
        case TetModel::Corotated:         return lcs::Material::ConstitutiveModelTet::Corotated;
        case TetModel::ARAP:              return lcs::Material::ConstitutiveModelTet::ARAP;
    }
    return lcs::Material::ConstitutiveModelTet::StVK;
}

lcs::Material::ConstitutiveModelRigid to_lcs(RigidModel m) {
    switch (m) {
        case RigidModel::Spring:            return lcs::Material::ConstitutiveModelRigid::Spring;
        case RigidModel::Orthogonality:     return lcs::Material::ConstitutiveModelRigid::Orthogonality;
        case RigidModel::ARAP:              return lcs::Material::ConstitutiveModelRigid::ARAP;
        case RigidModel::StableNeoHookean:  return lcs::Material::ConstitutiveModelRigid::StableNeoHookean;
    }
    return lcs::Material::ConstitutiveModelRigid::Orthogonality;
}

// Apply shared MaterialBase knobs (mass/density/d_hat/contact_offset/
// friction_mu) to a partially-constructed LCS material. MaterialBase is the
// common prefix of Cloth/Tet/Rigid/Rod materials, so we mutate via variant
// visit to avoid 4 near-identical overloads.
void apply_base_material(lcs::Material::MaterialVariant& var,
                         float mass, float density, float d_hat,
                         float contact_offset, float friction_mu) {
    std::visit([&](auto& m) {
        m.mass           = mass;
        m.density        = density;
        m.d_hat          = d_hat;
        m.contact_offset = contact_offset;
        m.friction_mu    = friction_mu;
    }, var);
}

// Extract LCS-format positions/faces from a ci::TriMesh.
struct TriMeshArrays {
    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<uint, 3>>  faces;
};

TriMeshArrays extract_arrays(const ci::TriMesh& tri) {
    TriMeshArrays out;
    const ci::vec3* positions = tri.getPositions<3>();
    const size_t    vcount    = tri.getNumVertices();
    out.positions.reserve(vcount);
    for (size_t i = 0; i < vcount; ++i) {
        out.positions.push_back({positions[i].x, positions[i].y, positions[i].z});
    }
    const auto& indices = tri.getIndices();
    LUISA_ASSERT(indices.size() % 3 == 0, "TriMesh index count must be a multiple of 3");
    out.faces.reserve(indices.size() / 3);
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        out.faces.push_back({indices[i], indices[i + 1], indices[i + 2]});
    }
    return out;
}

} // namespace

//==============================================================================
// Physics::Impl
//==============================================================================

struct Physics::Impl {
    // Declaration order = destruction order. Stream must outlive solver
    // because LCS holds a non-owning pointer to it.
    compute::Stream         stream;          // dedicated physics queue
    compute::TimelineEvent  stepDoneEvent;
    uint64_t                stepFence = 0u;

    unique_ptr<lcs::NewtonSolver> solver;
    bool                          prepared       = false;
    bool                          async_readback = true;

    struct Body {
        uint                  reg_id;          // LCS registration_id (stable)
        DeformableMesh*       mesh;            // non-owning; owned by Pipeline after addShape
        uint                  vertex_count;
        // Vertex -> adjacent-triangle adjacency (CSR). Built once at add_body
        // time; read-only on GPU. Each adj_tris entry stores the triangle's
        // three vertex indices, so the kernel doesn't need a separate
        // triangle-buffer binding.
        compute::Buffer<uint>      adj_offsets;  // size vertex_count + 1
        compute::Buffer<uint3>     adj_tris;     // total adjacent-triangle count
    };
    vector<Body> bodies;

    // Per-step CPU readback scratch (sync path only).
    std::vector<std::vector<std::array<float, 3>>> host_positions;

    // GPU kernel: LCS sa_x (float3) -> DeformableMesh next_vertex_buffer,
    // recomputing per-vertex normals as the area-weighted sum over adjacent
    // triangle face normals (unnormalized cross products carry 2*face-area
    // magnitude, so summing yields an area-weighted aggregate). Triangle
    // winding defines normal direction — do NOT flip, since deforming cloth
    // legitimately has regions facing any direction.
    // Tangents/UVs are preserved from the existing buffer contents.
    compute::Shader<1,
        compute::BufferView<float3>,   // lcs_pos (src positions)
        compute::BufferView<Vertex>,   // next_vertex_buffer (dst)
        compute::BufferView<uint>,     // adj_offsets
        compute::BufferView<uint3>,    // adj_tris (i0,i1,i2 per adjacent triangle)
        uint>                          // vertex count
        copy_positions;
};

//==============================================================================
// Lifecycle
//==============================================================================

Physics::Physics()
    : _impl(make_unique<Impl>()) {
    auto& device = core::Renderer::device();
    _impl->stream = device.create_stream();
    _impl->stepDoneEvent = device.create_timeline_event();

    _impl->solver = make_unique<lcs::NewtonSolver>();
    _impl->solver->set_device_from_pointers(
        reinterpret_cast<uintptr_t>(&device),
        reinterpret_cast<uintptr_t>(&_impl->stream));

    // Default config — mirrors PhysicsBridge's gravity-drop setup.
    auto& config = _impl->solver->get_config();
    config.implicit_dt          = 1.f / 60.f;
    config.use_floor            = false;
    config.use_self_collision   = false;
    config.nonlinear_iter_count = 3u;
    config.pcg_iter_count       = 11u;

    // Async copy kernel — topology-agnostic; reads existing buffer for
    // tangent/UV attributes, overwrites positions from LCS, and recomputes
    // normals from the deformed positions via the adjacency lists.
    _impl->copy_positions = device.compile<1>(
        [](compute::Var<compute::BufferView<float3>> lcs_pos,
           compute::Var<compute::BufferView<Vertex>>  next_verts,
           compute::Var<compute::BufferView<uint>>    adj_offsets,
           compute::Var<compute::BufferView<uint3>>   adj_tris,
           compute::UInt                              n) {
            compute::set_block_size(256u);
            compute::UInt i = compute::dispatch_x();
            $if (i < n) {
                compute::Var<Vertex> v = next_verts.read(i);
                compute::Var<float3> p = lcs_pos.read(i);
                v.px = p.x;  v.py = p.y;  v.pz = p.z;

                // Area-weighted normal: sum cross products of adjacent
                // triangles (each cross magnitude = 2*face_area).
                compute::UInt start = adj_offsets.read(i);
                compute::UInt end   = adj_offsets.read(i + 1u);
                compute::Var<float3> n_sum = compute::make_float3(0.f, 0.f, 0.f);
                $for (k, start, end) {
                    compute::Var<uint3>  tri = adj_tris.read(k);
                    compute::Var<float3> p0  = lcs_pos.read(tri.x);
                    compute::Var<float3> p1  = lcs_pos.read(tri.y);
                    compute::Var<float3> p2  = lcs_pos.read(tri.z);
                    n_sum = n_sum + compute::cross(p1 - p0, p2 - p0);
                };
                compute::Var<float3> normal = compute::normalize(n_sum);
                v.nx = normal.x;  v.ny = normal.y;  v.nz = normal.z;

                next_verts.write(i, v);
            };
        });
}

Physics::~Physics() {
    // Solver destroys first (declared after stream in Impl), so its cleanup
    // calls (if any) still see a valid stream.
    if (_impl->prepared) {
        _impl->stream << _impl->stepDoneEvent.signal(_impl->stepFence + 1)
                      << compute::synchronize();
    }
}

bool Physics::prepared()   const noexcept { return _impl->prepared; }
uint Physics::body_count() const noexcept { return static_cast<uint>(_impl->bodies.size()); }
bool Physics::async_readback() const noexcept { return _impl->async_readback; }
void Physics::set_async_readback(bool enable) noexcept { _impl->async_readback = enable; }

void Physics::dump_profile() const {
    // LCS resets the profiler tree at the start of each physics_step_GPU, so
    // this always reflects the most recent step's per-phase breakdown.
    lcs::Profiler::instance().print_tree();
}

//==============================================================================
// Body registration
//==============================================================================

namespace {

// Common LCS WorldData builder for all three add_body overloads.
lcs::Initializer::WorldData build_world_data(
    const TriMeshArrays&       arrays,
    const ClothBodyConfig&     cfg) {

    auto wd = lcs::Initializer::WorldData()
                  .set_name(cfg.name)
                  .load_mesh_from_array(arrays.positions, arrays.faces)
                  .set_material_type(lcs::Material::MaterialType::Cloth)
                  .set_physics_material(lcs::Material::ClothMaterial{
                      .stretch_model          = to_lcs(cfg.stretch_model),
                      .bending_model          = to_lcs(cfg.bending_model),
                      .thickness              = cfg.thickness,
                      .youngs_modulus         = cfg.youngs_modulus,
                      .poisson_ratio          = cfg.poisson_ratio,
                      .area_bending_stiffness = cfg.area_bending_stiffness,
                  })
                  .set_translation(cfg.translation.x, cfg.translation.y, cfg.translation.z)
                  .set_rotation(cfg.rotation_euler.x, cfg.rotation_euler.y, cfg.rotation_euler.z)
                  .set_scale(cfg.scale.x, cfg.scale.y, cfg.scale.z);

    for (auto m : cfg.pin_methods) {
        wd.add_fixed_point_from_method({
            .method = to_lcs(m),
            .range  = cfg.pin_range,
        });
    }
    if (!cfg.pin_indices.empty()) {
        std::vector<uint> idx_copy(cfg.pin_indices.begin(), cfg.pin_indices.end());
        wd.add_fixed_point_from_indices(idx_copy);
    }
    return wd;
}

// WorldData builder for tetrahedral bodies. Tet topology is authoritative;
// surface triangles are extracted by LCS via extract_surface_face_and_vert_
// from_tets inside load_tet_mesh_from_array. The shared MaterialBase knobs
// (mass/density/d_hat/contact_offset/friction_mu) are applied after
// construction via apply_base_material since load_tet_mesh_from_array has no
// chainable overload for them.
lcs::Initializer::WorldData build_world_data_tet(
    const std::vector<std::array<float, 3>>& tet_vertices,
    const std::vector<std::array<uint, 4>>&  tet_elements,
    const TetrahedralBodyConfig&             cfg) {

    // TetMaterial has a user-provided ctor (sets is_shell=false) and so is not
    // an aggregate — designated init fails. Default-construct then assign,
    // matching LCS's own set_physics_material_tet helper.
    lcs::Material::TetMaterial tet_mat;
    tet_mat.model          = to_lcs(cfg.model);
    tet_mat.youngs_modulus = cfg.youngs_modulus;
    tet_mat.poisson_ratio  = cfg.poisson_ratio;

    auto wd = lcs::Initializer::WorldData()
                  .set_name(cfg.name)
                  .load_tet_mesh_from_array(tet_vertices, tet_elements)
                  .set_material_type(lcs::Material::MaterialType::Tetrahedral)
                  .set_physics_material(tet_mat)
                  .set_translation(cfg.translation.x, cfg.translation.y, cfg.translation.z)
                  .set_rotation(cfg.rotation_euler.x, cfg.rotation_euler.y, cfg.rotation_euler.z)
                  .set_scale(cfg.scale.x, cfg.scale.y, cfg.scale.z);

    apply_base_material(wd.physics_material,
                        cfg.mass, cfg.density, cfg.d_hat,
                        cfg.contact_offset, cfg.friction_mu);

    for (auto m : cfg.pin_methods) {
        wd.add_fixed_point_from_method({
            .method = to_lcs(m),
            .range  = cfg.pin_range,
        });
    }
    if (!cfg.pin_indices.empty()) {
        std::vector<uint> idx_copy(cfg.pin_indices.begin(), cfg.pin_indices.end());
        wd.add_fixed_point_from_indices(idx_copy);
    }
    return wd;
}

// WorldData builder for rigid (ABD) bodies. Surface mesh only — LCS reduces
// to affine DOF internally. MaterialBase knobs applied post-construction.
lcs::Initializer::WorldData build_world_data_rigid(
    const TriMeshArrays&       arrays,
    const RigidBodyConfig&     cfg) {

    auto wd = lcs::Initializer::WorldData()
                  .set_name(cfg.name)
                  .load_mesh_from_array(arrays.positions, arrays.faces)
                  .set_material_type(lcs::Material::MaterialType::Rigid)
                  .set_physics_material(lcs::Material::RigidMaterial{
                      .model      = to_lcs(cfg.model),
                      .is_solid   = cfg.is_solid,
                      .thickness  = cfg.thickness,
                      .stiffness  = cfg.stiffness,
                  })
                  .set_translation(cfg.translation.x, cfg.translation.y, cfg.translation.z)
                  .set_rotation(cfg.rotation_euler.x, cfg.rotation_euler.y, cfg.rotation_euler.z)
                  .set_scale(cfg.scale.x, cfg.scale.y, cfg.scale.z);

    apply_base_material(wd.physics_material,
                        cfg.mass, cfg.density, cfg.d_hat,
                        cfg.contact_offset, cfg.friction_mu);
    return wd;
}

// Build a ci::TriMesh from raw position + face arrays. Used by the tet path
// where LCS extracts the surface triangles and we need to seed the NT-side
// DeformableMesh with matching topology. Normals are auto-recalculated by
// MeshShape::load_from; UVs default to (0,0).
ci::TriMesh build_trimesh_from_arrays(
    const std::vector<std::array<float, 3>>& positions,
    const std::vector<std::array<uint, 3>>&  faces) {
    ci::TriMesh::Format fmt;
    fmt.positions(3);
    ci::TriMesh tri(fmt);
    tri.appendPositions(reinterpret_cast<const ci::vec3*>(positions.data()),
                        static_cast<uint32_t>(positions.size()));
    // ci::TriMesh expects a flat uint32 index buffer.
    std::vector<uint32_t> indices_flat;
    indices_flat.reserve(faces.size() * 3u);
    for (const auto& f : faces) {
        indices_flat.push_back(f[0]);
        indices_flat.push_back(f[1]);
        indices_flat.push_back(f[2]);
    }
    tri.appendIndices(indices_flat.data(),
                      static_cast<uint32_t>(indices_flat.size()));
    return tri;
}

// Build a fresh DeformableMesh from a ci::TriMesh on the physics stream.
// Caller takes ownership (typically via Pipeline::addShape).
DeformableMesh* create_deformable_from_tri(
    const ci::TriMesh& tri, uint material_id, compute::Stream& stream) {
    auto mesh = DeformableMesh::create(core::Renderer::device(),
                                       /*requireDoubleBuffer=*/true,
                                       material_id);
    // load_from copies + (re)calculates normals/tangents/UVs.
    ci::TriMesh& tri_mut = const_cast<ci::TriMesh&>(tri);
    mesh->load_from(tri_mut);
    mesh->build(stream);
    return mesh.release();
}

// Build vertex -> adjacent-triangle adjacency in CSR form and upload to
// device buffers. Each adj_tris entry packs a triangle's three vertex
// indices so the kernel needs no extra bindings to read topology.
void upload_adjacency(compute::Buffer<uint>&   adj_offsets,
                      compute::Buffer<uint3>&  adj_tris,
                      uint                     vertex_count,
                      const std::vector<std::array<uint, 3>>& faces,
                      compute::Stream&         stream) {
    // Bucket adjacent triangle vertex-triples per vertex.
    std::vector<std::vector<luisa::uint3>> per_vertex(vertex_count);
    for (const auto& f : faces) {
        luisa::uint3 tri{f[0], f[1], f[2]};
        per_vertex[f[0]].push_back(tri);
        per_vertex[f[1]].push_back(tri);
        per_vertex[f[2]].push_back(tri);
    }

    // Flatten to CSR: offsets[vertex_count + 1] + packed triples.
    std::vector<uint> offsets(vertex_count + 1u, 0u);
    for (uint i = 0u; i < vertex_count; ++i) {
        offsets[i + 1u] = offsets[i] + static_cast<uint>(per_vertex[i].size());
    }
    std::vector<luisa::uint3> tris;
    tris.reserve(offsets.back());
    for (uint i = 0u; i < vertex_count; ++i) {
        for (auto t : per_vertex[i]) tris.push_back(t);
    }

    auto& device    = core::Renderer::device();
    adj_offsets     = device.create_buffer<uint>(vertex_count + 1u);
    adj_tris        = device.create_buffer<uint3>(tris.size());
    stream << adj_offsets.copy_from(offsets.data())
           << adj_tris.copy_from(tris.data());
}

} // namespace

DeformableMesh* Physics::add_body(
    const ci::geom::Source& geom, const ClothBodyConfig& cfg, uint material_id) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    ci::TriMesh tri(geom);
    auto arrays = extract_arrays(tri);
    auto* mesh  = create_deformable_from_tri(tri, material_id, _impl->stream);

    auto wd = build_world_data(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = mesh,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return mesh;
}

DeformableMesh* Physics::add_body(
    const ci::TriMesh& trimesh, const ClothBodyConfig& cfg, uint material_id) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    auto arrays = extract_arrays(trimesh);
    auto* mesh  = create_deformable_from_tri(trimesh, material_id, _impl->stream);

    auto wd = build_world_data(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = mesh,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return mesh;
}

DeformableMesh* Physics::add_body(
    DeformableMesh* existing, const ClothBodyConfig& cfg) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    LUISA_ASSERT(existing && existing->has_cpu_data(),
                 "Physics::add_body(existing, ...) requires has_cpu_data() == true.");

    TriMeshArrays arrays;
    arrays.positions.reserve(existing->vertices().size());
    for (const auto& v : existing->vertices()) {
        arrays.positions.push_back({v.px, v.py, v.pz});
    }
    arrays.faces.reserve(existing->triangles().size());
    for (const auto& t : existing->triangles()) {
        arrays.faces.push_back({t.i0, t.i1, t.i2});
    }

    auto wd = build_world_data(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = existing,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return existing;
}

//==============================================================================
// Tetrahedral (soft body) add_body
//==============================================================================

DeformableMesh* Physics::add_body(
    const TetrahedralBodyConfig&            cfg,
    uint                                    material_id,
    luisa::span<const std::array<float, 3>> tet_vertices,
    luisa::span<const std::array<uint, 4>>  tet_elements) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");

    std::vector<std::array<float, 3>> verts(tet_vertices.begin(), tet_vertices.end());
    std::vector<std::array<uint, 4>>  tets(tet_elements.begin(), tet_elements.end());

    auto wd = build_world_data_tet(verts, tets, cfg);

    // Pull the LCS-extracted surface faces back so we can seed the NT-side
    // TriMesh with matching topology. Surface faces index into model_positions
    // (all tet verts, including interior).
    const auto& mesh = wd.get_mesh();
    std::vector<std::array<float, 3>> surf_positions(mesh.model_positions.size());
    for (size_t i = 0; i < mesh.model_positions.size(); ++i) {
        surf_positions[i] = {mesh.model_positions[i][0],
                             mesh.model_positions[i][1],
                             mesh.model_positions[i][2]};
    }
    std::vector<std::array<uint, 3>> surf_faces(mesh.faces.size());
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        surf_faces[i] = {mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]};
    }

    auto tri    = build_trimesh_from_arrays(surf_positions, surf_faces);
    auto* dmesh = create_deformable_from_tri(tri, material_id, _impl->stream);

    uint reg_id = _impl->solver->register_world_data(wd);
    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = dmesh,
        .vertex_count = static_cast<uint>(surf_positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     surf_faces, _impl->stream);
    return dmesh;
}

DeformableMesh* Physics::add_body(
    const TetrahedralBodyConfig&            cfg,
    uint                                    material_id,
    const std::string&                      tet_mesh_path) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");

    // Build a placeholder WorldData and call load_tet_mesh_from_path. Material
    // config is applied post-load via the same path as the array overload.
    // TetMaterial has a user-provided ctor (sets is_shell=false) — not an
    // aggregate, so designated init fails. Use default-construct + assign.
    lcs::Material::TetMaterial tet_mat;
    tet_mat.model          = to_lcs(cfg.model);
    tet_mat.youngs_modulus = cfg.youngs_modulus;
    tet_mat.poisson_ratio  = cfg.poisson_ratio;

    auto wd = lcs::Initializer::WorldData()
                  .set_name(cfg.name)
                  .load_tet_mesh_from_path(tet_mesh_path)
                  .set_material_type(lcs::Material::MaterialType::Tetrahedral)
                  .set_physics_material(tet_mat)
                  .set_translation(cfg.translation.x, cfg.translation.y, cfg.translation.z)
                  .set_rotation(cfg.rotation_euler.x, cfg.rotation_euler.y, cfg.rotation_euler.z)
                  .set_scale(cfg.scale.x, cfg.scale.y, cfg.scale.z);

    apply_base_material(wd.physics_material,
                        cfg.mass, cfg.density, cfg.d_hat,
                        cfg.contact_offset, cfg.friction_mu);

    for (auto m : cfg.pin_methods) {
        wd.add_fixed_point_from_method({
            .method = to_lcs(m),
            .range  = cfg.pin_range,
        });
    }
    if (!cfg.pin_indices.empty()) {
        std::vector<uint> idx_copy(cfg.pin_indices.begin(), cfg.pin_indices.end());
        wd.add_fixed_point_from_indices(idx_copy);
    }

    const auto& mesh = wd.get_mesh();
    std::vector<std::array<float, 3>> surf_positions(mesh.model_positions.size());
    for (size_t i = 0; i < mesh.model_positions.size(); ++i) {
        surf_positions[i] = {mesh.model_positions[i][0],
                             mesh.model_positions[i][1],
                             mesh.model_positions[i][2]};
    }
    std::vector<std::array<uint, 3>> surf_faces(mesh.faces.size());
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        surf_faces[i] = {mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]};
    }

    auto tri    = build_trimesh_from_arrays(surf_positions, surf_faces);
    auto* dmesh = create_deformable_from_tri(tri, material_id, _impl->stream);

    uint reg_id = _impl->solver->register_world_data(wd);
    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = dmesh,
        .vertex_count = static_cast<uint>(surf_positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     surf_faces, _impl->stream);
    return dmesh;
}

//==============================================================================
// Rigid (ABD) add_body
//==============================================================================

DeformableMesh* Physics::add_body(
    const ci::geom::Source& geom, const RigidBodyConfig& cfg, uint material_id) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    ci::TriMesh tri(geom);
    auto arrays = extract_arrays(tri);
    auto* mesh  = create_deformable_from_tri(tri, material_id, _impl->stream);

    auto wd = build_world_data_rigid(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = mesh,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return mesh;
}

DeformableMesh* Physics::add_body(
    const ci::TriMesh& trimesh, const RigidBodyConfig& cfg, uint material_id) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    auto arrays = extract_arrays(trimesh);
    auto* mesh  = create_deformable_from_tri(trimesh, material_id, _impl->stream);

    auto wd = build_world_data_rigid(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = mesh,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return mesh;
}

DeformableMesh* Physics::add_body(
    DeformableMesh* existing, const RigidBodyConfig& cfg) {
    LUISA_ASSERT(!_impl->prepared,
                 "Physics::add_body called after prepare(); call restart() first.");
    LUISA_ASSERT(existing && existing->has_cpu_data(),
                 "Physics::add_body(existing, ...) requires has_cpu_data() == true.");

    TriMeshArrays arrays;
    arrays.positions.reserve(existing->vertices().size());
    for (const auto& v : existing->vertices()) {
        arrays.positions.push_back({v.px, v.py, v.pz});
    }
    arrays.faces.reserve(existing->triangles().size());
    for (const auto& t : existing->triangles()) {
        arrays.faces.push_back({t.i0, t.i1, t.i2});
    }

    auto wd = build_world_data_rigid(arrays, cfg);
    uint reg_id = _impl->solver->register_world_data(wd);

    _impl->bodies.push_back({
        .reg_id       = reg_id,
        .mesh         = existing,
        .vertex_count = static_cast<uint>(arrays.positions.size()),
    });
    auto& body = _impl->bodies.back();
    upload_adjacency(body.adj_offsets, body.adj_tris, body.vertex_count,
                     arrays.faces, _impl->stream);
    return existing;
}

//==============================================================================
// prepare / restart
//==============================================================================

void Physics::prepare() {
    LUISA_ASSERT(!_impl->prepared, "Physics::prepare() called twice.");
    if (_impl->bodies.empty()) {
        CI_LOG_W("Physics::prepare() called with no bodies; solver will idle.");
    }
    _impl->solver->init_solver();
    _impl->prepared = true;
}

void Physics::restart() noexcept {
    if (_impl->prepared) {
        // Drain any in-flight work before tearing down solver state.
        _impl->stream << compute::synchronize();
    }
    _impl->solver = make_unique<lcs::NewtonSolver>();
    _impl->solver->set_device_from_pointers(
        reinterpret_cast<uintptr_t>(&core::Renderer::device()),
        reinterpret_cast<uintptr_t>(&_impl->stream));
    auto& config = _impl->solver->get_config();
    config.implicit_dt          = 1.f / 60.f;
    config.use_floor            = false;
    config.use_self_collision   = false;
    config.nonlinear_iter_count = 3u;
    config.pcg_iter_count       = 11u;

    _impl->bodies.clear();
    _impl->host_positions.clear();
    _impl->prepared = false;
    _impl->stepFence = 0u;
}

//==============================================================================
// step / wait_for_step
//==============================================================================

void Physics::step() {
    LUISA_ASSERT(_impl->prepared,
                 "Physics::step() called before prepare().");
    if (_impl->bodies.empty()) return;

    {
        util::CpuScopedTimer _("Physics/step_GPU");
        _impl->solver->physics_step_GPU();   // commits on _impl->stream; host blocks internally
    }

    // Sync mode: read positions back to host for inspection/logging only.
    // The deformable-mesh update always goes through the GPU kernel below so
    // normals stay correct; sync vs async just toggles whether we pay for
    // the CPU round-trip.
    if (!_impl->async_readback) {
        util::CpuScopedTimer _("Physics/readback");
        _impl->solver->get_curr_vertices_to_host(_impl->host_positions);
    }

    // GPU kernel: LCS sa_x -> next_vertex_buffer (positions) + recompute
    // area-weighted normals via per-body adjacency.
    for (auto& body : _impl->bodies) {
        auto src = _impl->solver->get_curr_positions_device(body.reg_id);
        _impl->stream << _impl->copy_positions(
            src,
            body.mesh->next_vertex_buffer(),
            body.adj_offsets,
            body.adj_tris,
            body.vertex_count
        ).dispatch(body.vertex_count);
        body.mesh->mark_buffer_dirty();
    }

    // Signal completion — downstream waits via wait_for_step().
    _impl->stream << _impl->stepDoneEvent.signal(++_impl->stepFence);
}

void Physics::wait_for_step(compute::Stream& downstream) const {
    if (_impl->stepFence == 0u) return;
    downstream << _impl->stepDoneEvent.wait(_impl->stepFence);
}

//==============================================================================
// UI
//==============================================================================

void Physics::drawUi() {
    if (!ImGui::CollapsingHeader("Physics"))
        return;

    auto& config = _impl->solver->get_config();

    int   nonlinear = static_cast<int>(config.nonlinear_iter_count);
    int   pcg       = static_cast<int>(config.pcg_iter_count);
    int   interval  = static_cast<int>(config.pcg_check_interval);
    float rel_tol   = config.pcg_rel_tol;
    bool  async_on  = _impl->async_readback;
    bool  prep      = _impl->prepared;
    bool  self_coll = config.use_self_collision;
    bool  ccd_ls    = config.use_ccd_linesearch;
    float grav[3]   = { config.gravity.x, config.gravity.y, config.gravity.z };

    ImGui::Text("Bodies: %u   Prepared: %s",
                static_cast<uint>(_impl->bodies.size()),
                prep ? "yes" : "no");

    if (!prep && ImGui::Button("Prepare")) prepare();
    ImGui::SameLine();
    if (ImGui::Button("Restart")) restart();

    if (ImGui::Checkbox("Async readback (no CPU round trip)", &async_on)) {
        _impl->async_readback = async_on;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("When ON: GPU-only update (positions + recomputed normals).\nWhen OFF: additionally copies positions back to host for debug/profiling. Vertex update is identical.");

    if (ImGui::Checkbox("Self-collision (IPC)", &self_coll)) {
        config.use_self_collision = self_coll;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Enables LCS's IPC log-barrier collision between triangles of the same body.\nOff = cloth folds through itself, rigid bodies tunnel. Higher compute cost.");

    if (ImGui::Checkbox("CCD line search", &ccd_ls)) {
        config.use_ccd_linesearch = ccd_ls;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Continuous Collision Detection in the Newton line search — prevents tunneling at large timesteps.\nRequires self-collision to be meaningful; otherwise a no-op.");

    ImGui::PushItemWidth(180.f);
    if (ImGui::SliderInt("Newton iters", &nonlinear, 1, 20)) {
        config.nonlinear_iter_count = static_cast<uint>(nonlinear);
    }
    if (ImGui::SliderInt("PCG iters", &pcg, 1, 200)) {
        config.pcg_iter_count = static_cast<uint>(pcg);
    }
    if (ImGui::SliderInt("PCG check interval", &interval, 0, 25)) {
        config.pcg_check_interval = static_cast<uint>(interval);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("0 = run full PCG iter count (legacy).\n>0 = check convergence every N iters and exit early when rel tol is met.");
    if (ImGui::DragFloat("PCG rel tol", &rel_tol, 1e-6f, 1e-8f, 1e-1f, "%.1e")) {
        config.pcg_rel_tol = rel_tol;
    }
    if (ImGui::DragFloat3("gravity", grav, 0.1f, -50.f, 50.f, "%.2f")) {
        config.gravity = luisa::make_float3(grav[0], grav[1], grav[2]);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("World gravity vector in m/s^2. Default {0, -9.8, 0}. Apply before prepare() for clean state; LCS reads this each step.");
    ImGui::PopItemWidth();

    ImGui::Text("implicit_dt: %.4f s", config.implicit_dt);
    ImGui::Text("frame: %u", config.current_frame);

    if (ImGui::Button("Dump LCS profile")) {
        dump_profile();
    }
}

} // namespace newtype::physics
