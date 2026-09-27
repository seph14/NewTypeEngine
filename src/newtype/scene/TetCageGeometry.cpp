//==============================================================================
// TetCageGeometry — .tetcage loader + per-frame cage animation
// (docs/tetrahedral-cage-prototype.md, Stage 2)
//==============================================================================

#include "newtype/scene/TetCageGeometry.h"
#include "newtype/core/Config.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/util/Profiler.h"
#include "cinder/Log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace newtype::scene {

using namespace luisa;
using namespace luisa::compute;

namespace {

// Per-tet affine M = A · R⁻¹ from the 4 animated cage corners (R⁻¹
// precomputed in Tet::rinv by load()). Returns false for a collapsed tet
// (below kMinDet) — the caller keeps the previous matrix for the frame.
bool tet_affine(const TetCageGeometry::Tet &tet, const float3 a[4],
                float4x4 &m) noexcept {
    const float3 A[3] = {a[1] - a[0], a[2] - a[0], a[3] - a[0]};
    // L.col(j) = A · rinv[j]
    const float3 L0 = A[0] * tet.rinv[0].x + A[1] * tet.rinv[0].y + A[2] * tet.rinv[0].z;
    const float3 L1 = A[0] * tet.rinv[1].x + A[1] * tet.rinv[1].y + A[2] * tet.rinv[1].z;
    const float3 L2 = A[0] * tet.rinv[2].x + A[1] * tet.rinv[2].y + A[2] * tet.rinv[2].z;
    const float det = dot(L0, cross(L1, L2));
    if (!std::isfinite(det) || std::abs(det) < TetCageGeometry::kMinDet)
        return false;
    const float3 trans = a[0] - (L0 * tet.r0.x + L1 * tet.r0.y + L2 * tet.r0.z);
    m.cols[0] = make_float4(L0, 0.0f);
    m.cols[1] = make_float4(L1, 0.0f);
    m.cols[2] = make_float4(L2, 0.0f);
    m.cols[3] = make_float4(trans, 1.0f);
    return true;
}

// 3×3 inverse via the vector triple-product identity: for columns a, b, c the
// ROWS of the inverse are (b×c, c×a, a×b) / det, where det = a·(b×c).
// Returns false when |det| is indistinguishable from 0. (Callers wanting the
// inverse as columns must transpose the output — see load().)
bool invert_edges(const float3 &e1, const float3 &e2, const float3 &e3,
                  float3 inv[3]) noexcept {
    float3 c1 = cross(e1, e2);           // (a×b) — column 2 of the inverse
    float det = dot(e1, cross(e2, e3));  // a·(b×c)
    // Scale the epsilon by the edge matrix's magnitude so tiny-but-well-shaped
    // tets (small voxels) do not trip the guard.
    float scale = std::max(std::max(length(e1), length(e2)), length(e3));
    if (std::abs(det) < 1e-20f * scale * scale * scale || !std::isfinite(det))
        return false;
    float invDet = 1.0f / det;
    inv[0] = cross(e2, e3) * invDet;     // (b×c)/det
    inv[1] = cross(e3, e1) * invDet;     // (c×a)/det
    inv[2] = c1 * invDet;                // (a×b)/det
    return true;
}

// Column-major 4×4 product (a·b): column i of the result is a * b.col(i).
float4x4 mul4(const float4x4 &a, const float4x4 &b) noexcept {
    float4x4 r;
    for (int i = 0; i < 4; ++i) {
        r.cols[i] = a.cols[0] * b.cols[i].x +
                    a.cols[1] * b.cols[i].y +
                    a.cols[2] * b.cols[i].z +
                    a.cols[3] * b.cols[i].w;
    }
    return r;
}

} // namespace

//==============================================================================
// Loading
//==============================================================================

bool TetCageGeometry::load(const std::filesystem::path &path) noexcept {
    if (_built) {
        CI_LOG_E("TetCageGeometry: load() after build() is not supported");
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        CI_LOG_E("TetCageGeometry: cannot open " << path.string());
        return false;
    }

    auto readU32 = [&in]() -> uint32_t {
        uint32_t v; in.read(reinterpret_cast<char *>(&v), 4); return v;
    };
    auto readI32 = [&in]() -> int32_t {
        int32_t v; in.read(reinterpret_cast<char *>(&v), 4); return v;
    };
    auto readF32 = [&in]() -> float {
        float v; in.read(reinterpret_cast<char *>(&v), 4); return v;
    };

    char magic[9] = {};
    in.read(magic, 8);
    if (std::string_view(magic, 8) != "TETCAGE1") {
        CI_LOG_E("TetCageGeometry: bad magic in " << path.string());
        return false;
    }
    uint32_t version = readU32();
    if (version != 1u) {
        CI_LOG_E("TetCageGeometry: unsupported version " << version
            << " in " << path.string());
        return false;
    }

    // Grid header (informational for the runtime; the affine solves only need
    // the cage vertex table). Read into locals — argument evaluation order is
    // unspecified and MSVC runs right-to-left.
    float gox = readF32(), goy = readF32(), goz = readF32();
    float3 gridOrigin = make_float3(gox, goy, goz);
    float voxelSize = readF32();
    uint32_t dims[3] = {readU32(), readU32(), readU32()};
    (void)gridOrigin; (void)voxelSize; (void)dims;

    uint32_t cageVertCount = readU32();
    uint32_t tetCount = readU32();
    uint32_t pieceVertCount = readU32();
    uint32_t pieceTriCount = readU32();
    uint32_t frameCount = readU32();
    uint32_t flags = readU32();
    (void)flags;
    if (frameCount != 0u)
        CI_LOG_W("TetCageGeometry: " << path.string() << " carries "
            << frameCount << " baked cage frames (Stage 3 data) - ignored by "
            "the wind-driven runtime path");

    if (cageVertCount == 0u || tetCount == 0u || pieceTriCount == 0u) {
        CI_LOG_E("TetCageGeometry: empty cage in " << path.string());
        return false;
    }

    _restCage.resize(cageVertCount);
    for (auto &v : _restCage) {
        // Read into locals: argument evaluation order is unspecified (MSVC
        // runs right-to-left), so side-effecting reads must not sit inside
        // make_float3(...) directly.
        float x = readF32(), y = readF32(), z = readF32();
        v = make_float3(x, y, z);
    }

    _tets.resize(tetCount);
    for (auto &t : _tets) {
        for (int i = 0; i < 4; ++i) t.cage_verts[i] = readU32();
        for (int i = 0; i < 4; ++i) t.neighbors[i] = readI32();
        t.vert_start = readU32(); t.vert_count = readU32();
        t.tri_start = readU32(); t.tri_count = readU32();
    }

    _pieceVerts.resize(pieceVertCount);
    for (auto &v : _pieceVerts) {
        float px = readF32(), py = readF32(), pz = readF32();
        float nx = readF32(), ny = readF32(), nz = readF32();
        float u = readF32(), vv = readF32();
        // Tangents are not part of the .tetcage payload; the tangent property
        // flag is cleared per piece so shaders derive the frame from normals.
        v = Vertex::encode(make_float3(px, py, pz), make_float3(nx, ny, nz),
                           make_float4(1.f, 0.f, 0.f, 1.f), make_float2(u, vv));
    }

    _pieceTris.resize(pieceTriCount);
    for (auto &tri : _pieceTris)
        tri = Triangle{readU32(), readU32(), readU32()};

    if (!in.good()) {
        CI_LOG_E("TetCageGeometry: truncated file " << path.string());
        _tets.clear(); _restCage.clear(); _pieceVerts.clear(); _pieceTris.clear();
        return false;
    }

    // Validate indices + precompute the per-tet rest frame (r0 and the inverse
    // edge matrix). Tets failing the inversion are dropped from instancing
    // (they have no reliable affine map).
    uint32_t dropped = 0u, failEmpty = 0u, failCv = 0u, failRange = 0u,
             failTri = 0u, failInv = 0u;
    for (auto &t : _tets) {
        bool ok = t.tri_count > 0u;
        if (!ok) ++failEmpty;
        for (int i = 0; i < 4 && ok; ++i)
            ok = t.cage_verts[i] < cageVertCount;
        if (!ok && t.tri_count > 0u) ++failCv;
        if (ok) {
            ok = t.vert_start + t.vert_count <= pieceVertCount &&
                 t.tri_start + t.tri_count <= pieceTriCount;
            if (!ok) ++failRange;
        }
        if (ok) {
            for (uint32_t k = 0u; k < t.tri_count && ok; ++k) {
                const auto &tri = _pieceTris[t.tri_start + k]; // indices are piece-local
                ok = tri.i0 < t.vert_count && tri.i1 < t.vert_count &&
                     tri.i2 < t.vert_count;
            }
            if (!ok) ++failTri;
        }
        if (ok) {
            const float3 r0 = _restCage[t.cage_verts[0]];
            float3 inv[3];
            if (invert_edges(_restCage[t.cage_verts[1]] - r0,
                             _restCage[t.cage_verts[2]] - r0,
                             _restCage[t.cage_verts[3]] - r0, inv)) {
                t.r0 = r0;
                // invert_edges yields the inverse's ROWS (inv[k] = row k);
                // store transposed so rinv[j] is COLUMN j, as update() and the
                // bake path index it (L.col(j) = A · rinv[j]). Storing the
                // rows untransposed made every deformed affine garbage while
                // rest pose (identity) stayed watertight — the Stage-2
                // "shattered leaf" bug.
                t.rinv[0] = make_float3(inv[0].x, inv[1].x, inv[2].x);
                t.rinv[1] = make_float3(inv[0].y, inv[1].y, inv[2].y);
                t.rinv[2] = make_float3(inv[0].z, inv[1].z, inv[2].z);
            } else {
                ok = false;
                ++failInv;
            }
        }
        if (!ok) {
            if (t.tri_count > 0u) ++dropped;
            t.tri_count = 0u; // treat as empty
        }
    }
    _pieceCount = 0u;
    for (const auto &t : _tets)
        if (t.tri_count > 0u) ++_pieceCount;
    if (dropped > 0u)
        CI_LOG_W("TetCageGeometry: dropped " << dropped
            << " non-instancable tets (empty " << failEmpty << ", cvIdx " << failCv
            << ", range " << failRange << ", triIdx " << failTri
            << ", invert " << failInv << ") from " << path.string());
    if (_pieceCount == 0u) {
        CI_LOG_E("TetCageGeometry: no usable pieces in " << path.string());
        _tets.clear(); _restCage.clear(); _pieceVerts.clear(); _pieceTris.clear();
        return false;
    }

    _restMin = make_float3(std::numeric_limits<float>::max());
    _restMax = make_float3(std::numeric_limits<float>::lowest());
    for (const auto &t : _tets) {
        if (t.tri_count == 0u) continue;
        for (uint32_t i = 0u; i < t.vert_count; ++i) {
            float3 p = _pieceVerts[t.vert_start + i].position();
            _restMin = min(_restMin, p);
            _restMax = max(_restMax, p);
        }
    }

    CI_LOG_I("TetCageGeometry: loaded " << path.string()
        << " - tets " << tetCount << " (pieces " << _pieceCount << ")"
        << ", cage verts " << cageVertCount
        << ", piece tris " << pieceTriCount
        << ", bounds [" << _restMin.x << "," << _restMin.y << "," << _restMin.z
        << "]..[" << _restMax.x << "," << _restMax.y << "," << _restMax.z << "]");
    return true;
}

uint TetCageGeometry::add_copy(const luisa::float4x4 &world) noexcept {
    if (_built) {
        CI_LOG_E("TetCageGeometry: add_copy() after build() is not supported");
        return ~0u;
    }
    _copyWorld.push_back(world);
    return static_cast<uint>(_copyWorld.size()) - 1u;
}

void TetCageGeometry::set_deform_shader_id(luisa::string_view shaderId,
                                           uint blockSize) noexcept {
    if (_built) {
        CI_LOG_E("TetCageGeometry: set_deform_shader_id() after build() is not supported");
        return;
    }
    _deformShader.assign(shaderId);
    _deformShaderBlock = blockSize;
}

void TetCageGeometry::set_deform_state(uint copy,
                                       const TetCageDeformState &state) noexcept {
    if (copy >= _copyWorld.size()) return;
    if (_built) {
        // Post-build states feed the shader; without a shader nobody consumes
        // them (and no state buffer exists).
        if (!_deformShader.is_set()) return;
        _cageStateCPU[copy] = state;
        _dirtyCageStates.push_back(copy);
    } else {
        if (copy >= _cageStateCPU.size()) _cageStateCPU.resize(copy + 1u);
        _cageStateCPU[copy] = state;
    }
}

void TetCageGeometry::set_deform_state_all(const TetCageDeformState &state) noexcept {
    if (_copyWorld.empty()) return;
    _cageStateCPU.resize(_copyWorld.size());
    std::fill(_cageStateCPU.begin(), _cageStateCPU.end(), state);
    if (_built) {
        if (!_deformShader.is_set()) return;
        _dirtyCageStates.clear(); // subsumed by the whole-buffer upload
        _allStatesDirty = true;
    }
}

void TetCageGeometry::bake_initial_pose() noexcept {
    if (_built) {
        CI_LOG_E("TetCageGeometry: bake_initial_pose() after build() is not supported");
        return;
    }
    _bakeInitialPose = true;
}

//==============================================================================
// Registration
//==============================================================================

void TetCageGeometry::build(core::Pipeline &pipeline, Stream &stream,
                            uint material_id, bool double_sided) noexcept {
    if (_built || _tets.empty()) return;
    if (_copyWorld.empty()) _copyWorld.push_back(make_float4x4(1.0f));

    const uint cageVerts = static_cast<uint>(_restCage.size());
    const uint copies = static_cast<uint>(_copyWorld.size());

    // Deform GPU state (only meaningful with a registered shader id). When
    // baking, solve the current deformed pose once so it ships with the
    // initial instance upload (no per-frame transform writes).
    bool baked = false;
    if (_deformShader.is_set()) {
        _animCage.assign(static_cast<size_t>(copies) * cageVerts, make_float3(0.f));
        _cageStateCPU.resize(copies);
        _restCageBuffer = _device.create_buffer<float3>(std::max(1u, cageVerts));
        _animCageBuffer = _device.create_buffer<float3>(std::max(1u, copies * cageVerts));
        _cageStateBuffer = _device.create_buffer<TetCageDeformState>(std::max(1u, copies));
        stream << _restCageBuffer.copy_from(_restCage.data());
        stream << _cageStateBuffer.copy_from(_cageStateCPU.data());
        if (_bakeInitialPose) {
            dispatch_deform(stream);
            baked = true;
        }
    } else if (_bakeInitialPose) {
        CI_LOG_W("TetCageGeometry: bake_initial_pose() without a deform shader"
            " id - baking the rest pose");
    }

    // One prototype per non-empty piece. Piece vertices/triangles are sliced
    // from the soup (triangle indices are already local to their piece).
    // Geometry::build() builds every prototype's BLAS once.
    const uint tetCount = static_cast<uint>(_tets.size());
    luisa::vector<core::Pipeline::PrototypeHandle> protoOf(tetCount);
    for (uint t = 0u; t < tetCount; ++t) {
        const auto &tet = _tets[t];
        if (tet.tri_count == 0u) continue;

        auto shape = MeshShape::create(_device, material_id);
        shape->set_data(
            luisa::span<const Vertex>{_pieceVerts.data() + tet.vert_start, tet.vert_count},
            luisa::span<const Triangle>{_pieceTris.data() + tet.tri_start, tet.tri_count});
        shape->set_property_flag(PROPERTY_HAS_VERTEX_TANGENT, false);
        shape->set_double_sided(double_sided);
        protoOf[t] = pipeline.addPrototype(std::move(shape));
    }

    // One TLAS instance per tet per copy, seeded with the rest-pose matrix
    // (copy world × identity affine) — or, when bake_initial_pose() ran, with
    // the deformed pose so no post-build transform writes happen.
    _instanceIds.resize(_copyWorld.size() * tetCount, kInvalidShapeId);
    for (uint c = 0u; c < _copyWorld.size(); ++c) {
        for (uint t = 0u; t < tetCount; ++t) {
            if (_tets[t].tri_count == 0u) continue;
            float4x4 m = _copyWorld[c];
            if (baked) {
                const auto &tet = _tets[t];
                float3 a[4];
                for (int i = 0; i < 4; ++i)
                    a[i] = _animCage[c * cageVerts + tet.cage_verts[i]];
                float4x4 local;
                if (tet_affine(tet, a, local))
                    m = mul4(_copyWorld[c], local);
            }
            _instanceIds[c * tetCount + t] =
                pipeline.addPrototypeInstance(protoOf[t], m);
        }
    }

    // GPU solve path (docs/TLAS-instance-transform-updates.md Part 3): the
    // per-tet affine runs on the device and the TLAS copies the registered
    // rows from the instance-transform buffer during its build — no
    // readback, no CPU solve, no per-instance setShapeTransform.
    // NT_TETCAGE_CPU_PATH=1 forces the original readback path (A/B baseline).
    static const bool sForceCpuPath = std::getenv("NT_TETCAGE_CPU_PATH") != nullptr;
    if (_deformShader.is_set() && !sForceCpuPath) {
        _gpuPathWanted = true;

        // Rest frames: one per valid tet (copy-independent half of the solve).
        luisa::vector<uint> restIdxOf(tetCount, ~0u);
        _restFramesCPU.clear();
        for (uint t = 0u; t < tetCount; ++t) {
            if (_tets[t].tri_count == 0u) continue;
            TetCageRestFrame fr{};
            fr.r0 = _tets[t].r0;
            fr.rin[0] = _tets[t].rinv[0];
            fr.rin[1] = _tets[t].rinv[1];
            fr.rin[2] = _tets[t].rinv[2];
            restIdxOf[t] = static_cast<uint>(_restFramesCPU.size());
            _restFramesCPU.push_back(fr);
        }
        _restFrameBuffer = _device.create_buffer<TetCageRestFrame>(
            static_cast<uint>(_restFramesCPU.size()));
        stream << _restFrameBuffer.copy_from(_restFramesCPU.data());
        _copyWorldBuffer = _device.create_buffer<float4x4>(copies);
        stream << _copyWorldBuffer.copy_from(_copyWorld.data());

        // TetSolve kernel — mirrors the CPU tet_affine() op-for-op (same
        // expression order, so the A/B paths produce the same matrices;
        // |det| below kMinDet, including NaN, keeps the old matrix like the
        // CPU isfinite+abs guard). Engine-internal logic: compiled directly,
        // NOT via ShaderManager (the user customization point stays the wind
        // shader).
        _tetSolveShader = _device.compile<1>(
            [](compute::BufferVar<luisa::float4x4> transforms,
               compute::BufferVar<luisa::float4x4> transforms_prev,
               compute::BufferVar<TetCageSolveRow> rows,
               compute::BufferVar<TetCageRestFrame> rest,
               compute::BufferVar<luisa::float3> anim_cage,
               compute::BufferVar<luisa::float4x4> copy_world) noexcept {
                const UInt ridx = dispatch_id().x;
                Var<TetCageSolveRow> row = rows.read(ridx);
                Var<TetCageRestFrame> fr = rest.read(row.rest_idx);
                const Float3 a0 = anim_cage.read(row.anim_base + row.cage_verts[0]);
                const Float3 a1 = anim_cage.read(row.anim_base + row.cage_verts[1]);
                const Float3 a2 = anim_cage.read(row.anim_base + row.cage_verts[2]);
                const Float3 a3 = anim_cage.read(row.anim_base + row.cage_verts[3]);
                // A = [a1-a0, a2-a0, a3-a0]; L.col(j) = A · rin[j]
                const Float3 e0 = a1 - a0;
                const Float3 e1 = a2 - a0;
                const Float3 e2 = a3 - a0;
                const Float3 L0 = e0 * fr.rin[0].x + e1 * fr.rin[0].y + e2 * fr.rin[0].z;
                const Float3 L1 = e0 * fr.rin[1].x + e1 * fr.rin[1].y + e2 * fr.rin[1].z;
                const Float3 L2 = e0 * fr.rin[2].x + e1 * fr.rin[2].y + e2 * fr.rin[2].z;
                const Float det = dot(L0, cross(L1, L2));
                const Float3 trans = a0 - (L0 * fr.r0.x + L1 * fr.r0.y + L2 * fr.r0.z);
                // prev ← old curr unconditionally (collapsed tets freeze with
                // prev == curr, matching the CPU path's kept-matrix frame).
                transforms_prev.write(row.tlas_row, transforms.read(row.tlas_row));
                $if (abs(det) >= TetCageGeometry::kMinDet) {
                    Var<float4x4> cw = copy_world.read(row.copy_idx);
                    // world = copyWorld · [L|trans] (column-major compose)
                    transforms.write(row.tlas_row, make_float4x4(
                        cw * make_float4(L0, 0.0f),
                        cw * make_float4(L1, 0.0f),
                        cw * make_float4(L2, 0.0f),
                        cw * make_float4(trans, 1.0f)));
                };
            });
    }

    _built = true;
    // Register the solve rows now that every instance exists (rows resolve
    // through the slot map; registration validates the single-contiguous-run
    // v1 constraint). Failure (e.g. another shape wedged between cage copies)
    // keeps the CPU path.
    if (_gpuPathWanted && !register_solve_rows(pipeline, stream)) {
        _gpuPathWanted = false;
    }

    CI_LOG_I("TetCageGeometry: built - "
        << static_cast<uint>(_copyWorld.size()) << " copies x "
        << _pieceCount << " pieces = "
        << static_cast<uint>(_instanceIds.size())
        << " TLAS instances over " << _pieceCount << " shared piece BLASes"
        << (baked ? " (deformed pose baked into initial matrices)" : "")
        << (_gpuPathWanted ? " [GPU solve path]" : " [CPU solve path]"));
}

bool TetCageGeometry::register_solve_rows(core::Pipeline &pipeline,
                                          Stream &stream) noexcept {
    luisa::vector<ShapeId> ids;
    ids.reserve(_instanceIds.size());
    for (ShapeId id : _instanceIds)
        if (id != kInvalidShapeId) ids.push_back(id);
    if (!pipeline.registerGpuTransformRows(
            luisa::span<const ShapeId>{ids.data(), ids.size()}))
        return false;

    // Solve rows: one per valid (copy, tet). tlas_row re-resolved fresh —
    // a topology change (post-build add/remove) reshuffles dense rows.
    const uint tetCount = static_cast<uint>(_tets.size());
    const uint cageVerts = static_cast<uint>(_restCage.size());
    luisa::vector<uint> restIdxOf(tetCount, ~0u);
    uint restIdx = 0u;
    for (uint t = 0u; t < tetCount; ++t)
        if (_tets[t].tri_count != 0u) restIdxOf[t] = restIdx++;

    _solveRowsCPU.clear();
    _solveRowsCPU.reserve(ids.size());
    for (uint c = 0u; c < _copyWorld.size(); ++c) {
        for (uint t = 0u; t < tetCount; ++t) {
            if (_tets[t].tri_count == 0u) continue;
            ShapeId id = _instanceIds[c * tetCount + t];
            uint row = pipeline.geometryTlasRow(id);
            if (row == ~0u) {
                CI_LOG_E("TetCageGeometry: instance ShapeId " << id
                    << " lost its TLAS row mid-registration");
                return false;
            }
            TetCageSolveRow r{};
            for (int i = 0; i < 4; ++i) r.cage_verts[i] = _tets[t].cage_verts[i];
            r.anim_base = c * cageVerts;
            r.rest_idx = restIdxOf[t];
            r.copy_idx = c;
            r.tlas_row = row;
            _solveRowsCPU.push_back(r);
        }
    }
    if (!_solveRowBuffer ||
        _solveRowBuffer.size() != static_cast<uint>(_solveRowsCPU.size()))
        _solveRowBuffer = _device.create_buffer<TetCageSolveRow>(
            std::max(1u, static_cast<uint>(_solveRowsCPU.size())));
    stream << _solveRowBuffer.copy_from(_solveRowsCPU.data());

    _gpuTopologyGeneration = pipeline.topologyGeneration();
    _gpuPathActive = true;
    return true;
}

void TetCageGeometry::dispatch_tet_solve(core::Pipeline &pipeline) noexcept {
    Geometry *geom = pipeline.geometry();
    pipeline.computeStream() << _tetSolveShader(
        geom->instance_transform_buffer(),
        geom->instance_transform_prev_buffer(),
        _solveRowBuffer, _restFrameBuffer, _animCageBuffer, _copyWorldBuffer)
        .dispatch(static_cast<uint>(_solveRowsCPU.size()));
    pipeline.notifyGpuTransformsDirty();
}

//==============================================================================
// Deform dispatch (user shader animates the cage verts)
//==============================================================================

void TetCageGeometry::dispatch_deform(Stream &stream) noexcept {
    if (!_deformShader.is_set()) return;
    stream << core::ShaderManager::instance().shader(
        _deformShader, _animCageBuffer, _restCageBuffer, _cageStateBuffer)
        .dispatch(static_cast<uint>(_copyWorld.size()), _deformShaderBlock);
    // The CPU affine solve consumes the verts immediately — sync readback.
    stream << _animCageBuffer.copy_to(_animCage.data());
    stream << synchronize();
}

//==============================================================================
// Per-frame cage animation
//==============================================================================

void TetCageGeometry::update(core::Pipeline &pipeline) noexcept {
    if (!_built) return;
    // No deform shader: the rest pose IS the shape and it ships with the
    // initial instance upload — nothing to animate.
    if (!_deformShader.is_set()) { _lastUpdateMs = 0.0f; return; }

    const uint tetCount = static_cast<uint>(_tets.size());
    const auto t0 = std::chrono::steady_clock::now();
    util::CpuScopedTimer profileScope("TetCage/Update");

    Stream &stream = pipeline.computeStream();

    // 1. Upload dirty per-copy states (set_deform_state) — both paths. A
    // set_deform_state_all supersedes per-row dirt with one whole-buffer
    // upload (O(1) stream commands regardless of copy count).
    if (_allStatesDirty) {
        _allStatesDirty = false;
        _dirtyCageStates.clear();
        stream << _cageStateBuffer.copy_from(_cageStateCPU.data());
    } else {
        for (uint idx : _dirtyCageStates)
            stream << _cageStateBuffer.view(idx, 1u).copy_from(&_cageStateCPU[idx]);
        _dirtyCageStates.clear();
    }

    // 2. Topology watch (GPU path): a post-build add/remove reshuffles dense
    // TLAS rows — re-resolve + re-register before writing. Non-contiguous
    // (or any resolve failure) → permanent CPU fallback for this session.
    if (_gpuPathWanted &&
        (!_gpuPathActive || pipeline.topologyGeneration() != _gpuTopologyGeneration)) {
        if (!register_solve_rows(pipeline, stream)) {
            _gpuPathWanted = false;
            _gpuPathActive = false;
            CI_LOG_W("TetCageGeometry: GPU solve rows unavailable - falling "
                     "back to the CPU path (readback + solve + per-instance "
                     "setShapeTransform)");
        }
    }

    if (_gpuPathWanted) {
        // 3. GPU path: wind shader + TetSolve back to back on the compute
        // stream; matrices land in the instance-transform buffer and the
        // TLAS build copies them on the device. No readback, no sync, no
        // per-instance CPU work.
        stream << core::ShaderManager::instance().shader(
            _deformShader, _animCageBuffer, _restCageBuffer, _cageStateBuffer)
            .dispatch(static_cast<uint>(_copyWorld.size()), _deformShaderBlock);
        dispatch_tet_solve(pipeline);
    } else {
        // 3'. CPU path (baseline / fallback): wind dispatch + readback.
        dispatch_deform(stream);

        // 4'. Per-tet affine M = A · R⁻¹ (R⁻¹ precomputed), world = Tcopy · M.
        const uint cageVerts = static_cast<uint>(_restCage.size());
        for (uint c = 0u; c < _copyWorld.size(); ++c) {
            const float3 *anim = _animCage.data() + static_cast<size_t>(c) * cageVerts;
            for (uint t = 0u; t < tetCount; ++t) {
                ShapeId id = _instanceIds[c * tetCount + t];
                if (id == kInvalidShapeId) continue;
                const auto &tet = _tets[t];

                float3 a[4];
                for (int i = 0; i < 4; ++i)
                    a[i] = anim[tet.cage_verts[i]];
                float4x4 local;
                if (!tet_affine(tet, a, local))
                    continue; // collapsed tet: keep the previous matrix this frame
                pipeline.setShapeTransform(id, mul4(_copyWorld[c], local), Change::Affine);
            }
        }
    }

#if NT_PROFILING
    _lastUpdateMs = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // Optional update-cost log for perf runs: NT_TETCAGE_LOG_UPDATE=1
    // prints every 120th animated update (perf harness scenes run headless-ish;
    // the profiler window does not cover scene update).
    static const bool sLogUpdate = std::getenv("NT_TETCAGE_LOG_UPDATE") != nullptr;
    static uint32_t sLogTick = 0u;
    if (sLogUpdate && ++sLogTick % 120u == 0u)
        CI_LOG_I("TetCage/Update: " << tetCount << " tets x "
            << static_cast<uint>(_copyWorld.size()) << " copies -> " << _lastUpdateMs
            << " ms (" << (_gpuPathWanted ? "GPU dispatch" : "CPU solve") << ")");
#endif
}

void TetCageGeometry::settle(core::Pipeline &pipeline) noexcept {
    // CPU path needs nothing: Geometry's stop-frame prev←curr fix covers it.
    // GPU path: one TetSolve dispatch on the unchanged animated cage writes
    // prev = curr (the solve is idempotent), zeroing object motion.
    if (!_built || !_gpuPathActive) return;
    dispatch_tet_solve(pipeline);
}

//==============================================================================
// Runtime state
//==============================================================================

void TetCageGeometry::set_copy_visible(core::Pipeline &pipeline,
                                       uint copy, bool visible) noexcept {
    if (!_built || copy >= _copyWorld.size()) return;
    const uint tetCount = static_cast<uint>(_tets.size());
    for (uint t = 0u; t < tetCount; ++t) {
        ShapeId id = _instanceIds[copy * tetCount + t];
        if (id != kInvalidShapeId)
            pipeline.setShapeVisibility(id, visible);
    }
}

void TetCageGeometry::set_visible(core::Pipeline &pipeline, bool visible) noexcept {
    for (uint c = 0u; c < _copyWorld.size(); ++c)
        set_copy_visible(pipeline, c, visible);
}

void TetCageGeometry::set_material_layers(core::Pipeline &pipeline,
                                          uint32_t layers) noexcept {
    for (ShapeId id : _instanceIds)
        if (id != kInvalidShapeId)
            pipeline.setShapeMaterial(id, layers);
}

} // namespace newtype::scene
