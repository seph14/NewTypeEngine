#pragma once

// ============================================================================
// SHARC world-space radiance cache — LuisaCompute DSL port
// ============================================================================
//
// Ported from the NVIDIA SHARC SDK v1.8.3 headers (include/SharcCommon.h,
// SharcTypes.h, HashGridCommon.h, HashGridTypes.h at
// D:\Projects\RTX_study\SHARC). Every device-side function is a direct
// transcription; the upstream name is referenced in a comment so the port
// can be reviewed line-by-line against the source. Integration plan and
// feasibility notes: docs/sharc_rough_glass_plan.md.
//
// Ported configuration (fixed, see plan §2):
//   SHARC_ENABLE_SH_ENCODING         0   no SH2 directional encoding
//   SHARC_ENABLE_RESPONSIVE_LIGHTING 0
//   SHARC_MATERIAL_DEMODULATION      0   (=> demodulation factors are 1)
//   SHARC_SEPARATE_EMISSIVE          0
//   SHARC_ENABLE_FADE_ACCELERATION   0
//   SHARC_ENABLE_CACHE_RESAMPLING    1   => SHARC_PROPAGATION_DEPTH == 2
//   SHARC_BLEND_ADJACENT_LEVELS      1
//   SHARC_LINEAR_PROBE_WINDOW_SIZE   8
//   SHARC_RESAMPLING_DEPTH_MIN       1
//   SHARC_SAMPLE_NUM_THRESHOLD       0
//   HASH_GRID_USE_NORMALS            1
//   HASH_GRID_LIMIT_EMPTY_SLOTS      2   (SHARC's override, not the 0 default)
//   HASH_GRID_COMPACT                both key layouts ported (template arg)
//   HASH_GRID_ENABLE_64_BIT_ATOMICS  both insert routes ported (template arg)
//
// Deviations from upstream, all forced by the DSL/engine environment:
//   - fp16 storage is a hand-rolled round-to-nearest-even f32<->f16 bit
//     conversion into uints (upstream uses native float16_t4 which requires
//     DXC -enable-16bit-types; the DSL pipeline does not enable it).
//   - SharcPackedData keeps upstream's 16-byte shape but stores the
//     fp16 radiance+sampleNum quad in two packed uints.
//   - The accumulation buffer is a plain Buffer<uint4> (upstream struct
//     wrapping a single uint4) so per-component atomics map onto the DSL's
//     AtomicRef.
//   - Buffers ride in a SharcCache<> value bundle instead of SharcParameters.
//   - upstream rcp()/div-by-reciprocal is written as plain division (the DSL
//     has no exact-rcp; both sides of the parity harness use IEEE division).
//   - SHARC_UPDATE / SHARC_QUERY translation-unit modes are flattened: the
//     update-side, query-side and resolve-side entry points coexist here;
//     passes call the ones they need.
//   - The SHARC_PROPAGATION_DEPTH backprop loops are hand-unrolled for
//     depth == 2 (static_assert below); dynamic indexing into DSL value
//     arrays is not expressible.
//   - The lock-buffer insert route initializes the timed-out compare result
//     to a never-matching sentinel (upstream leaves the out value
//     uninitialized in that case).
//
// GPU/CPU parity of this port against a plain-C++ transcription of the same
// headers is regression-tested by tools/sharc_harness.cpp (the Phase 0 gate
// in the plan).

#include <luisa/luisa-compute.h>

#include "newtype/core/Config.h"

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Constants (SharcCommon.h / HashGridTypes.h, ported configuration)
//==============================================================================

inline constexpr uint kSharcBucketSize = 16u;              // HASH_GRID_HASH_MAP_BUCKET_SIZE
inline constexpr uint kSharcInvalidCacheIndex =            // HASH_GRID_INVALID_CACHE_INDEX
    0xFFFFFFFFu;
inline constexpr float kSharcPositionBias = 1e-4f;         // HASH_GRID_POSITION_BIAS
inline constexpr float kSharcNormalBias = 1e-3f;           // HASH_GRID_NORMAL_BIAS
inline constexpr uint kSharcLimitEmptySlots = 2u;          // HASH_GRID_LIMIT_EMPTY_SLOTS (SHARC override)

inline constexpr uint kSharcAccumFrameNumBitOffset = 0u;   // SHARC_ACCUMULATED_FRAME_NUM_*
inline constexpr uint kSharcAccumFrameNumBitNum = 16u;
inline constexpr uint kSharcAccumFrameNumBitMask = (1u << kSharcAccumFrameNumBitNum) - 1u;
inline constexpr uint kSharcStaleFrameNumBitOffset = 16u;  // SHARC_STALE_FRAME_NUM_*
inline constexpr uint kSharcStaleFrameNumBitNum = 16u;
inline constexpr uint kSharcStaleFrameNumBitMask = (1u << kSharcStaleFrameNumBitNum) - 1u;

inline constexpr float kSharcGridLogarithmBase = 2.0f;     // SHARC_GRID_LOGARITHM_BASE
inline constexpr uint kSharcAccumFrameNumMin = 1u;         // SHARC_ACCUMULATED_FRAME_NUM_MIN
inline constexpr uint kSharcAccumFrameNumMax = 1024u;      // SHARC_ACCUMULATED_FRAME_NUM_MAX
inline constexpr uint kSharcStaleFrameNumMin = 8u;         // SHARC_STALE_FRAME_NUM_MIN
inline constexpr uint kSharcStaleFrameNumMax = 1024u;      // SHARC_STALE_FRAME_NUM_MAX

inline constexpr uint kSharcSampleNumThreshold = 0u;       // SHARC_SAMPLE_NUM_THRESHOLD
inline constexpr uint kSharcLinearProbeWindowSize = 8u;    // SHARC_LINEAR_PROBE_WINDOW_SIZE
inline constexpr uint kSharcPropagationDepth = 2u;         // SHARC_PROPAGATION_DEPTH (resampling on)
inline constexpr uint kSharcResamplingDepthMin = 1u;       // SHARC_RESAMPLING_DEPTH_MIN
inline constexpr float kSharcFloat16Max = 65504.0f;

static_assert(kSharcPropagationDepth == 2u,
              "the backpropagation loops below are hand-unrolled for depth 2");

//==============================================================================
// Key layouts (HASH_GRID_COMPACT selects between these)
//==============================================================================

struct SharcLayoutFull {                                   // HASH_GRID_COMPACT == 0
    using KeyHost = luisa::ulong;
    static constexpr uint key_bits = 64u;
    static constexpr uint position_bits = 17u;             // HASH_GRID_POSITION_BIT_NUM
    static constexpr uint level_bits = 9u;                 // HASH_GRID_LEVEL_BIT_NUM
    static constexpr uint normal_bits = 3u;                // HASH_GRID_NORMAL_BIT_NUM
    static constexpr uint position_mask = (1u << position_bits) - 1u;
    static constexpr uint level_mask = (1u << level_bits) - 1u;
    static constexpr uint normal_mask = (1u << normal_bits) - 1u;
    static constexpr uint level_offset = position_bits * 3u;   // HASH_GRID_LEVEL_BIT_OFFSET
    static constexpr uint normal_offset = level_offset + level_bits;
};

struct SharcLayoutCompact {                                // HASH_GRID_COMPACT == 1
    using KeyHost = luisa::uint;
    static constexpr uint key_bits = 32u;
    static constexpr uint position_bits = 8u;
    static constexpr uint level_bits = 5u;
    static constexpr uint normal_bits = 3u;
    static constexpr uint position_mask = (1u << position_bits) - 1u;
    static constexpr uint level_mask = (1u << level_bits) - 1u;
    static constexpr uint normal_mask = (1u << normal_bits) - 1u;
    static constexpr uint level_offset = position_bits * 3u;
    static constexpr uint normal_offset = level_offset + level_bits;
};

//==============================================================================
// Hand-rolled fp16 (upstream SharcPackFloat16 / SharcUnpackFloat16 use native
// f32tof16 / f16tof32). Round-to-nearest-even; exact and branch-UB-free so
// the CPU transcription in the parity harness matches bit-for-bit.
//==============================================================================

[[nodiscard]] inline UInt sharc_pack_f16(Expr<float> value) noexcept {
    UInt bits = value.bitcast<uint>();
    UInt sign = (bits >> 16u) & 0x8000u;
    UInt exponent = (bits >> 23u) & 0xFFu;
    UInt mantissa = bits & 0x7FFFFFu;

    // Rebiased half exponent as a signed value: 1.m * 2^(e-15) in half terms.
    Int e = cast<Int>(exponent) - 112;

    // Inf / NaN keep their class (NaN payload's top bits preserved).
    UInt nan_or_inf = 0x7C00u | ite(mantissa != 0u, 0x0200u | (mantissa >> 13u), 0u);

    // Normal path: rebias, round-to-nearest-even from the low 13 mantissa
    // bits. Carrying into the exponent (including to Inf) falls out.
    UInt normal = (cast<UInt>(e) << 10u) | (mantissa >> 13u);
    UInt rem = mantissa & 0x1FFFu;
    Bool round_up = (rem > 0x1000u) | ((rem == 0x1000u) & ((normal & 1u) != 0u));
    normal = normal + ite(round_up, 1u, 0u);

    // Zero / subnormal path: restore the implicit one, place it into the
    // half subnormal scale (2^-24 LSB). e in [-10, 0] => shift in [14, 24];
    // e < -10 underflows to zero (handled by the select below). The clamp
    // keeps the unselected lanes' shifts well-defined on both CPU and GPU.
    UInt mantissa24 = mantissa | 0x800000u;
    UInt shift = min(cast<UInt>(14 - e), 24u);
    UInt sub = mantissa24 >> shift;
    UInt rem_sub = mantissa24 & ((1u << shift) - 1u);
    UInt half_ulp = 1u << (shift - 1u);
    Bool round_up_sub = (rem_sub > half_ulp) | ((rem_sub == half_ulp) & ((sub & 1u) != 0u));
    sub = sub + ite(round_up_sub, 1u, 0u);

    UInt result = ite(exponent == 255u, nan_or_inf,
                      ite(e >= 31, 0x7C00u,
                          ite(e <= 0, ite(e < -10, 0u, sub),
                              normal)));
    return sign | result;
}

[[nodiscard]] inline Float sharc_unpack_f16(Expr<uint> value) noexcept {
    UInt h = value & 0xFFFFu;
    UInt sign_bits = (h & 0x8000u) << 16u;
    UInt exponent = (h >> 10u) & 0x1Fu;
    UInt mantissa = h & 0x03FFu;

    // Normal / Inf / NaN by pure bit construction (rebias 15 -> 127).
    UInt normal_bits = ite(exponent >= 31u, 0x7F800000u | (mantissa << 13u),
                           ((exponent + 112u) << 23u) | (mantissa << 13u));
    Float normal_f = (sign_bits | normal_bits).bitcast<float>();

    // Subnormal / zero: man * 2^-24 is exactly representable in f32
    // (man <= 1023 scales by a power of two above FLT_MIN). Sign applied
    // arithmetically — negation is exact.
    Float sub = cast<Float>(mantissa) * 5.9604644775390625e-08f; // 2^-24
    sub = ite(sign_bits != 0u, -sub, sub);

    return ite(exponent == 0u, sub, normal_f);
}

[[nodiscard]] inline UInt sharc_pack_f16_pair(Expr<float2> value) noexcept {
    return sharc_pack_f16(value.x) | (sharc_pack_f16(value.y) << 16u);
}

[[nodiscard]] inline Float2 sharc_unpack_f16_pair(Expr<uint> value) noexcept {
    return make_float2(sharc_unpack_f16(value), sharc_unpack_f16(value >> 16u));
}

//==============================================================================
// Buffer element types (SharcTypes.h, no SH encoding)
//==============================================================================

// SharcPackedData: resolved radiance as an fp16 (r, g, b, sampleNum) quad in
// two packed uints + frame counters + the reserved SHARC sampleDataExt slot.
struct SharcPackedData {
    luisa::uint radiance_rg;
    luisa::uint radiance_bn;
    luisa::uint sample_data;    // accumulatedFrameNum | staleFrameNum << 16
    luisa::uint sample_data_ext;
};

}// namespace newtype::render

LUISA_STRUCT(newtype::render::SharcPackedData,
             radiance_rg, radiance_bn, sample_data, sample_data_ext) {};

namespace newtype::render {

//==============================================================================
// Grid parameters (DSL value bundles built from kernel arguments)
//==============================================================================

struct SharcGridParams {                   // HashGridParameters
    Float3 camera_position;                // world-space camera position
    Float log_base;                        // SHARC uses 2.0
    Float scene_scale;                     // world-space scale for voxel size
    Float level_bias;                      // LOD bias (start with 0)
};

struct SharcResolveParams {                // SharcResolveParameters (responsive off)
    Float3 camera_position_prev;
    UInt accumulation_frame_num;           // temporal window
    UInt stale_frame_num_max;              // eviction threshold
};

//==============================================================================
// Cache storage bundle (HashGridData + the scalar part of SharcParameters).
// The insert route is selected by the bundle type: SharcCache uses native
// compare-exchange on the key type (HASH_GRID_ENABLE_64_BIT_ATOMICS == 1;
// 64-bit keys need SM6.6 — call set_warp_size() in kernels using that
// route); SharcLockCache uses upstream's lock-buffer route.
//==============================================================================

template<typename Layout>
struct SharcCache {
    using LayoutT = Layout;
    using KeyHost = typename Layout::KeyHost;
    static constexpr bool uses_lock_buffer = false;

    BufferVar<KeyHost> entries;            // hashEntriesBuffer
    BufferVar<luisa::uint4> accumulation;  // accumulationBuffer (uint4 data)
    BufferVar<SharcPackedData> resolved;   // resolvedBuffer
    UInt capacity;
    Float radiance_scale;                  // atomic accumulation quantization (start 1e3)
};

// Lock-buffer route (upstream: !HASH_GRID_ENABLE_64_BIT_ATOMICS && !COMPACT).
// Standalone rather than derived so both bundles aggregate-initialize the
// same way — deriving forces a copy of the (move-only) BufferVar members.
template<typename Layout>
struct SharcLockCache {
    static_assert(Layout::key_bits == 64u,
                  "upstream provides the lock-buffer route for 64-bit keys only");
    using LayoutT = Layout;
    using KeyHost = typename Layout::KeyHost;
    static constexpr bool uses_lock_buffer = true;

    BufferVar<KeyHost> entries;            // hashEntriesBuffer
    BufferVar<luisa::uint> locks;          // lockBuffer
    BufferVar<luisa::uint4> accumulation;  // accumulationBuffer (uint4 data)
    BufferVar<SharcPackedData> resolved;   // resolvedBuffer
    UInt capacity;
    Float radiance_scale;                  // atomic accumulation quantization (start 1e3)
};

// Read-only query-side view (Phase 2): binds only the two buffers the query
// path touches. Satisfies the sharc_find / sharc_get_voxel_data /
// sharc_get_cached_radiance template requirements, so query consumers (GI
// initial) share the exact code paths the insert/resolve parity harness
// validates — no duplicated find/read logic.
template<typename Layout>
struct SharcQueryCache {
    using LayoutT = Layout;
    using KeyHost = typename Layout::KeyHost;
    static constexpr bool uses_lock_buffer = false;

    BufferVar<KeyHost> entries;            // hashEntriesBuffer
    BufferVar<SharcPackedData> resolved;   // resolvedBuffer
    UInt capacity;
};

//==============================================================================
// Hash grid math (HashGridCommon.h)
//==============================================================================

// HashGrid_LogBase — log with an arbitrary base.
[[nodiscard]] inline Float sharc_log_base(Float x, Float base) noexcept {
    return log2(x) / log2(base);
}

// HashGrid_HashJenkins32 — http://burtleburtle.net/bob/hash/integer.html
[[nodiscard]] inline UInt sharc_hash_jenkins32(UInt a) noexcept {
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
}

// HashGrid_Hash32
template<typename Layout>
[[nodiscard]] inline UInt sharc_hash32(Expr<typename Layout::KeyHost> hash_key) noexcept {
    if constexpr (Layout::key_bits == 64u) {
        return sharc_hash_jenkins32(cast<uint>(hash_key)) ^
               sharc_hash_jenkins32(cast<uint>(hash_key >> 32u));
    } else {
        return sharc_hash_jenkins32(hash_key);
    }
}

// HashGrid_GetBaseSlot — buckets stay contiguous and do not wrap; allocate
// BUCKET_SIZE - 1 extra slots past capacity to avoid clamping near the end.
template<typename Layout>
[[nodiscard]] inline UInt sharc_get_base_slot(Expr<typename Layout::KeyHost> hash_key,
                                              Expr<uint> capacity) noexcept {
    UInt base_slot_count = capacity - kSharcBucketSize + 1u;
    return sharc_hash32<Layout>(hash_key) % base_slot_count;
}

// HashGrid_GetLevel — logarithmic LOD from distance to camera.
template<typename Layout>
[[nodiscard]] inline UInt sharc_get_level(Float3 sample_position,
                                          const SharcGridParams &grid) noexcept {
    Float3 camera_offset = grid.camera_position - sample_position;
    Float distance2 = dot(camera_offset, camera_offset);
    distance2 = max(distance2, 1e-10f);
    Float grid_level = 0.5f * sharc_log_base(distance2, grid.log_base) + grid.level_bias;
    return cast<UInt>(clamp(grid_level, 1.0f, static_cast<float>(Layout::level_mask)));
}

// HashGrid_GetVoxelSize
[[nodiscard]] inline Float sharc_get_voxel_size(Expr<uint> grid_level,
                                                const SharcGridParams &grid) noexcept {
    Float exponent = log2(grid.log_base) * (cast<Float>(grid_level) - grid.level_bias);
    return exp2(exponent) / grid.scene_scale;
}

// HashGrid_CalculatePositionLogWithVoxelSize
template<typename Layout>
[[nodiscard]] inline Int4 sharc_calculate_position_log(Float3 sample_position,
                                                       const SharcGridParams &grid,
                                                       Float &voxel_size) noexcept {
    sample_position = sample_position + make_float3(kSharcPositionBias);

    UInt grid_level = sharc_get_level<Layout>(sample_position, grid);
    voxel_size = sharc_get_voxel_size(grid_level, grid);
    Int3 grid_position = cast<Int3>(floor(sample_position / voxel_size));
    return make_int4(grid_position, cast<Int>(grid_level));
}

// HashGrid_ComputeSpatialHashFromGridPosition (HASH_GRID_USE_NORMALS == 1)
template<typename Layout>
[[nodiscard]] inline auto sharc_key_from_grid_position(UInt4 grid_position,
                                                       Float3 sample_normal) noexcept {
    using K = typename Layout::KeyHost;
    auto key = cast<K>(grid_position.x & Layout::position_mask)
             | ((cast<K>(grid_position.y & Layout::position_mask)) << (Layout::position_bits * 1u))
             | ((cast<K>(grid_position.z & Layout::position_mask)) << (Layout::position_bits * 2u))
             | ((cast<K>(grid_position.w & Layout::level_mask)) << Layout::level_offset);

    UInt normal_bits = ite(sample_normal.x + kSharcNormalBias >= 0.0f, 0u, 1u)
                     + ite(sample_normal.y + kSharcNormalBias >= 0.0f, 0u, 2u)
                     + ite(sample_normal.z + kSharcNormalBias >= 0.0f, 0u, 4u);
    key = key | (cast<K>(normal_bits) << Layout::normal_offset);
    return key;
}

// HashGrid_ComputeSpatialHashWithVoxelSize
template<typename Layout>
[[nodiscard]] inline auto sharc_compute_spatial_hash(Float3 sample_position, Float3 sample_normal,
                                                     const SharcGridParams &grid,
                                                     Float &voxel_size) noexcept {
    Int4 pos_log = sharc_calculate_position_log<Layout>(sample_position, grid, voxel_size);
    return sharc_key_from_grid_position<Layout>(cast<UInt4>(pos_log), sample_normal);
}

// HashGrid_ComputeSpatialHash
template<typename Layout>
[[nodiscard]] inline auto sharc_compute_spatial_hash(Float3 sample_position, Float3 sample_normal,
                                                     const SharcGridParams &grid) noexcept {
    Float voxel_size;
    return sharc_compute_spatial_hash<Layout>(sample_position, sample_normal, grid, voxel_size);
}

// HashGrid_GetPositionFromKey — voxel-center position of a key.
template<typename Layout>
[[nodiscard]] inline Float3 sharc_get_position_from_key(Expr<typename Layout::KeyHost> hash_key,
                                                        const SharcGridParams &grid) noexcept {
    auto pos_mask = def<typename Layout::KeyHost>(
        static_cast<typename Layout::KeyHost>(Layout::position_mask));
    auto level_mask = def<typename Layout::KeyHost>(
        static_cast<typename Layout::KeyHost>(Layout::level_mask));
    Int3 grid_position;
    grid_position.x = cast<Int>((hash_key >> (Layout::position_bits * 0u)) & pos_mask);
    grid_position.y = cast<Int>((hash_key >> (Layout::position_bits * 1u)) & pos_mask);
    grid_position.z = cast<Int>((hash_key >> (Layout::position_bits * 2u)) & pos_mask);
    // Sign-extend packed coordinates without divergent branches.
    grid_position = (grid_position << (32u - Layout::position_bits)) >> (32u - Layout::position_bits);

    UInt grid_level = cast<UInt>((hash_key >> Layout::level_offset) & level_mask);
    Float voxel_size = sharc_get_voxel_size(grid_level, grid);
    return (cast<Float3>(grid_position) + 0.5f) * voxel_size;
}

//==============================================================================
// Insert / find (HashGridCommon.h). Upstream's early returns are restructured
// into $break / $if with identical control flow.
//==============================================================================

// HashGrid_AtomicCompareExchange — returns the original value.
// Lock-buffer route (HASH_GRID_ENABLE_64_BIT_ATOMICS == 0): acquire the slot
// lock with an atomic exchange, emulate the compare-exchange with guarded
// read/write, restore the lock. "ANY rearrangements to the code below lead
// to device hang if fuse is unlimited" (upstream comment) — keep the shape.
template<typename Cache>
[[nodiscard]] inline auto sharc_atomic_compare_exchange_key(const Cache &cache, UInt dst_offset,
                                                            Expr<typename Cache::KeyHost> compare_value,
                                                            Expr<typename Cache::KeyHost> value) noexcept {
    using K = typename Cache::KeyHost;
    constexpr uint c_lock = 0xAAAAAAAAu;
    constexpr uint fuse_length = 8u;
    if constexpr (Cache::uses_lock_buffer) {
        // Sentinel for the lock-timeout path (upstream leaves the out value
        // uninitialized there); chosen to never match INVALID(0) or a real key.
        auto original_value = def<K>(static_cast<K>(0xCCCCCCCCCCCCCCCCull));
        UInt fuse = 0u;
        Bool busy = def(true);
        $while(busy & (fuse < fuse_length)) {
            UInt state = cache.locks.atomic(dst_offset).exchange(c_lock);
            busy = state != 0u;
            $if(state != c_lock) {
                original_value = cache.entries.read(dst_offset);
                $if(original_value == compare_value) {
                    cache.entries.write(dst_offset, value);
                };
                cache.locks.atomic(dst_offset).exchange(state);
                fuse = fuse_length;
            };
            fuse = fuse + 1u;
        };
        return original_value;
    } else {
        return cache.entries.atomic(dst_offset).compare_exchange(compare_value, value);
    }
}

// HashGrid_Insert — CAS INVALID->hashKey across the probe bucket; succeeds on
// the first empty or equal-key slot.
template<typename Cache>
inline Bool sharc_insert(const Cache &cache, Expr<typename Cache::KeyHost> hash_key,
                         UInt base_slot, UInt probe_range,
                         UInt &cache_index, UInt &bucket_offset) noexcept {
    using K = typename Cache::KeyHost;
    auto invalid = def<K>(static_cast<K>(0));
    cache_index = kSharcInvalidCacheIndex;
    probe_range = min(probe_range, kSharcBucketSize);

    Bool inserted = def(false);
    $for(offset, 0u, probe_range) {
        bucket_offset = offset;
        auto prev_key = sharc_atomic_compare_exchange_key(cache, base_slot + offset,
                                                          invalid, hash_key);
        $if((prev_key == invalid) | (prev_key == hash_key)) {
            cache_index = base_slot + offset;
            inserted = true;
            $break;
        };
    };
    return inserted;
}

// HashGrid_Find — linear probe with early stop after LIMIT_EMPTY_SLOTS + 1
// empty slots.
template<typename Cache>
inline Bool sharc_find(const Cache &cache, Expr<typename Cache::KeyHost> hash_key,
                       UInt base_slot, UInt probe_range,
                       UInt &cache_index, UInt &bucket_offset) noexcept {
    using K = typename Cache::KeyHost;
    auto invalid = def<K>(static_cast<K>(0));
    cache_index = kSharcInvalidCacheIndex;
    probe_range = min(probe_range, kSharcBucketSize);
    UInt empty_entries = 0u;

    Bool found = def(false);
    $for(offset, 0u, probe_range) {
        bucket_offset = offset;
        auto stored_key = cache.entries.read(base_slot + offset);
        // HASH_GRID_LIMIT_EMPTY_SLOTS == 2
        $if(stored_key == invalid) {
            $if(empty_entries > kSharcLimitEmptySlots) {
                $break;
            };
            empty_entries = empty_entries + 1u;
        };
        $if(stored_key == hash_key) {
            cache_index = base_slot + offset;
            found = true;
            $break;
        };
    };
    return found;
}

// HashGrid_InsertEntry
template<typename Cache>
inline Bool sharc_insert_entry(const Cache &cache, const SharcGridParams &grid,
                               Float3 sample_position, Float3 sample_normal,
                               Var<typename Cache::KeyHost> &hash_key, UInt &cache_index) noexcept {
    Float voxel_size;
    hash_key = sharc_compute_spatial_hash<typename Cache::LayoutT>(sample_position, sample_normal,
                                                                    grid, voxel_size);
    UInt base_slot = sharc_get_base_slot<typename Cache::LayoutT>(hash_key, cache.capacity);
    UInt bucket_offset;
    return sharc_insert(cache, hash_key, base_slot, kSharcBucketSize, cache_index, bucket_offset);
}

// HashGrid_FindEntry
template<typename Cache>
[[nodiscard]] inline UInt sharc_find_entry(const Cache &cache, const SharcGridParams &grid,
                                           Float3 sample_position, Float3 sample_normal,
                                           Var<typename Cache::KeyHost> &hash_key) noexcept {
    hash_key = sharc_compute_spatial_hash<typename Cache::LayoutT>(sample_position, sample_normal, grid);
    UInt base_slot = sharc_get_base_slot<typename Cache::LayoutT>(hash_key, cache.capacity);
    UInt cache_index;
    UInt bucket_offset;
    sharc_find(cache, hash_key, base_slot, kSharcBucketSize, cache_index, bucket_offset);
    return cache_index;
}

//==============================================================================
// Packed / accumulated data access (SharcCommon.h, no SH encoding)
//==============================================================================

// SharcVoxelData — decoded view of a resolved entry.
struct SharcVoxelData {
    Float3 radiance;
    Float sample_num;
    UInt accumulated_frame_num;
    UInt stale_frame_num;
    UInt sample_data_ext;
};

// SharcZeroPackedData
[[nodiscard]] inline Var<SharcPackedData> sharc_zero_packed_data() noexcept {
    Var<SharcPackedData> packed;
    packed.radiance_rg = 0u;
    packed.radiance_bn = 0u;
    packed.sample_data = 0u;
    packed.sample_data_ext = 0u;
    return packed;
}

// SharcUnpackVoxelData
[[nodiscard]] inline SharcVoxelData sharc_unpack_voxel_data(
    const Var<SharcPackedData> &packed) noexcept {
    Float2 rg = sharc_unpack_f16_pair(packed.radiance_rg);
    Float2 bn = sharc_unpack_f16_pair(packed.radiance_bn);
    SharcVoxelData voxel;
    voxel.radiance = make_float3(rg.x, rg.y, bn.x);
    voxel.sample_num = bn.y;
    voxel.accumulated_frame_num =
        (packed.sample_data >> kSharcAccumFrameNumBitOffset) & kSharcAccumFrameNumBitMask;
    voxel.stale_frame_num =
        (packed.sample_data >> kSharcStaleFrameNumBitOffset) & kSharcStaleFrameNumBitMask;
    voxel.sample_data_ext = packed.sample_data_ext;
    return voxel;
}

// SharcPackVoxelData — radiance and sample count are clamped to fp16 range.
[[nodiscard]] inline Var<SharcPackedData> sharc_pack_voxel_data(
    Float3 radiance, Float sample_num, UInt accumulated_frame_num,
    UInt stale_frame_num, UInt sample_data_ext) noexcept {
    Float4 radiance_and_num = min(make_float4(radiance, sample_num),
                                  make_float4(kSharcFloat16Max));
    Var<SharcPackedData> packed;
    packed.radiance_rg = sharc_pack_f16_pair(make_float2(radiance_and_num.x, radiance_and_num.y));
    packed.radiance_bn = sharc_pack_f16_pair(make_float2(radiance_and_num.z, radiance_and_num.w));
    packed.sample_data = accumulated_frame_num | (stale_frame_num << kSharcStaleFrameNumBitOffset);
    packed.sample_data_ext = sample_data_ext;
    return packed;
}

// SharcGetVoxelData
template<typename Cache>
[[nodiscard]] inline SharcVoxelData sharc_get_voxel_data(const Cache &cache,
                                                         UInt hash_grid_index) noexcept {
    Var<SharcPackedData> packed = cache.resolved.read(hash_grid_index);
    return sharc_unpack_voxel_data(packed);
}

//==============================================================================
// Update path (SHARC_UPDATE == 1)
//==============================================================================

// SharcState — SHARC_PROPAGATION_DEPTH == 2 vertices.
struct SharcState {
    UInt cache_indices[kSharcPropagationDepth];
    Float3 sample_weights[kSharcPropagationDepth];
    UInt path_length;
};

// SharcInit
inline void sharc_init(SharcState &state) noexcept {
    state.path_length = 0u;
}

// SharcAddVoxelData — quantized atomic accumulation.
template<typename Cache>
inline void sharc_add_voxel_data(const Cache &cache, UInt hash_grid_index,
                                 Float3 sample_value, Float3 sample_weight,
                                 UInt sample_data) noexcept {
    UInt3 scaled_radiance = cast<UInt3>(sample_value * sample_weight * cache.radiance_scale);
    $if(scaled_radiance.x != 0u) {
        cache.accumulation.atomic(hash_grid_index).x.fetch_add(scaled_radiance.x);
    };
    $if(scaled_radiance.y != 0u) {
        cache.accumulation.atomic(hash_grid_index).y.fetch_add(scaled_radiance.y);
    };
    $if(scaled_radiance.z != 0u) {
        cache.accumulation.atomic(hash_grid_index).z.fetch_add(scaled_radiance.z);
    };
    $if(sample_data != 0u) {
        cache.accumulation.atomic(hash_grid_index).w.fetch_add(sample_data);
    };
}

// SharcSetThroughput — folds the segment throughput into all stored vertices.
inline void sharc_set_throughput(SharcState &state, Float3 throughput) noexcept {
    // upstream: for (i < pathLength) sampleWeights[i] *= throughput
    $if(state.path_length > 0u) {
        state.sample_weights[0] = state.sample_weights[0] * throughput;
    };
    $if(state.path_length > 1u) {
        state.sample_weights[1] = state.sample_weights[1] * throughput;
    };
}

// SharcUpdateMiss — environment radiance back-propagated to stored vertices.
template<typename Cache>
inline void sharc_update_miss(const Cache &cache, const SharcState &state,
                              Float3 radiance) noexcept {
    // upstream: for (i < pathLength) AddVoxelData(indices[i], radiance,
    //            sampleWeights[i], ..., isNewSample ? 1 : 0 == 0)
    $if(state.path_length > 0u) {
        sharc_add_voxel_data(cache, state.cache_indices[0], radiance,
                             state.sample_weights[0], 0u);
    };
    $if(state.path_length > 1u) {
        sharc_add_voxel_data(cache, state.cache_indices[1], radiance,
                             state.sample_weights[1], 0u);
    };
}

// SharcUpdateHit — insert/find the entry, add fresh direct lighting, and
// back-propagate the (possibly cache-resampled) radiance to stored vertices.
// Returns whether the path should keep tracing (false on failed insert or
// successful resample).
template<typename Cache>
inline Bool sharc_update_hit(const Cache &cache, const SharcGridParams &grid,
                             SharcState &state, Float3 position_world, Float3 normal_world,
                             Float3 direct_lighting, Float random) noexcept {
    using K = typename Cache::KeyHost;
    auto hash_key = def<K>(static_cast<K>(0));
    UInt cache_index = kSharcInvalidCacheIndex;
    Bool inserted = sharc_insert_entry(cache, grid, position_world, normal_world,
                                       hash_key, cache_index);

    Bool continue_tracing = def(true);
    $if(inserted) {
        Float3 sharc_radiance = direct_lighting;

        // SHARC_ENABLE_CACHE_RESAMPLING == 1:
        // resamplingDepth = uint(round(lerp(RESAMPLING_DEPTH_MIN,
        //                                   PROPAGATION_DEPTH, random)))
        UInt resampling_depth = cast<UInt>(round(
            cast<Float>(kSharcResamplingDepthMin) +
            random * static_cast<float>(kSharcPropagationDepth - kSharcResamplingDepthMin)));
        $if(resampling_depth <= state.path_length) {
            SharcVoxelData voxel = sharc_get_voxel_data(cache, cache_index);
            $if(voxel.sample_num > static_cast<float>(kSharcSampleNumThreshold)) {
                sharc_radiance = voxel.radiance; // SharcDecodeRadiance is identity (no SH)
                continue_tracing = false;        // demodulation is 1, so no rescale
            };
        };

        $if(continue_tracing) {
            sharc_add_voxel_data(cache, cache_index, direct_lighting,
                                 make_float3(1.0f), 1u);
        };

        // upstream: for (i < pathLength) AddVoxelData(indices[i], sharcRadiance,
        //            sampleWeights[i], ..., 0)
        $if(state.path_length > 0u & state.cache_indices[0] != kSharcInvalidCacheIndex) {
            sharc_add_voxel_data(cache, state.cache_indices[0], sharc_radiance,
                                 state.sample_weights[0], 0u);
        };
        $if(state.path_length > 1u & state.cache_indices[1] != kSharcInvalidCacheIndex) {
            sharc_add_voxel_data(cache, state.cache_indices[1], sharc_radiance,
                                 state.sample_weights[1], 0u);
        };

        // upstream: shift vertices back (single copy at DEPTH == 2)
        state.cache_indices[1] = state.cache_indices[0];
        state.sample_weights[1] = state.sample_weights[0];
        state.cache_indices[0] = cache_index;
        state.sample_weights[0] = make_float3(1.0f); // inverseMaterialDemodulation
        state.path_length = min(state.path_length + 1u, kSharcPropagationDepth);
    };
    // upstream returns false immediately on failed insert, else continueTracing
    return inserted & continue_tracing;
}

//==============================================================================
// Query path (SHARC_QUERY == 1)
//==============================================================================

// SharcGetCachedRadianceFromHash + SharcGetCachedRadiance (no responsive /
// demodulation / separate-emissive tail).
template<typename Cache>
inline Bool sharc_get_cached_radiance(const Cache &cache, const SharcGridParams &grid,
                                      Float3 position_world, Float3 normal_world,
                                      Float3 &radiance) noexcept {
    using K = typename Cache::KeyHost;
    auto hash_key = sharc_compute_spatial_hash<typename Cache::LayoutT>(position_world,
                                                                        normal_world, grid);
    UInt base_slot = sharc_get_base_slot<typename Cache::LayoutT>(hash_key, cache.capacity);
    UInt cache_index;
    UInt bucket_offset;
    Bool found = sharc_find(cache, hash_key, base_slot, kSharcBucketSize,
                            cache_index, bucket_offset);

    Bool valid = def(false);
    radiance = def(make_float3(0.0f));
    $if(found) {
        SharcVoxelData voxel = sharc_get_voxel_data(cache, cache_index);
        $if(voxel.sample_num > static_cast<float>(kSharcSampleNumThreshold)) {
            radiance = voxel.radiance;
            valid = true;
        };
    };
    return valid;
}

// Query eligibility gates (plan §7 Phase 2; SHARC Integration.md "SHaRC
// Render"): the x1->x2 segment must outlast a voxel diagonal, and the cone
// spread of the lobe that launched it — 2*t*sqrt(0.5*a^2/(1-a^2)) with
// a = roughness^2 (Integration.md's estimate) — must exceed the voxel size,
// so the cache is only queried where its low-pass content covers the lobe.
// Roughness 1 saturates the spread (diffuse passes by construction);
// roughness ~0 yields ~0 (mirrors keep their exact reflection). The 1e-6
// denominator clamp only fires at a^2 >= 1-1e-6, where the outcome is
// pass-either-way, and keeps the division finite on GPU.
[[nodiscard]] inline Bool sharc_query_eligible(Expr<float> segment_length,
                                               Expr<float> launch_roughness,
                                               Expr<uint> x2_level,
                                               const SharcGridParams &grid) noexcept {
    Float voxel_size = sharc_get_voxel_size(x2_level, grid);
    Bool segment_ok = segment_length > voxel_size * 1.7320508f; // sqrt(3)
    Float alpha_sq = launch_roughness * launch_roughness;
    alpha_sq = alpha_sq * alpha_sq;                       // a^2, a = roughness^2
    Float spread = 2.0f * segment_length *
                   sqrt(0.5f * alpha_sq / max(1.0f - alpha_sq, 1e-6f));
    return segment_ok & (spread > voxel_size);
}

//==============================================================================
// Resolve path (SharcResolveEntry + adjacent-level blending helpers)
//==============================================================================

// SharcGetAdjacentLevelHashKey — re-key an entry at the neighboring LOD in
// the direction the camera moved.
template<typename Layout>
[[nodiscard]] inline auto sharc_get_adjacent_level_hash_key(
    Expr<typename Layout::KeyHost> hash_key, const SharcGridParams &grid,
    Float3 camera_position_prev) noexcept {
    using K = typename Layout::KeyHost;
    auto pos_mask = def<K>(static_cast<K>(Layout::position_mask));
    auto level_mask = def<K>(static_cast<K>(Layout::level_mask));
    Int3 grid_position;
    grid_position.x = cast<Int>((hash_key >> (Layout::position_bits * 0u)) & pos_mask);
    grid_position.y = cast<Int>((hash_key >> (Layout::position_bits * 1u)) & pos_mask);
    grid_position.z = cast<Int>((hash_key >> (Layout::position_bits * 2u)) & pos_mask);
    // Sign-extend packed coordinates without divergent branches.
    grid_position = (grid_position << (32u - Layout::position_bits)) >> (32u - Layout::position_bits);

    Int level = cast<Int>((hash_key >> Layout::level_offset) & level_mask);

    Float voxel_size = sharc_get_voxel_size(cast<UInt>(level), grid);
    Float inverse_voxel_size = 1.0f / voxel_size;
    Int3 camera_grid_position = cast<Int3>(floor(grid.camera_position * inverse_voxel_size));
    Int3 camera_vector = camera_grid_position - grid_position;
    Int camera_distance = camera_vector.x * camera_vector.x +
                          camera_vector.y * camera_vector.y +
                          camera_vector.z * camera_vector.z;

    Int3 camera_grid_position_prev = cast<Int3>(floor(camera_position_prev * inverse_voxel_size));
    Int3 camera_vector_prev = camera_grid_position_prev - grid_position;
    Int camera_distance_prev = camera_vector_prev.x * camera_vector_prev.x +
                               camera_vector_prev.y * camera_vector_prev.y +
                               camera_vector_prev.z * camera_vector_prev.z;

    $if(camera_distance < camera_distance_prev) {
        grid_position = cast<Int3>(floor(cast<Float3>(grid_position) / grid.log_base));
        level = min(level + 1, cast<Int>(Layout::level_mask));
    } $else { // "this may be inaccurate" (upstream)
        grid_position = cast<Int3>(floor(cast<Float3>(grid_position) * grid.log_base));
        level = max(level - 1, 1);
    };

    auto modified_key = cast<K>(cast<UInt>(grid_position.x) & Layout::position_mask)
                      | ((cast<K>(cast<UInt>(grid_position.y) & Layout::position_mask)) << (Layout::position_bits * 1u))
                      | ((cast<K>(cast<UInt>(grid_position.z) & Layout::position_mask)) << (Layout::position_bits * 2u))
                      | ((cast<K>(cast<UInt>(level) & Layout::level_mask)) << Layout::level_offset);
    // HASH_GRID_USE_NORMALS == 1: carry the normal bits over.
    modified_key = modified_key |
                   (hash_key & (cast<K>(Layout::normal_mask) << Layout::normal_offset));
    return modified_key;
}

// SharcResolveEntry — one thread per entry. Combines per-frame accumulation
// with previously resolved data (EWMA), recovers data displaced by insertion
// collisions via the linear probe window, blends adjacent levels on camera
// motion, evicts stale entries, and clears accumulation for the next frame.
template<typename Cache>
inline void sharc_resolve_entry(const Cache &cache, const SharcGridParams &grid,
                                const SharcResolveParams &resolve, UInt entry_index) noexcept {
    using K = typename Cache::KeyHost;
    auto invalid = def<K>(static_cast<K>(0));

    $if(entry_index < cache.capacity) {
        auto hash_key = cache.entries.read(entry_index);
        $if(hash_key != invalid) {
            Var<luisa::uint4> accumulated = cache.accumulation.read(entry_index);
            Var<SharcPackedData> resolved = cache.resolved.read(entry_index);
            SharcVoxelData voxel = sharc_unpack_voxel_data(resolved);

            Float sample_num = cast<Float>(accumulated.w); // SharcGetAccumulatedSampleNum
            Float sample_num_prev = voxel.sample_num;
            UInt accumulated_frame_num = voxel.accumulated_frame_num + 1u;
            UInt stale_frame_num = voxel.stale_frame_num;

            stale_frame_num = ite(sample_num != 0.0f, 0u, stale_frame_num + 1u);
            UInt stale_frame_num_max = clamp(resolve.stale_frame_num_max,
                                             kSharcStaleFrameNumMin, kSharcStaleFrameNumMax);

            Bool is_valid_element = stale_frame_num < stale_frame_num_max;
            $if(!is_valid_element) {
                // Evict: clear all three buffers at this entry.
                cache.entries.write(entry_index, invalid);
                cache.accumulation.write(entry_index, make_uint4(0u));
                cache.resolved.write(entry_index, sharc_zero_packed_data());
            } $else {
                $if(sample_num == 0.0f) {
                    // No new samples: bump both 16-bit frame counters.
                    Var<SharcPackedData> bumped = resolved;
                    bumped.sample_data = resolved.sample_data +
                        ((1u << kSharcAccumFrameNumBitOffset) | (1u << kSharcStaleFrameNumBitOffset));
                    cache.resolved.write(entry_index, bumped);
                } $else {
                    // Recover previous data when previous insertions collided
                    // into a different slot (fixed-size linear probe window).
                    $if(sample_num_prev == 0.0f) {
                        UInt search_end = min(entry_index + 1u + kSharcLinearProbeWindowSize,
                                              cache.capacity);
                        $for(i, entry_index + 1u, search_end) {
                            auto hash_key_old = cache.entries.read(i);
                            $if(hash_key_old == hash_key) {
                                resolved = cache.resolved.read(i);
                                voxel = sharc_unpack_voxel_data(resolved);
                                sample_num_prev = voxel.sample_num;
                                accumulated_frame_num = voxel.accumulated_frame_num + 1u;
                                stale_frame_num = 0u;
                                $break;
                            };
                        };
                    };

                    // SharcGetAccumulatedRadianceData
                    Float3 accumulated_radiance = make_float3(cast<Float>(accumulated.x),
                                                              cast<Float>(accumulated.y),
                                                              cast<Float>(accumulated.z)) /
                                                  (cache.radiance_scale * max(sample_num, 1e-6f));
                    Float3 accumulated_radiance_prev = voxel.radiance;

                    UInt accumulation_frame_num = clamp(resolve.accumulation_frame_num,
                                                        kSharcAccumFrameNumMin,
                                                        kSharcAccumFrameNumMax);
                    $if(accumulated_frame_num > accumulation_frame_num) {
                        Float normalization_scale =
                            cast<Float>(accumulated_frame_num) / cast<Float>(accumulation_frame_num);
                        accumulated_frame_num = accumulation_frame_num;
                        sample_num_prev = sample_num_prev * normalization_scale;
                    };

                    Float sample_total_inv = 1.0f / (sample_num_prev + sample_num);
                    Float3 blended = accumulated_radiance_prev * (sample_num_prev * sample_total_inv) +
                                     accumulated_radiance * (sample_num * sample_total_inv);
                    Float accumulated_sample_num = sample_num_prev + sample_num;

                    // SHARC_BLEND_ADJACENT_LEVELS == 1 — reproject from the
                    // adjacent level on camera movement (young entries only).
                    Float3 camera_offset = grid.camera_position - resolve.camera_position_prev;
                    $if((dot(camera_offset, camera_offset) > 1e-6f) &
                        (accumulated_frame_num <= 2u)) {
                        auto adjacent_key = sharc_get_adjacent_level_hash_key<typename Cache::LayoutT>(
                            hash_key, grid, resolve.camera_position_prev);
                        UInt adjacent_base_slot = sharc_get_base_slot<typename Cache::LayoutT>(
                            adjacent_key, cache.capacity);
                        UInt adjacent_index;
                        UInt adjacent_bucket_offset;
                        Bool adjacent_found = sharc_find(cache, adjacent_key, adjacent_base_slot,
                                                         kSharcBucketSize, adjacent_index,
                                                         adjacent_bucket_offset);
                        $if(adjacent_found) {
                            Var<SharcPackedData> adjacent_packed = cache.resolved.read(adjacent_index);
                            SharcVoxelData adjacent_voxel = sharc_unpack_voxel_data(adjacent_packed);
                            Float adjacent_sample_num = adjacent_voxel.sample_num;
                            $if(adjacent_sample_num > static_cast<float>(kSharcSampleNumThreshold)) {
                                Float blend_weight = 1.0f / (adjacent_sample_num + accumulated_sample_num);
                                blended = adjacent_voxel.radiance * (adjacent_sample_num * blend_weight) +
                                          blended * (accumulated_sample_num * blend_weight);
                                accumulated_sample_num = accumulated_sample_num + adjacent_sample_num;
                            };
                        };
                    };

                    cache.resolved.write(entry_index,
                        sharc_pack_voxel_data(blended, accumulated_sample_num,
                                              accumulated_frame_num, stale_frame_num,
                                              voxel.sample_data_ext));
                    // Clear the accumulation entry for the next frame.
                    cache.accumulation.write(entry_index, make_uint4(0u));
                };
            };
        };
    };
}

//==============================================================================
// Debug (HashGridCommon.h debug helpers; the occupancy / collision overlays
// are pass-level and live with the future PassSharc)
//==============================================================================

// HashGrid_GetColorFromHash32
[[nodiscard]] inline Float3 sharc_get_color_from_hash32(Expr<uint> hash) noexcept {
    return make_float3(cast<Float>((hash >> 0u) & 0x3FFu) / 1023.0f,
                       cast<Float>((hash >> 11u) & 0x7FFu) / 2047.0f,
                       cast<Float>((hash >> 22u) & 0x7FFu) / 2047.0f);
}

// HashGrid_DebugColoredHash
template<typename Layout>
[[nodiscard]] inline Float3 sharc_debug_colored_hash(Float3 sample_position, Float3 sample_normal,
                                                     const SharcGridParams &grid) noexcept {
    auto hash_key = sharc_compute_spatial_hash<Layout>(sample_position, sample_normal, grid);
    UInt grid_level = sharc_get_level<Layout>(sample_position, grid);
    return sharc_get_color_from_hash32(sharc_hash32<Layout>(hash_key)) *
           sharc_get_color_from_hash32(sharc_hash_jenkins32(grid_level));
}

//==============================================================================
// Host-side helpers (non-kernel)
//==============================================================================

// Default capacity: 2^20 entries (40 MiB at 40 B/entry). Halved from 2^21 by
// Phase-2 perf tuning (2026-09-02): Resolve walks every entry every frame, and
// measured hit rates (99.99%+ on MaterialTest/Cornell) showed no capacity
// pressure — watch the occupancy/hit-rate readouts after scene swaps.
// Owner decision framework stays: capacity is fixed per scene (plan §8-1);
// scenes that need more override this.
inline constexpr uint kSharcDefaultEntriesNum = 1u << 20u;

// Bytes backing one hash-grid entry across the three buffers (plus the lock
// buffer when the lock-buffer insert route is active).
[[nodiscard]] inline size_t sharc_entry_bytes(bool compact, bool lock_buffer) noexcept {
    size_t key = compact ? 4u : 8u;
    size_t lock = lock_buffer ? 4u : 0u;
    return key + lock + 16u /*accumulation uint4*/ + 16u /*resolved SharcPackedData*/;
}

[[nodiscard]] inline size_t sharc_total_bytes(uint entries_num, bool compact, bool lock_buffer) noexcept {
    return static_cast<size_t>(entries_num) * sharc_entry_bytes(compact, lock_buffer);
}

// Capacity must be a power of two >= bucket size so that base-slot clamping
// (capacity - BUCKET + 1) keeps buckets contiguous; returns 0 when invalid.
[[nodiscard]] inline uint sharc_validate_capacity(uint entries_num) noexcept {
    bool pow2 = entries_num >= kSharcBucketSize &&
                (entries_num & (entries_num - 1u)) == 0u;
    return pow2 ? entries_num : 0u;
}

// Engine-integration alias: Config.h selects the key layout and insert route
// (owner decision: prefer the compare-exchange route for insert throughput;
// 64-bit keys make it an SM6.6 kernel — the pass must call set_warp_size()).
using SharcEngineLayout = std::conditional_t<NT_SHARC_COMPACT != 0,
                                             SharcLayoutCompact, SharcLayoutFull>;
#if NT_SHARC_64_BIT_ATOMICS != 0
using SharcEngineCache = SharcCache<SharcEngineLayout>;
#else
using SharcEngineCache = SharcLockCache<SharcEngineLayout>;
#endif

}// namespace newtype::render
