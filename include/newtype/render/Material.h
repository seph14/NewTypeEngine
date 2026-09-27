#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::render {

//==============================================================================
// Material Type Enum
//==============================================================================

enum class MaterialType : uint {
    Null           = 0u,   // unused slot
    Diffuse        = 1u,   // Lambertian diffuse
    Conductor      = 2u,   // Metallic conductor (complex Fresnel)
    Dielectric     = 3u,   // Glass / transmissive dielectric
    Plastic        = 4u,   // Dielectric clearcoat over diffuse
    Emissive       = 5u,   // Light emission
    Subsurface     = 6u,   // BSSRDF approximation
    Clearcoat      = 7u,   // Additive clearcoat layer (used as layer 1-3)
    Sheen          = 8u,   // additive sheen layer
    Anisotropy     = 9u,   // transformative modifier layer
    Iridescence    = 10u,  // transformative modifier layer
    ThinDielectric = 11u,  // thin-walled dielectric
    Unlit          = 12u,  // no shading, flat albedo/texture output
    Fabric         = 13u,  // fabric diffuse (Ashikhmin-Premoze) + sheen + anisotropy
};

/// First custom material type id — material callables are assigned sequential
/// ids starting here. Derived from the last built-in enum value so adding a
/// built-in type shifts it automatically. Projects and runtime DLLs must
/// reference this (or render::Material::CustomType) instead of hardcoding
/// 14u, or materials break when the engine gains a new built-in type
/// (docs/custom_material_callables.md).
static constexpr uint kFirstCustomMaterialType =
    static_cast<uint>(MaterialType::Fabric) + 1u;

//==============================================================================
// MaterialData — Flat GPU struct (192 bytes, alignas 16; luisa float3 = 16B slot)
//==============================================================================

/**
 * @brief Flat GPU-accessible material data structure
 *
 * All fields from all material types are a single struct. The `type` tag
 * determines which fields are active. Fields default to zero/null
 * for inactive types.
 *
 * CPU side: Use typed param structs (DiffuseParams, etc.) with to_data()
 * GPU side: LUISA_STRUCT registered, shader dispatch via type tag
 *
 * Texture indices: -1 means no texture. Stored as int for DSL compatibility.
 * Albedo: Stored as float3 [0,1] for DSL compatibility.
 */
struct alignas(16) MaterialData {
    // --- Classification & texture indices (32 bytes) ---
    uint   type               {0u};
    int    albedoTexIdx       {-1};
    int    normalTexIdx       {-1};
    int    rmaTexIdx          {-1};    // R=roughness, G=metallic, B=AO
    int    emissiveTexIdx     {-1};
    int    transmissionTexIdx {-1};
    int    anisoTexIdx        {-1};
    float  meta               {0.f};   // floor()=flags (bit0=receiveGI), frac()=layer_weight

    // --- Base colors (32 bytes) ---
    luisa::float4 albedo     {0.8f, 0.8f, 0.8f, 1.0f};  // xyz=color, w=alpha (constant opacity)
    luisa::float3 emission   {0.f, 0.f, 0.f};

    // --- Core PBR (16 bytes) ---
    float roughness           {0.5f};
    // Conductor/standard-layered: metalness. Dielectric (type 3): overloaded as
    // the interior medium priority for nested dielectrics (Schmidt-Budge 2002;
    // lower value = higher priority, 0 = default). Never read as metalness for
    // type 3 — the delta BSDF path ignores it. RMA textures do NOT reach it on
    // the PSR path (the loop reads the raw material field).
    float metallic            {0.f};
    float ior                 {1.5f};
    float alphacut            {0.f};   // Alpha cutout threshold (0=no cutout)

    // --- Specular (8 bytes) ---
    float specular_tint       {0.f};
    float specular_trans      {0.f};

    // --- Clearcoat (8 bytes) ---
    float clearcoat           {0.f};
    float clearcoat_gloss     {0.5f};

    // --- Sheen (8 bytes) ---
    float sheen               {0.f};
    float sheen_tint          {0.f};

    // --- Anisotropy (8 bytes) ---
    float anisotropic         {0.f};
    float anisotropic_rot     {0.f};

    // --- Subsurface (8 bytes) ---
    float flatness            {0.f};
    // Diffuse (thin-wall) transmission fraction: 0 = volumetric SSS (Burley
    // probe), > 0 = thin-wall subsurface (paper/leaves) — the HK lobe keeps
    // the surface response and the transmission lobe carries the rest; the
    // volumetric SSS probe is suppressed for diffuse_trans > 0.
    float diffuse_trans       {0.f};

    // --- Iridescence (8 bytes) ---
    float iridescence         {0.f};
    float iridescence_ior     {1.3f};

    // --- Glass / SSS attenuation / Conductor complex IOR real part (16 bytes) ---
    // Overloaded: Dielectric/Subsurface use this as Beer's law absorption tint;
    // Conductor uses it as eta_re for complex Fresnel.
    luisa::float3 attenuation   {1.f, 1.f, 1.f};

    // --- Conductor complex IOR imaginary part (16 bytes) ---
    // Conductor materials always carry k != 0 (complex-IOR-only mode). k = 0
    // occurs only on non-Conductor types, where metals fall back to the
    // Schlick albedo workflow (MicrofacetBSDF::evaluate).
    luisa::float3 conductor_k   {0.f, 0.f, 0.f};
    
    // --- Reserved (16 bytes) ---
    float attenuation_distance  {1.f};
    float iridescence_thickness {0.f};
    float bsdf_type_override    {0.f};  // For custom callables: override BSDF type for G-Buffer PSR. 0 = use material.type
    float fabric                {0.f};  // Blend factor: 0=Lambertian, 1=full Ashikhmin-Premoze fabric diffuse

    // --- Iridescence thickness map (appended; consumes former tail padding) ---
    // R = factor mask (multiplies `iridescence`), G = thickness mix parameter:
    //   thickness = mix(iridescence_thickness, iridescence_thickness_max, G)
    // when a texture is bound AND thickness_max >= 0; otherwise thickness is the
    // flat `iridescence_thickness` (nm). Sentinel -1 keeps legacy scenes
    // bit-identical. B/A reserved (e.g. per-pixel film IOR).
    int   iridescenceTexIdx       {-1};
    float iridescence_thickness_max {-1.f};

    // --- Dispersion (KHR_materials_dispersion; consumes remaining tail padding) ---
    // Abbe number V of the dielectric: 0 = off, typical 20-70 (lower = stronger
    // rainbow spread). Only read for MaterialType::Dielectric refraction paths
    // (PSR glass chain, rough-glass gather, transmission BSDF sampling) —
    // ThinDielectric never refracts a direction, so the field is inert there.
    // `ior` remains the d-line (589nm) index; per-channel indices are derived
    // on the GPU via dispersed_ior(). Struct stays 192 bytes (8 bytes of tail
    // padding remained after the iridescence append).
    float dispersion {0.f};
};

static_assert(sizeof(MaterialData) == 192u,
    "dispersion must land in the MaterialData tail padding; growing the struct "
    "changes every material-buffer stride");

// Field-by-field equality (memcmp is unsafe due to struct padding from alignas(16)
// + mixed float3/float layout; LuisaCompute Vector<T,N> has no operator==).
// Used for early-exit in MaterialPool::updateMaterialData.
[[nodiscard]] inline bool material_data_equal(const MaterialData& a, const MaterialData& b) noexcept {
    return a.type == b.type
        && a.albedoTexIdx == b.albedoTexIdx
        && a.normalTexIdx == b.normalTexIdx
        && a.rmaTexIdx == b.rmaTexIdx
        && a.emissiveTexIdx == b.emissiveTexIdx
        && a.transmissionTexIdx == b.transmissionTexIdx
        && a.anisoTexIdx == b.anisoTexIdx
        && a.meta == b.meta
        && a.albedo.x == b.albedo.x && a.albedo.y == b.albedo.y
        && a.albedo.z == b.albedo.z && a.albedo.w == b.albedo.w
        && a.emission.x == b.emission.x && a.emission.y == b.emission.y
        && a.emission.z == b.emission.z
        && a.roughness == b.roughness
        && a.metallic == b.metallic
        && a.ior == b.ior
        && a.alphacut == b.alphacut
        && a.specular_tint == b.specular_tint
        && a.specular_trans == b.specular_trans
        && a.clearcoat == b.clearcoat
        && a.clearcoat_gloss == b.clearcoat_gloss
        && a.sheen == b.sheen
        && a.sheen_tint == b.sheen_tint
        && a.anisotropic == b.anisotropic
        && a.anisotropic_rot == b.anisotropic_rot
        && a.flatness == b.flatness
        && a.diffuse_trans == b.diffuse_trans
        && a.iridescence == b.iridescence
        && a.iridescence_ior == b.iridescence_ior
        && a.attenuation.x == b.attenuation.x && a.attenuation.y == b.attenuation.y
        && a.attenuation.z == b.attenuation.z
        && a.conductor_k.x == b.conductor_k.x && a.conductor_k.y == b.conductor_k.y
        && a.conductor_k.z == b.conductor_k.z
        && a.attenuation_distance == b.attenuation_distance
        && a.iridescence_thickness == b.iridescence_thickness
        && a.bsdf_type_override == b.bsdf_type_override
        && a.fabric == b.fabric
        && a.iridescenceTexIdx == b.iridescenceTexIdx
        && a.iridescence_thickness_max == b.iridescence_thickness_max
        && a.dispersion == b.dispersion;
}

//==============================================================================
// Typed Parameter Structs (CPU-side)
//==============================================================================

struct DiffuseParams {
    luisa::float3 albedo  {0.8f, 0.8f, 0.8f};
    float roughness       {0.5f};
    float alphacut        {0.f};       // Alpha cutout threshold (0=no cutout)
    float albedo_alpha    {1.f};       // Constant opacity (albedo.w)
    luisa::float3 emission{0.f, 0.f, 0.f};
    int albedoTexIdx      {-1};
    int normalTexIdx      {-1};
    int rmaTexIdx         {-1};
    int emissiveTexIdx    {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type      = static_cast<uint>(MaterialType::Diffuse);
        d.albedo    = luisa::float4{albedo.x, albedo.y, albedo.z, albedo_alpha};
        d.roughness = roughness;
        d.alphacut  = alphacut;
        d.emission  = emission;
        d.albedoTexIdx  = albedoTexIdx;
        d.normalTexIdx  = normalTexIdx;
        d.rmaTexIdx     = rmaTexIdx;
        d.emissiveTexIdx= emissiveTexIdx;
        d.ior = 1.f;
        return d;
    }
};

struct ConductorParams {
    luisa::float3 albedo  {1.f, 1.f, 1.f};  // placeholder; complex IOR drives Fresnel
    float roughness       {0.3f};
    int albedoTexIdx      {-1};
    int normalTexIdx      {-1};
    int rmaTexIdx         {-1};
    // Complex IOR — the only conductor mode. Defaults: Aluminium
    // (keep in sync with kMetalPresets[MetalPreset::Aluminum]).
    luisa::float3 eta     {1.357068f, 0.876447f, 0.646272f};  // written to MaterialData.attenuation (overloaded)
    luisa::float3 k       {7.587795f, 6.447291f, 5.590148f};  // written to MaterialData.conductor_k

    MaterialData to_data() const {
        MaterialData d{};
        d.type          = static_cast<uint>(MaterialType::Conductor);
        d.albedo        = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.roughness     = roughness;
        d.metallic      = 1.f;
        // Overload: attenuation holds conductor_eta when conductor_k != 0.
        d.attenuation   = eta;
        d.conductor_k   = k;
        d.albedoTexIdx  = albedoTexIdx;
        d.normalTexIdx  = normalTexIdx;
        d.rmaTexIdx     = rmaTexIdx;
        return d;
    }
};

struct DielectricParams {
    luisa::float3 attenuation   {1.f, 1.f, 1.f};
    float attenuation_distance  {1.f};
    float roughness             {0.f};
    float ior                   {1.5f};
    float specular_trans        {1.f};
    // Interior medium priority for nested dielectrics (stored in
    // MaterialData.metallic, which is dead for this type). Lower value =
    // higher priority; 0 = default (equal priorities never cut out).
    float interior_priority     {0.f};
    // Abbe number (0 = off, typical 20-70). Drives per-channel IORs for the
    // refraction paths; `ior` is the d-line index (see MaterialData.dispersion).
    float dispersion            {0.f};
    int albedoTexIdx            {-1};
    int normalTexIdx            {-1};
    int rmaTexIdx               {-1};
    int transmissionTexIdx      {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type              = static_cast<uint>(MaterialType::Dielectric);
        d.attenuation       = attenuation;
        d.attenuation_distance=attenuation_distance;
        d.roughness         = roughness;
        d.ior               = ior;
        d.specular_trans    = specular_trans;
        d.metallic          = interior_priority;
        d.dispersion        = dispersion;
        d.albedoTexIdx      = albedoTexIdx;
        d.normalTexIdx      = normalTexIdx;
        d.rmaTexIdx         = rmaTexIdx;
        d.transmissionTexIdx= transmissionTexIdx;
        return d;
    }
};

struct PlasticParams {
    luisa::float3 albedo  {0.8f, 0.8f, 0.8f};
    float roughness       {0.3f};
    float ior             {1.5f};
    float specular_tint   {0.f};
    float clearcoat       {0.f};
    float clearcoat_gloss {0.5f};
    int albedoTexIdx      {-1};
    int normalTexIdx      {-1};
    int rmaTexIdx         {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type      = static_cast<uint>(MaterialType::Plastic);
        d.albedo    = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.roughness = roughness;
        d.ior       = ior;
        d.specular_tint=specular_tint;
        d.clearcoat =   clearcoat;
        d.clearcoat_gloss=clearcoat_gloss;
        d.albedoTexIdx  = albedoTexIdx;
        d.normalTexIdx  = normalTexIdx;
        d.rmaTexIdx     = rmaTexIdx;
        return d;
    }
};

struct EmissiveParams {
    luisa::float3 emission{1.f, 1.f, 1.f};
    luisa::float3 albedo {0.f, 0.f, 0.f};
    int emissiveTexIdx   {-1};
    int albedoTexIdx     {-1};
    int normalTexIdx     {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type          = static_cast<uint>(MaterialType::Emissive);
        d.emission      = emission;
        d.albedo        = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.emissiveTexIdx= emissiveTexIdx;
        d.albedoTexIdx  = albedoTexIdx;
        d.normalTexIdx  = normalTexIdx;
        return d;
    }
};

struct SubsurfaceParams {
    luisa::float3 albedo       {0.8f, 0.8f, 0.8f};
    luisa::float3 attenuation  {1.f, 1.f, 1.f};
    float attenuation_distance {1.f};
    float roughness    {0.5f};
    float flatness     {1.f};
    float ior          {1.4f};
    // Thin-wall transmission fraction (paper/leaves). 0 = volumetric SSS
    // (Burley probe handles diffusion); > 0 = thin subsurface: the HK lobe
    // is scaled by (1 - p_trans) and the transmission lobe carries the rest.
    float diffuse_trans {0.f};
    int albedoTexIdx   {-1};
    int normalTexIdx   {-1};
    int rmaTexIdx      {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type          = static_cast<uint>(MaterialType::Subsurface);
        d.albedo        = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.attenuation   = attenuation;
        d.attenuation_distance=attenuation_distance;
        d.roughness     = roughness;
        d.flatness      = flatness;
        d.ior           = ior;
        d.diffuse_trans = diffuse_trans;
        d.albedoTexIdx  = albedoTexIdx;
        d.normalTexIdx  = normalTexIdx;
        d.rmaTexIdx     = rmaTexIdx;
        return d;
    }
};

struct ClearcoatParams {
    float clearcoat       {1.f};
    float clearcoat_gloss {0.5f};
    // Coat interface IOR (drives the lobe's Schlick R0 AND the layered
    // F12/F23 base attenuation — resolve_surface_layered reads layer ior).
    // Disney default 1.5 → R0 0.04 (legacy look).
    float ior             {1.5f};
    int normalTexIdx      {-1};   // coat normal map (sampled by resolve_surface_layered)

    MaterialData to_data() const {
        MaterialData d{};
        d.type           = static_cast<uint>(MaterialType::Clearcoat);
        d.clearcoat      = clearcoat;
        d.clearcoat_gloss= clearcoat_gloss;
        d.ior            = ior;
        d.normalTexIdx   = normalTexIdx;
        return d;
    }
};

struct SheenParams {
    float sheen      {1.f};
    float sheen_tint {0.f};

    MaterialData to_data() const {
        MaterialData d{};
        d.type      = static_cast<uint>(MaterialType::Sheen);
        d.sheen     = sheen;
        d.sheen_tint= sheen_tint;
        return d;
    }
};

struct AnisotropyParams {
    float anisotropic     {1.f};
    float anisotropic_rot {0.f};
    int   anisoTexIdx     {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type           =static_cast<uint>(MaterialType::Anisotropy);
        d.anisotropic    =anisotropic;
        d.anisotropic_rot=anisotropic_rot;
        d.anisoTexIdx    =anisoTexIdx;
        return d;
    }
};

struct IridescenceParams {
    float iridescence             {1.f};
    float iridescence_ior         {1.3f};
    // Film thickness in nm. Without a thickness texture this is the fixed
    // thickness; with one, it is the range minimum.
    float iridescence_thickness   {100.f};
    // Range maximum (nm) for the thickness texture's G channel. Sentinel -1
    // disables texture mapping (fixed thickness above).
    float iridescence_thickness_max {-1.f};
    int   iridescenceTexIdx       {-1};   // R=factor mask, G=thickness mix

    MaterialData to_data() const {
        MaterialData d{};
        d.type           =static_cast<uint>(MaterialType::Iridescence);
        d.iridescence    =iridescence;
        d.iridescence_ior=iridescence_ior;
        d.iridescence_thickness=iridescence_thickness;
        d.iridescence_thickness_max=iridescence_thickness_max;
        d.iridescenceTexIdx=iridescenceTexIdx;
        return d;
    }
};

struct UnlitParams {
    luisa::float3 albedo {0.8f, 0.8f, 0.8f};
    int albedoTexIdx     {-1};
    int normalTexIdx     {-1};
    bool receiveGI       {false};

    MaterialData to_data() const {
        MaterialData d{};
        d.type         = static_cast<uint>(MaterialType::Unlit);
        d.albedo       = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.albedoTexIdx = albedoTexIdx;
        d.normalTexIdx = normalTexIdx;
        d.meta         = receiveGI ? 1.0f : 0.0f;
        return d;
    }
};

struct ThinDielectricParams {
    luisa::float3 attenuation {1.f, 1.f, 1.f};  // transmission tint
    float roughness           {0.f};
    float ior                 {1.5f};
    int albedoTexIdx          {-1};
    int normalTexIdx          {-1};
    int rmaTexIdx             {-1};
    int transmissionTexIdx    {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type              = static_cast<uint>(MaterialType::ThinDielectric);
        d.albedo            = luisa::float4{attenuation.x, attenuation.y, attenuation.z, 1.0f};
        d.attenuation       = attenuation;
        d.roughness         = roughness;
        d.ior               = ior;
        d.albedoTexIdx      = albedoTexIdx;
        d.normalTexIdx      = normalTexIdx;
        d.rmaTexIdx         = rmaTexIdx;
        d.transmissionTexIdx= transmissionTexIdx;
        return d;
    }
};

struct FabricParams {
    luisa::float3 albedo       {0.8f, 0.8f, 0.8f};
    float roughness            {0.5f};
    float fabric               {1.f};          // 0=Lambertian, 1=full fabric diffuse
    float sheen                {0.f};
    float sheen_tint           {0.f};
    float anisotropic          {0.f};
    float anisotropic_rot      {0.f};
    int albedoTexIdx           {-1};
    int normalTexIdx           {-1};
    int rmaTexIdx              {-1};
    int anisoTexIdx            {-1};

    MaterialData to_data() const {
        MaterialData d{};
        d.type           = static_cast<uint>(MaterialType::Fabric);
        d.albedo         = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
        d.roughness      = roughness;
        d.fabric         = fabric;
        d.sheen          = sheen;
        d.sheen_tint     = sheen_tint;
        d.anisotropic    = anisotropic;
        d.anisotropic_rot= anisotropic_rot;
        d.albedoTexIdx   = albedoTexIdx;
        d.normalTexIdx   = normalTexIdx;
        d.rmaTexIdx      = rmaTexIdx;
        d.anisoTexIdx    = anisoTexIdx;
        return d;
    }
};

//==============================================================================
// Convenience Factory Functions
//==============================================================================

inline MaterialData make_diffuse(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    float roughness = 0.5f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1) {
    DiffuseParams p;
    p.albedo        =albedo;
    p.roughness     =roughness;
    p.albedoTexIdx  =albedoTexIdx;
    p.normalTexIdx  =normalTexIdx;
    p.rmaTexIdx     =rmaTexIdx;
    return p.to_data();
}

inline MaterialData make_conductor(
    const luisa::float3 &albedo = {1.f, 1.f, 1.f},
    float roughness     = 0.3f,
    int albedoTexIdx    = -1,
    int normalTexIdx    = -1,
    int rmaTexIdx       = -1) {
    ConductorParams p;
    p.albedo        =albedo;
    p.roughness     =roughness;
    p.albedoTexIdx  =albedoTexIdx;
    p.normalTexIdx  =normalTexIdx;
    p.rmaTexIdx     =rmaTexIdx;
    return p.to_data();
}

inline MaterialData make_dielectric(
    const luisa::float3 &attenuation = {1.f, 1.f, 1.f},
    float ior        = 1.5f,
    float roughness  = 0.f,
    int albedoTexIdx = -1,
    int transmissionTexIdx = -1,
    float interior_priority = 0.f,
    float dispersion = 0.f) {
    DielectricParams p;
    p.attenuation   =attenuation;
    p.ior           =ior;
    p.roughness     =roughness;
    p.albedoTexIdx  =albedoTexIdx;
    p.transmissionTexIdx=transmissionTexIdx;
    p.interior_priority = interior_priority;
    p.dispersion    =dispersion;
    return p.to_data();
}

inline MaterialData make_plastic(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    float roughness = 0.3f,
    float ior = 1.5f,
    float clearcoat = 0.f,
    float clearcoat_gloss = 0.5f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1) {
    PlasticParams p;
    p.albedo        =albedo;
    p.roughness     =roughness;
    p.ior           =ior;
    p.clearcoat     =clearcoat;
    p.clearcoat_gloss=clearcoat_gloss;
    p.albedoTexIdx  =albedoTexIdx;
    p.normalTexIdx  =normalTexIdx;
    p.rmaTexIdx     =rmaTexIdx;
    return p.to_data();
}

inline MaterialData make_emissive(
    const luisa::float3 &emission = {1.f, 1.f, 1.f},
    int emissiveTexIdx = -1) {
    EmissiveParams p;
    p.emission      =emission;
    p.emissiveTexIdx=emissiveTexIdx;
    return p.to_data();
}

inline MaterialData make_subsurface(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    float roughness = 0.5f,
    float flatness = 1.f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1,
    float diffuse_trans = 0.f,
    float ior = 1.4f) {
    SubsurfaceParams p;
    p.albedo        =albedo;
    p.roughness     =roughness;
    p.flatness      =flatness;
    p.albedoTexIdx  =albedoTexIdx;
    p.normalTexIdx  =normalTexIdx;
    p.rmaTexIdx     =rmaTexIdx;
    p.diffuse_trans =diffuse_trans;
    p.ior           =ior;
    return p.to_data();
}

/// Thin-wall subsurface preset (paper / lampshade / leaf). Sheets should also
/// be flagged double-sided (instance kDoubleSidedFlag) so both faces shade.
inline MaterialData make_paper(
    const luisa::float3 &albedo = {0.9f, 0.9f, 0.88f},
    float diffuse_trans = 0.7f,
    float attenuation_distance = 0.05f) {
    SubsurfaceParams p;
    p.albedo               = albedo;
    p.attenuation          = albedo;          // transmitted tint tracks the sheet color
    p.attenuation_distance = attenuation_distance;
    p.roughness            = 1.f;
    p.flatness             = 1.f;
    p.ior                  = 1.3f;
    p.diffuse_trans        = diffuse_trans;
    return p.to_data();
}

inline MaterialData make_sheen(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    float roughness = 0.5f,
    float sheen = 1.f,
    float sheen_tint = 0.f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1) {
    DiffuseParams p;
    p.albedo        = albedo;
    p.roughness     = roughness;
    p.albedoTexIdx  = albedoTexIdx;
    p.normalTexIdx  = normalTexIdx;
    p.rmaTexIdx     = rmaTexIdx;
    auto d = p.to_data();
    d.type = static_cast<uint>(MaterialType::Sheen);
    d.sheen      = sheen;
    d.sheen_tint = sheen_tint;
    return d;
}

inline MaterialData make_clearcoat(
    float clearcoat = 1.f,
    float clearcoat_gloss = 0.5f,
    int normalTexIdx = -1,
    float ior = 1.5f) {
    ClearcoatParams p;
    p.clearcoat      =clearcoat;
    p.clearcoat_gloss=clearcoat_gloss;
    p.normalTexIdx   =normalTexIdx;
    p.ior            =ior;
    return p.to_data();
}

inline MaterialData make_iridescence(
    float iridescence = 1.f,
    float iridescence_ior = 1.3f,
    float iridescence_thickness = 400.f,
    float iridescence_thickness_max = -1.f,
    int iridescenceTexIdx = -1) {
    IridescenceParams p;
    p.iridescence            =iridescence;
    p.iridescence_ior        =iridescence_ior;
    p.iridescence_thickness  =iridescence_thickness;
    p.iridescence_thickness_max=iridescence_thickness_max;
    p.iridescenceTexIdx      =iridescenceTexIdx;
    return p.to_data();
}

inline MaterialData make_anisotropy(
    float anisotropic = 1.f,
    float anisotropic_rot = 0.f,
    int anisoTexIdx = -1) {
    AnisotropyParams p;
    p.anisotropic    =anisotropic;
    p.anisotropic_rot=anisotropic_rot;
    p.anisoTexIdx    =anisoTexIdx;
    return p.to_data();
}

inline MaterialData make_unlit(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    int albedoTexIdx = -1,
    bool receiveGI = false) {
    UnlitParams p;
    p.albedo        = albedo;
    p.albedoTexIdx  = albedoTexIdx;
    p.receiveGI     = receiveGI;
    return p.to_data();
}

inline MaterialData make_thin_dielectric(
    const luisa::float3 &attenuation = {1.f, 1.f, 1.f},
    float ior        = 1.5f,
    float roughness  = 0.f,
    int albedoTexIdx = -1,
    int transmissionTexIdx = -1) {
    ThinDielectricParams p;
    p.attenuation       = attenuation;
    p.ior               = ior;
    p.roughness         = roughness;
    p.albedoTexIdx      = albedoTexIdx;
    p.transmissionTexIdx= transmissionTexIdx;
    return p.to_data();
}

inline MaterialData make_fabric(
    const luisa::float3 &albedo = {0.8f, 0.8f, 0.8f},
    float roughness     = 0.5f,
    float fabric        = 1.f,
    float sheen         = 0.f,
    float sheen_tint    = 0.f,
    float anisotropic   = 0.f,
    float anisotropic_rot = 0.f,
    int albedoTexIdx    = -1,
    int normalTexIdx    = -1,
    int rmaTexIdx       = -1,
    int anisoTexIdx     = -1) {
    FabricParams p;
    p.albedo         = albedo;
    p.roughness      = roughness;
    p.fabric         = fabric;
    p.sheen          = sheen;
    p.sheen_tint     = sheen_tint;
    p.anisotropic    = anisotropic;
    p.anisotropic_rot= anisotropic_rot;
    p.albedoTexIdx   = albedoTexIdx;
    p.normalTexIdx   = normalTexIdx;
    p.rmaTexIdx      = rmaTexIdx;
    p.anisoTexIdx    = anisoTexIdx;
    return p.to_data();
}

} // namespace newtype::render
