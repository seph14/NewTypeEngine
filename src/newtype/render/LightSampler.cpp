//
// Created by Claude on 2026/03/25.
//

#include "newtype/render/LightSampler.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/scene/Geometry.h"
#include "cinder/Log.h"
#include "cinder/Surface.h"

#include <algorithm>
#include <numeric>
#include <numbers>

namespace newtype::render {

namespace {
// World-space geometric normal for a triangle light, matching the shader-side
// normalize(cross(v1 - v0, v2 - v0)) the buffer normal replaces.
luisa::float3 triangle_world_normal(
    luisa::float3 v0, luisa::float3 v1, luisa::float3 v2) noexcept {
    luisa::float3 e1 = v1 - v0;
    luisa::float3 e2 = v2 - v0;
    luisa::float3 c = make_float3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x);
    float len = luisa::sqrt(luisa::max(luisa::dot(c, c), 1e-30f));
    return c * (1.0f / len);
}
}

//==============================================================================
// EnvironmentLight Implementation
//==============================================================================

EnvironmentLight::EnvironmentLight(Device &device) noexcept
    : _device(device) {}

void EnvironmentLight::build(Image<float> &&envmap, Stream &stream) noexcept {
    _envmap = std::move(envmap);
    _width = _envmap.size().x;
    _height = _envmap.size().y;

    // Download envmap pixels to CPU for CDF construction
    auto pixel_count = _width * _height * 4;
    luisa::vector<float> pixels(pixel_count);
    stream << _envmap.copy_to(pixels.data()) << luisa::compute::synchronize();

    build_cdfs(pixels.data(), stream);
}

void EnvironmentLight::build_cdfs(const float* rgba_data, Stream &stream) noexcept {
    // Build CDFs from pixel data on CPU
    // Marginal CDF: (height + 1) floats
    // Conditional CDF: height * (width + 1) floats
    _cdfs_buffer_marginal = _device.create_buffer<float>(_height + 1u);
    _cdfs_buffer_conditional = _device.create_buffer<float>(_height * (_width + 1u));

    luisa::vector<float> marginal_cdf(_height + 1, 0.0f);
    luisa::vector<float> conditional_cdf(_height * (_width + 1), 0.0f);

    float integral = 0.0f;
    for (uint j = 0; j < _height; ++j) {
        float theta = std::numbers::pi_v<float> * (static_cast<float>(j) + 0.5f) / static_cast<float>(_height);
        float sin_theta = std::sin(theta);

        float row_total = 0.0f;
        conditional_cdf[j * (_width + 1)] = 0.0f;

        for (uint i = 0; i < _width; ++i) {
            const float* pixel = &rgba_data[(j * _width + i) * 4];
            float lum = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
            float weight = lum * sin_theta;
            row_total += weight;
            conditional_cdf[j * (_width + 1) + i + 1] = row_total;
        }

        integral += row_total;
        marginal_cdf[j + 1] = integral;
    }
    _cdfs.integral = integral;

    stream << _cdfs_buffer_marginal.copy_from(marginal_cdf.data());
    stream << _cdfs_buffer_conditional.copy_from(conditional_cdf.data());

    _built = true;
    CI_LOG_I("EnvironmentLight built: " << _width << "x" << _height
        << ", integral=" << integral);
}

//==============================================================================
// Rotation / Exposure
//==============================================================================

void EnvironmentLight::_update_rotation() noexcept {
    // Combined rotation: Ry(yaw) * Rx(elevation)
    // Yaw rotates horizontally, elevation tilts up/down
    float cy = std::cos(_yaw), sy = std::sin(_yaw);
    float ce = std::cos(_elevation), se = std::sin(_elevation);

    // Ry(yaw)
    auto ry = luisa::make_float3x3(
        cy,  0.0f, sy,
        0.0f, 1.0f, 0.0f,
        -sy, 0.0f, cy);
    // Rx(elevation)
    auto rx = luisa::make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, ce,   -se,
        0.0f, se,    ce);

    _rotationMatrix = luisa::make_float3x3(
        // Row 0: ry * rx[0..2]
        cy * 1.0f + sy * 0.0f,  cy * 0.0f + sy * se,   cy * 0.0f + sy * ce,
        // Row 1: rx[4..6]
        0.0f, ce, -se,
        // Row 2: ry * rx[8..10]
        -sy * 1.0f + cy * 0.0f, -sy * 0.0f + cy * se,  -sy * 0.0f + cy * ce);

    // Sun direction (world space) from elevation + yaw
    _sunDirection = luisa::make_float3(ce * sy, se, ce * cy);

    _rotationDirty = true;
}

void EnvironmentLight::set_yaw(float yaw_rad) noexcept {
    _yaw = yaw_rad;
    _update_rotation();
}

void EnvironmentLight::set_elevation(float elev_rad) noexcept {
    _elevation = elev_rad;
    _update_rotation();
}

void EnvironmentLight::update_rotation_buffer(Stream &stream) noexcept {
    if (!_rotationBuffer) {
        _rotationBuffer = _device.create_buffer<luisa::float3x3>(1u);
        _rotationDirty = true;
    }
    if (_rotationDirty) {
        stream << _rotationBuffer.copy_from(&_rotationMatrix);
        _rotationDirty = false;
    }
}

//==============================================================================
// Procedural Sky (Rayleigh / Mie Scattering)
//==============================================================================

void EnvironmentLight::generate_procedural_sky(Stream &stream, uint width, uint height) noexcept {
    _update_rotation();

    constexpr float pi = std::numbers::pi_v<float>;
    luisa::vector<float> pixels(width * height * 4);

    // Sun direction in envmap-local space (before rotation)
    // Envmap Y-up: theta from +Y, phi from +X toward +Z
    // Sun direction = (cos(elev)*sin(yaw), sin(elev), cos(elev)*cos(yaw))
    // But for envmap generation we use the LOCAL sun dir (pre-rotation)
    // so the sun is at a fixed position in the envmap, and rotation
    // transforms the whole thing at sample time.
    // Use elevation for sun height, yaw=0 for the default sun position in envmap.
    float sun_elev = _elevation;
    float ce = std::cos(sun_elev), se = std::sin(sun_elev);
    luisa::float3 sun_dir = luisa::make_float3(0.0f, se, ce); // sun at phi=0

    // Rayleigh scattering coefficients (sea level, approximate)
    luisa::float3 rayleigh_coeff = luisa::make_float3(5.5e-6f, 13.0e-6f, 22.4e-6f);
    float rayleigh_scale_height = 8500.0f;

    // Mie scattering
    float mie_coeff_val = 21.0e-6f;
    float mie_scale_height = 1200.0f;
    float mie_g = 0.76f; // Henyey-Greenstein asymmetry

    float turbidity = _turbidity;
    float rayleigh_factor = 1.0f + (turbidity - 1.0f) * 0.5f;

    // Mie phase function constant factor
    float mie_phase_factor = (3.0f * (1.0f - mie_g * mie_g)) /
        (8.0f * pi * (2.0f + mie_g * mie_g));

    // Sun color (6500K daylight)
    luisa::float3 sun_color = luisa::make_float3(1.0f, 0.95f, 0.85f);

    for (uint j = 0; j < height; ++j) {
        float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(height);
        float theta = v * pi;
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        for (uint i = 0; i < width; ++i) {
            float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(width);
            float phi = u * 2.0f * pi;

            // Direction in envmap-local space (Y-up)
            luisa::float3 dir = luisa::make_float3(
                sin_theta * std::cos(phi), cos_theta, sin_theta * std::sin(phi));

            float cos_sun = std::max(luisa::dot(dir, sun_dir), 0.0f);

            // Rayleigh phase function: 3/(16*pi) * (1 + cos^2)
            float rayleigh_phase = 3.0f / (16.0f * pi) * (1.0f + cos_sun * cos_sun);

            // Mie phase function (Henyey-Greenstein)
            float mie_denom = 1.0f + mie_g * mie_g - 2.0f * mie_g * cos_sun;
            float mie_phase = mie_phase_factor * (1.0f + cos_sun * cos_sun) /
                std::pow(std::max(mie_denom, 1e-10f), 1.5f);

            // Air mass (Kasten-Young approximation)
            float zenith_angle = std::acos(std::max(dir.y, 0.0f));
            float zenith_deg = zenith_angle * 180.0f / pi;
            float air_mass = 1.0f / (std::cos(zenith_angle) +
                0.15f * std::pow(3.885f + 96.0f - zenith_deg, -1.253f));

            // Optical depth
            float rayleigh_od = air_mass * rayleigh_scale_height * rayleigh_factor;
            float mie_od = air_mass * mie_scale_height;

            luisa::float3 extinction_rayleigh = luisa::make_float3(
                std::exp(-rayleigh_coeff.x * rayleigh_od),
                std::exp(-rayleigh_coeff.y * rayleigh_od),
                std::exp(-rayleigh_coeff.z * rayleigh_od));
            float extinction_mie = std::exp(-mie_coeff_val * mie_od);
            luisa::float3 extinction = luisa::make_float3(
                extinction_rayleigh.x * extinction_mie,
                extinction_rayleigh.y * extinction_mie,
                extinction_rayleigh.z * extinction_mie);

            // Inscattering
            luisa::float3 rayleigh_inscatter = luisa::make_float3(
                rayleigh_coeff.x * rayleigh_phase * rayleigh_od,
                rayleigh_coeff.y * rayleigh_phase * rayleigh_od,
                rayleigh_coeff.z * rayleigh_phase * rayleigh_od);
            float mie_inscatter = mie_coeff_val * mie_phase * mie_od * 10.0f;

            // Sun disc + glow
            float sun_cos = luisa::dot(dir, sun_dir);
            // Sharp sun disc (angular radius ~0.27 deg ≈ cos > 0.99995)
            float sun_disc = (sun_cos > 0.99995f) ? 1000.0f : 0.0f;
            // Exponential glow for sun aureole
            float sun_glow = std::exp(-(1.0f - sun_cos) * 500.0f) * 5.0f;

            // Combine sky
            luisa::float3 sky = (rayleigh_inscatter + luisa::make_float3(mie_inscatter)) * sun_color * extinction;
            sky = sky + sun_color * (sun_disc + sun_glow) * extinction;

            // Ground albedo bounce (subtle)
            float ground_factor = std::max(-dir.y, 0.0f) * 0.1f;
            sky = sky + _groundAlbedo * ground_factor;

            // Below-horizon blend to dark ground
            float ground_blend = std::clamp(-dir.y * 10.0f, 0.0f, 1.0f);
            luisa::float3 ground = luisa::make_float3(0.05f, 0.04f, 0.03f);
            sky = luisa::make_float3(
                sky.x * (1.0f - ground_blend) + ground.x * ground_blend,
                sky.y * (1.0f - ground_blend) + ground.y * ground_blend,
                sky.z * (1.0f - ground_blend) + ground.z * ground_blend);

            // Intensity scaling
            sky = sky * _sunIntensity;

            float* pixel = &pixels[(j * width + i) * 4];
            pixel[0] = std::max(sky.x, 0.0f);
            pixel[1] = std::max(sky.y, 0.0f);
            pixel[2] = std::max(sky.z, 0.0f);
            pixel[3] = 1.0f;
        }
    }

    auto envmap_image = _device.create_image<float>(luisa::compute::PixelStorage::FLOAT4, width, height);
    stream << envmap_image.copy_from(pixels.data());
    auto compressed = TextureConverter::compressTexture(envmap_image, _device, stream, PixelStorage::BC6);
    // Keep the uncompressed source when compression is skipped (size gates,
    // missing extension) — an invalid image would bind a null SRV.
    if (compressed.valid()) {
        envmap_image.release();
        _envmap = std::move(compressed);
    } else {
        _envmap = std::move(envmap_image);
    }
    _width = width;
    _height = height;
    _isProcedural = true;
    build_cdfs(pixels.data(), stream);

    CI_LOG_I("EnvironmentLight: Procedural sky generated (" << width << "x" << height
        << ", elev=" << (_elevation * 180.0f / pi) << "deg"
        << ", turb=" << _turbidity << ")");
}

void EnvironmentLight::regenerate_procedural_sky(Stream &stream) noexcept {
    if (_isProcedural) {
        generate_procedural_sky(stream, _width, _height);
    }
}

//==============================================================================
// LightSampler Implementation
//==============================================================================

LightSampler::LightSampler(Device &device) noexcept
    : _device(device), _env_light(device) {}

//==============================================================================
// Building
//==============================================================================

void LightSampler::build(Stream &stream, const scene::Geometry &geometry,
                         const MaterialPool &material_pool) noexcept {
    _collect_emissive_tris(geometry, material_pool);
    _build_alias_table(stream);
    _upload_to_gpu(stream);
}

void LightSampler::build(Stream &stream, const scene::Geometry &geometry,
                         const MaterialPool &material_pool,
                         luisa::compute::Image<float> &&envmap) noexcept {
    build(stream, geometry, material_pool);
    _env_light.build(std::move(envmap), stream);
}

void LightSampler::build_envmap(uint width, uint height, const float* rgba_data, Stream &stream) noexcept {
    auto envmap_image = _device.create_image<float>(luisa::compute::PixelStorage::FLOAT4, width, height);
    stream << envmap_image.copy_from(rgba_data);
    auto compressed = TextureConverter::compressTexture(envmap_image, _device, stream, PixelStorage::BC6);
    // Keep the uncompressed source when compression is skipped (size gates,
    // missing extension) — an invalid image would bind a null SRV.
    if (compressed.valid()) {
        _env_light._envmap = std::move(compressed);
        envmap_image.release();
    } else {
        _env_light._envmap = std::move(envmap_image);
    }
    _env_light._width = width;
    _env_light._height = height;
    _env_light.build_cdfs(rgba_data, stream);
}

void LightSampler::build_envmap(const ci::Surface32f &surface, Stream &stream) noexcept {
    auto width = static_cast<uint>(surface.getWidth());
    auto height = static_cast<uint>(surface.getHeight());

    if (surface.hasAlpha()) {
        build_envmap(width, height, surface.getData(), stream);
    } else {
        // Pack into RGBA float array
        luisa::vector<float> pixels(width * height * 4);
        float* ptr = pixels.data();

        auto iter = surface.getIter();
        while (iter.line()) {
            while (iter.pixel()) {
                *ptr++ = iter.r();
                *ptr++ = iter.g();
                *ptr++ = iter.b();
                *ptr++ = 1.f;
            }
        }

        build_envmap(width, height, pixels.data(), stream);
    }
}

void LightSampler::rebuild(Stream &stream, const scene::Geometry &geometry,
                           const MaterialPool &material_pool) noexcept {
    _emissive_tris_cpu.clear();
    _total_emissive_count = 0;
    _total_power = 0.0f;
    build(stream, geometry, material_pool);
}

void LightSampler::update_weights(Stream &stream, const MaterialPool &material_pool,
                                  const scene::Geometry &geometry) noexcept {
    if (_emissive_tris_cpu.empty()) return;

    _total_power = 0.0f;
    for (auto &rec : _emissive_tris_cpu) {
        const auto &material = material_pool.getMaterial(rec.material_index);
        luisa::float3 emission = material.data.emission;
        float luminance_val = dot(emission, make_float3(0.2126f, 0.7152f, 0.0722f));
        rec.emission = emission;
        // Zero power for hidden instances so the alias table excludes them
        // from sampling. Allows cheap updates on visibility toggles.
        float base_power = luminance_val * rec.area;
        rec.power = geometry.is_instance_visible(rec.instance_id) ? base_power : 0.0f;
        _total_power += rec.power;
    }
    _total_power_inv = (_total_power > 0.0f) ? (1.0f / _total_power) : 0.0f;

    _build_alias_table(stream);
    _upload_to_gpu(stream);
}

void LightSampler::update_transforms(Stream &stream, const scene::Geometry &geometry,
                                     const MaterialPool &material_pool,
                                     bool scale_changed) noexcept {
    uint n = _total_emissive_count;
    if (n == 0) return;

    _vertex_staging.resize(n);

    if (scale_changed) {
        // Full recompute: areas, powers, alias table, vertices
        _light_staging.resize(n);
        _total_power = 0.0f;

        for (uint i = 0; i < n; ++i) {
            auto &rec = _emissive_tris_cpu[i];

            // Recompute area from current transform
            rec.area = _triangle_area(*rec.shape, rec.primitive_id);

            // Recompute power
            const auto &material = material_pool.getMaterial(rec.material_index);
            luisa::float3 emission = material.data.emission;
            float luminance_val = dot(emission, make_float3(0.2126f, 0.7152f, 0.0722f));
            rec.emission = emission;
            rec.power = luminance_val * rec.area;
            _total_power += rec.power;

            _light_staging[i].instance_id    = rec.instance_id;
            _light_staging[i].primitive_id   = rec.primitive_id;
            _light_staging[i].area           = rec.area;
            _light_staging[i].ex             = emission.x;
            _light_staging[i].ey             = emission.y;
            _light_staging[i].ez             = emission.z;

            // Recompute world-space vertices + light normal
            _get_triangle_vertices(*rec.shape, rec.primitive_id,
                _vertex_staging[i].v0, _vertex_staging[i].v1, _vertex_staging[i].v2);
            luisa::float3 ln = triangle_world_normal(
                _vertex_staging[i].v0, _vertex_staging[i].v1, _vertex_staging[i].v2);
            _light_staging[i].nx = ln.x;
            _light_staging[i].ny = ln.y;
            _light_staging[i].nz = ln.z;
        }

        _total_power_inv = (_total_power > 0.0f) ? (1.0f / _total_power) : 0.0f;

        // Fill pdfs after total_power is known
        for (uint i = 0; i < n; ++i)
            _light_staging[i].pdf = (_total_power > 0.0f) ? (_emissive_tris_cpu[i].power * _total_power_inv) : 0.0f;

        _build_alias_table(stream);
        stream << _triangle_lights.copy_from(_light_staging.data());
        stream << _triangle_vertices.copy_from(_vertex_staging.data());
    } else {
        // Affine-only: re-upload vertex positions AND light normals (a stored
        // normal would go stale under rotation). Staging is refilled fully from
        // CPU records — it may be empty when this runs before any scale_changed
        // pass (_upload_to_gpu fills a local vector, not _light_staging).
        _light_staging.resize(n);
        for (uint i = 0; i < n; ++i) {
            const auto &rec = _emissive_tris_cpu[i];
            _light_staging[i].instance_id    = rec.instance_id;
            _light_staging[i].primitive_id   = rec.primitive_id;
            _light_staging[i].area           = rec.area;
            _light_staging[i].ex             = rec.emission.x;
            _light_staging[i].ey             = rec.emission.y;
            _light_staging[i].ez             = rec.emission.z;
            _light_staging[i].pdf            = (_total_power > 0.0f) ? (rec.power * _total_power_inv) : 0.0f;

            _get_triangle_vertices(*rec.shape, rec.primitive_id,
                _vertex_staging[i].v0, _vertex_staging[i].v1, _vertex_staging[i].v2);
            luisa::float3 ln = triangle_world_normal(
                _vertex_staging[i].v0, _vertex_staging[i].v1, _vertex_staging[i].v2);
            _light_staging[i].nx = ln.x;
            _light_staging[i].ny = ln.y;
            _light_staging[i].nz = ln.z;
        }
        stream << _triangle_lights.copy_from(_light_staging.data());
        stream << _triangle_vertices.copy_from(_vertex_staging.data());
    }
}

//==============================================================================
// Emissive Triangle Collection
//==============================================================================

void LightSampler::_collect_emissive_tris(
    const scene::Geometry &geometry,
    const MaterialPool &material_pool) noexcept {

    _emissive_tris_cpu.clear();
    _total_power = 0.0f;

    uint global_tri_index = 0;

    const auto& instances = geometry.instances();

    for (uint idx : geometry.light_indices()) {
        const auto *shape = instances[idx].get_shape();

        // skip dark geom
        const auto &material = material_pool.getMaterial(shape->material_id());
        float3 emission = material.data.emission;
        float luminance_val = dot(emission, make_float3(0.2126f, 0.7152f, 0.0722f));
        if (luminance_val <= 0.001f) continue;

        // skip zero size geom
        const auto& matrix = shape->transform()->matrix();
        if(shape->transform()->is_dirty() &&
          (luisa::dot(matrix.cols[0].xyz(), matrix.cols[0].xyz()) +
           luisa::dot(matrix.cols[1].xyz(), matrix.cols[1].xyz())+
           luisa::dot(matrix.cols[2].xyz(), matrix.cols[2].xyz())) < 1e-6f) continue;

        // Include hidden lights with power=0 so the record set stays stable
        // across visibility toggles (avoids full rebuild on every toggle —
        // update_weights re-zeros powers instead). Sampling skips zero-power
        // entries via the alias table's pdf/alias logic.
        const bool visible = instances[idx].visible;
        uint tlas_index = idx;  // idx is the dense TLAS instance index
        uint tri_count = shape->triangle_count();

        for (uint prim_id = 0; prim_id < tri_count; ++prim_id) {
            float area = _triangle_area(*shape, prim_id);
            float power = visible ? (luminance_val * area) : 0.0f;

            if (visible && power <= 0.0f) continue;

            EmissiveTriangleRecord tri;
            tri.shape = shape;
            tri.instance_id = tlas_index;
            tri.primitive_id = prim_id;
            tri.triangle_index = global_tri_index++;
            tri.area = area;
            tri.emission = emission;
            tri.power = power;
            tri.material_index = shape->material_id();

            _emissive_tris_cpu.push_back(tri);
            _total_power += power;
        }
    }

    uint tlas_idx = 0;
    for (const auto &instance : instances) {
        const auto *shape = instance.get_shape();
        if (!shape || shape->properties() & scene::PROPERTY_HAS_LIGHT) { tlas_idx++; continue; }

        // skip dark geom
        const auto &material = material_pool.getMaterial(shape->material_id());
        float3 emission = material.data.emission;
        float luminance_val = dot(emission, make_float3(0.2126f, 0.7152f, 0.0722f));
        if (luminance_val <= 0.001f) { tlas_idx++; continue; }

        // skip zero size geom
        const auto& matrix = shape->transform()->matrix();
        if (shape->transform()->is_dirty() &&
           (luisa::dot(matrix.cols[0].xyz(), matrix.cols[0].xyz()) +
            luisa::dot(matrix.cols[1].xyz(), matrix.cols[1].xyz()) +
            luisa::dot(matrix.cols[2].xyz(), matrix.cols[2].xyz())) < 1e-6f) { tlas_idx++; continue; }

        const bool visible = instance.visible;
        uint tri_count = shape->triangle_count();

        for (uint prim_id = 0; prim_id < tri_count; ++prim_id) {
            float area = _triangle_area(*shape, prim_id);
            float power = visible ? (luminance_val * area) : 0.0f;

            if (visible && power <= 0.0f) continue;

            EmissiveTriangleRecord tri;
            tri.shape = shape;
            tri.instance_id = tlas_idx;
            tri.primitive_id = prim_id;
            tri.triangle_index = global_tri_index++;
            tri.area = area;
            tri.emission = emission;
            tri.power = power;
            tri.material_index = shape->material_id();

            _emissive_tris_cpu.push_back(tri);
            _total_power += power;
        }
        tlas_idx++;
    }

    _total_emissive_count = static_cast<uint>(_emissive_tris_cpu.size());
    _total_emissive_count_float = static_cast<float>(_total_emissive_count);
    _total_power_inv = (_total_power > 0.0f) ? (1.0f / _total_power) : 0.0f;
    _emissive_count_inv = (_total_emissive_count > 0u) ? (1.0f / _total_emissive_count_float) : 0.0f;

    // Build reverse mapping: material_index -> [record indices]
    _materialToRecords.clear();
    for (uint i = 0; i < _emissive_tris_cpu.size(); ++i) {
        _materialToRecords[_emissive_tris_cpu[i].material_index].push_back(i);
    }
}

//==============================================================================
// Triangle Area Calculation
//==============================================================================

float LightSampler::_triangle_area(const newtype::scene::MeshShape &shape, uint primitive_id) const noexcept {
    if (!shape.has_cpu_data()) {
        LUISA_WARNING("Cannot compute triangle area without CPU data");
        return 0.0f;
    }

    const auto &triangles = shape.mesh().triangles;
    const auto &vertices = shape.mesh().vertices;

    if (primitive_id >= triangles.size()) return 0.0f;

    const auto &tri = triangles[primitive_id];

    auto v0 = vertices[tri.i0].position();
    auto v1 = vertices[tri.i1].position();
    auto v2 = vertices[tri.i2].position();

    auto matrix = shape.transform()->matrix();
    auto wv0 = make_float3(matrix * make_float4(v0, 1.0f));
    auto wv1 = make_float3(matrix * make_float4(v1, 1.0f));
    auto wv2 = make_float3(matrix * make_float4(v2, 1.0f));

    float3 e1 = wv1 - wv0;
    float3 e2 = wv2 - wv0;
    float3 cross = make_float3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );
    return 0.5f * luisa::sqrt(luisa::max(1e-8f, dot(cross, cross)));
}

//==============================================================================
// Triangle Vertex Positions
//==============================================================================

void LightSampler::_get_triangle_vertices(
    const newtype::scene::MeshShape &shape, uint primitive_id,
    luisa::float3 &v0, luisa::float3 &v1, luisa::float3 &v2) const noexcept {

    if (!shape.has_cpu_data()) {
        LUISA_WARNING("Cannot get triangle vertices without CPU data");
        v0 = v1 = v2 = make_float3(0.0f);
        return;
    }

    const auto &triangles = shape.mesh().triangles;
    const auto &vertices = shape.mesh().vertices;

    if (primitive_id >= triangles.size()) {
        v0 = v1 = v2 = make_float3(0.0f);
        return;
    }

    const auto &tri = triangles[primitive_id];

    auto lv0 = vertices[tri.i0].position();
    auto lv1 = vertices[tri.i1].position();
    auto lv2 = vertices[tri.i2].position();

    auto matrix = shape.transform()->matrix();
    v0 = make_float3(matrix * make_float4(lv0, 1.0f));
    v1 = make_float3(matrix * make_float4(lv1, 1.0f));
    v2 = make_float3(matrix * make_float4(lv2, 1.0f));
}

//==============================================================================
// Alias Table Construction (Vose's Algorithm)
//==============================================================================

void LightSampler::_build_alias_table(Stream &stream) noexcept {
    uint n = _total_emissive_count;
    if (n == 0) return;

    luisa::vector<AliasEntry> alias_table(n);

    if (n == 1) {
        alias_table[0].alias_index = 0;
        alias_table[0].triangle_index = 0u;
        alias_table[0].pdf = 1.0f;
        alias_table[0].padding = 0;
        if (!_alias_table || _alias_table.size() != n)
            _alias_table = _device.create_buffer<AliasEntry>(n);
        stream << _alias_table.copy_from(alias_table.data());
        return;
    }

    float avg_power = (_uniform_sampling || _total_power == 0.0f) ? 1.0f : (_total_power / static_cast<float>(n));
    float avg_power_inv = 1.0f / avg_power;

    luisa::vector<size_t> small, large;

    for (size_t i = 0; i < n; ++i) {
        float weight = _uniform_sampling ? 1.0f : _emissive_tris_cpu[i].power;
        float scaled_power = weight * avg_power_inv;

        if (scaled_power < 1.0f) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        size_t l = small.back();
        size_t g = large.back();
        small.pop_back();
        large.pop_back();

        float l_weight = _uniform_sampling ? 1.0f : _emissive_tris_cpu[l].power;
        float g_weight = _uniform_sampling ? 1.0f : _emissive_tris_cpu[g].power;

        alias_table[l].triangle_index = static_cast<uint>(l);
        alias_table[l].alias_index = static_cast<uint>(g);
        alias_table[l].pdf = l_weight * avg_power_inv;
        alias_table[l].padding = 0;

        float new_power = (g_weight + l_weight) - avg_power;

        if (new_power < avg_power) {
            small.push_back(g);
        } else {
            large.push_back(g);
        }
    }

    for (size_t g : large) {
        alias_table[g].triangle_index = static_cast<uint>(g);
        alias_table[g].alias_index = static_cast<uint>(g);
        alias_table[g].pdf = 1.0f;
        alias_table[g].padding = 0;
    }

    for (size_t l : small) {
        alias_table[l].triangle_index = static_cast<uint>(l);
        alias_table[l].alias_index = static_cast<uint>(l);
        alias_table[l].pdf = 1.0f;
        alias_table[l].padding = 0;
    }

    if (!_alias_table || _alias_table.size() != n)
        _alias_table = _device.create_buffer<AliasEntry>(n);
    stream << _alias_table.copy_from(alias_table.data());
}

//==============================================================================
// Upload to GPU
//==============================================================================

void LightSampler::_upload_to_gpu(Stream &stream) noexcept {
    uint n = _total_emissive_count;

    uint max_inst = 0u;
    for (const auto &tri : _emissive_tris_cpu)
        max_inst = std::max(max_inst, tri.instance_id);
    _instanceCount = max_inst + 1u;

    luisa::vector<uint> inst_to_light(_instanceCount, ~0u);
    for (const auto &tri : _emissive_tris_cpu) {
        uint &base = inst_to_light[tri.instance_id];
        if (base == ~0u || tri.triangle_index < base)
            base = tri.triangle_index;
    }

    if (!_instanceToLightBase || _instanceToLightBase.size() != _instanceCount)
        _instanceToLightBase = _device.create_buffer<uint>(_instanceCount);
    stream << _instanceToLightBase.copy_from(inst_to_light.data());

    if (n == 0) {
        // Env-only scene (no emissive geometry): allocate 1-element stub
        // buffers so shader binding stays valid. Every kernel guards local-
        // light reads with `emissive_count > 0`, so these stubs are never
        // read on the GPU — they only need to be bindable.
        if (!_triangle_lights) {
            _triangle_lights   = _device.create_buffer<TriangleLight>(1u);
            _triangle_vertices = _device.create_buffer<TriangleVertexData>(1u);
            _alias_table       = _device.create_buffer<AliasEntry>(1u);
        }
        return;
    }

    luisa::vector<TriangleLight> triangle_lights(n);
    luisa::vector<TriangleVertexData> triangle_vertices(n);

    for (uint i = 0; i < n; ++i) {
        const auto &src = _emissive_tris_cpu[i];

        triangle_lights[i].instance_id    = src.instance_id;
        triangle_lights[i].primitive_id   = src.primitive_id;
        triangle_lights[i].area           = src.area;
        triangle_lights[i].pdf            = (_total_power > 0.0f) ? (src.power * _total_power_inv) : 0.0f;
        triangle_lights[i].ex             = src.emission.x;
        triangle_lights[i].ey             = src.emission.y;
        triangle_lights[i].ez             = src.emission.z;
        triangle_lights[i].nx = 0.0f;
        triangle_lights[i].ny = 0.0f;
        triangle_lights[i].nz = 0.0f;

        if (src.shape) {
            luisa::float3 v0, v1, v2;
            _get_triangle_vertices(*src.shape, src.primitive_id, v0, v1, v2);
            triangle_vertices[i].v0 = v0;
            triangle_vertices[i].v1 = v1;
            triangle_vertices[i].v2 = v2;
            luisa::float3 ln = triangle_world_normal(v0, v1, v2);
            triangle_lights[i].nx = ln.x;
            triangle_lights[i].ny = ln.y;
            triangle_lights[i].nz = ln.z;
        } else {
            triangle_vertices[i].v0 = make_float3(0.0f);
            triangle_vertices[i].v1 = make_float3(0.0f);
            triangle_vertices[i].v2 = make_float3(0.0f);
        }
    }

    if (!_triangle_lights || _triangle_lights.size() != n)
        _triangle_lights = _device.create_buffer<TriangleLight>(n);
    if (!_triangle_vertices || _triangle_vertices.size() != n)
        _triangle_vertices = _device.create_buffer<TriangleVertexData>(n);
    stream << _triangle_lights.copy_from(triangle_lights.data());
    stream << _triangle_vertices.copy_from(triangle_vertices.data());
}

//==============================================================================
// GPU Sampling Functions (DSL) — called from within compiled kernels
//==============================================================================

auto LightSampler::sample_light(const Float &u) const noexcept {
    using namespace luisa::compute;
    auto n = _total_emissive_count_float;
    Float u_scaled = u * n;
    UInt idx = cast<uint>(u_scaled);
    idx = min(idx, _total_emissive_count - 1u);
    auto entry = _alias_table->read(idx);
    UInt selected_index = ite(u_scaled - cast<float>(idx) < entry.pdf,
        entry.triangle_index, entry.alias_index);
    Float pdf = entry.pdf * _total_power_inv;
    return std::make_pair(selected_index, pdf);
}

auto LightSampler::sample_light_with_uv(const Float2 &u) const noexcept {
    using namespace luisa::compute;
    auto n = _total_emissive_count_float;
    Float u_scaled = u.x * n;
    UInt idx = cast<uint>(u_scaled);
    idx = min(idx, _total_emissive_count - 1u);
    auto entry = _alias_table->read(idx);
    UInt selected_index = ite(u_scaled - cast<float>(idx) < entry.pdf,
        entry.triangle_index, entry.alias_index);
    Float su = sqrt(u.y);
    Float2 uv = make_float2(1.0f - su, u.x * su);
    return std::make_pair(selected_index, uv);
}

Float LightSampler::light_pdf(const UInt &triangle_index) const noexcept {
    if (_total_emissive_count == 0u) return 0.0f;
    return _total_power_inv;
}

auto LightSampler::sample_env(const Float2 &u) const noexcept {
    // Placeholder — actual sampling happens inline in PipelineDI.cpp kernels
    // using the CDF buffers passed as shader parameters
    using namespace luisa::compute;
    Float phi = 2.0f * 3.14159265359f * u.x;
    Float cos_theta = 1.0f - 2.0f * u.y;
    Float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    Float3 dir = make_float3(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
    Float pdf = 1.0f / (4.0f * 3.14159265359f);
    return std::make_pair(dir, pdf);
}

auto LightSampler::env_pdf(const Float3 &direction) const noexcept {
    using namespace luisa::compute;
    return 1.0f / (4.0f * 3.14159265359f);
}

auto LightSampler::eval_env(const Float3 &direction) const noexcept {
    using namespace luisa::compute;
    return make_float3(0.0f);
}

//==============================================================================
// Triangle Sampling Helper
//==============================================================================

Float3 LightSampler::_sample_triangle(
    const Float3 &p0, const Float3 &p1, const Float3 &p2,
    const Float2 &u) noexcept {
    Float su = sqrt(u.x);
    Float b0 = 1.0f - su;
    Float b1 = u.y * su;
    Float b2 = 1.0f - b0 - b1;
    return b0 * p0 + b1 * p1 + b2 * p2;
}

//==============================================================================
// Legacy/Compatibility Functions
//==============================================================================

Buffer<EmissiveTriangle> create_emissive_buffer(
    Device& device,
    Buffer<util::Vertex> vertices,
    uint triangleCount,
    const luisa::float3& emissionThreshold) {
    luisa::vector<EmissiveTriangle> emissives;
    emissives.reserve(triangleCount);
    LUISA_INFO("Creating emissive buffer for {} triangles", triangleCount);
    return device.create_buffer<EmissiveTriangle>(
        std::max(emissives.size(), size_t{1}));
}

} // namespace newtype::render
