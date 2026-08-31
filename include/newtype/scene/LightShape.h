//
// Created by Claude on 2026/03/24.
//

#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/scene/MeshShape.h"

namespace newtype::scene {

using namespace luisa;

/**
 * @brief Emissive triangle mesh (marker class)
 *
 * A MeshShape that represents an area light source.
 * Emission is stored solely in the material (MaterialData::emission).
 * This class exists only to set the PROPERTY_HAS_LIGHT flag so the
 * LightSampler can identify light geometry.
 */
class LightShape : public MeshShape {
public:
    LightShape(Device &device, uint material_id) noexcept
        : MeshShape(device, material_id) {

        // Mark as light in properties
        _properties |= PROPERTY_HAS_LIGHT;
    }
};

typedef luisa::unique_ptr<LightShape> LightShapePtr;

/**
 * @brief Helper to create LightShape from Cinder geometry
 */
inline luisa::unique_ptr<LightShape> make_light(
    Device &device, ci::TriMesh &triMesh,
    uint material_id) noexcept
{
    auto light = luisa::make_unique<LightShape>(device, material_id);
    light->load_from(triMesh);
    return light;
}

} // namespace newtype::scene
