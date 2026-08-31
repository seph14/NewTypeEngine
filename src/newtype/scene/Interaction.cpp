//
// Created by Claude on 2026/03/23.
//

#include "newtype/scene/Interaction.h"

namespace newtype::scene {

//==============================================================================
// Interaction
//==============================================================================

Var<Ray> Interaction::spawn_ray(Expr<float3> wi, Expr<float> t_max) const noexcept {
    static constexpr auto ray_eps = 1e-4f;  // Ray epsilon for self-intersection avoidance
    // Use ite() for conditional: origin = valid && !back_facing ? (p + ng * eps) : (p - ng * eps)
    auto origin = ite(valid() & !back_facing(),
                       p() + ng() * ray_eps,           // Hit front face: offset along normal
                       p() - ng() * ray_eps);           // Hit back face: offset against normal
    return make_ray(origin, wi, ray_eps, t_max);
}

Var<Ray> Interaction::spawn_ray_to(Expr<float3> p) const noexcept {
    auto dir = p - this->p();
    auto len = length(dir);
    return spawn_ray(dir / len, len);
}

} // namespace newtype::scene
