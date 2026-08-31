#pragma once

/**
 * @brief Custom Material Shader — DLL header
 *
 * In Debug/Release (static): no-op, callables registered inline in app code.
 * In Debug_Runtime: DLL loaded by CallableDLLLoader for hot-reload.
 */

#include "CustomMaterialShaderAPI.h"
