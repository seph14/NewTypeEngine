// Precompiled header for the CustomMaterialShader hot-reload DLL.
// The heavy template headers (engine Shading.h + LuisaCompute DSL) dominate
// the single-TU compile (~9s without a PCH). Compiled once via pch.cpp; the
// editable CustomMaterialShader.cpp consumes it through ForcedIncludeFiles,
// so resolvers keep their own include list untouched.
#pragma once

#include "CustomMaterialShaderAPI.h"
#include "newtype/render/Shading.h"
#include <luisa/dsl/sugar.h>
#include <cstdio>
