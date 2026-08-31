#pragma once
// Created by Seph Li on 2026/03/23.

#include <memory>
#include "newtype/core/Renderer.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/util/Camera.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/TypeConv.h"
#include "newtype/core/Pipeline.h"

// Create a namepace alias as shorthand for newtype::
namespace nt = newtype;

namespace newtype {
	using ImgFlt  = luisa::compute::Image<float>;
	using ImgUint = luisa::compute::Image<luisa::uint>;
	using ImgByte = luisa::compute::Image<luisa::byte>;
	
	template<typename T>
	using Data	  = luisa::compute::Buffer<T>;

	using Vert    = scene::MeshShape::Vertex;
	using Tri	  = luisa::compute::Triangle;

	using flt2 = luisa::float2;
	using flt3 = luisa::float3;
	using flt4 = luisa::float4;
}