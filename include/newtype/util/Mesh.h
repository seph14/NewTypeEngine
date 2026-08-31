#pragma once
#include "cinder/TriMesh.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include "newtype/util/Vertex.h"

namespace newtype::util {
	class Mesh;
	typedef std::shared_ptr<Mesh> MeshRef;

	class Mesh {
	protected:
		luisa::compute::Mesh					 mMesh;
		luisa::compute::Buffer<Vertex>			 mVertexBuffer;     // Now uses new Vertex (32 bytes)
		luisa::compute::Buffer<luisa::float3>	 mPositionBuffer;    // Legacy position-only buffer
		luisa::compute::Buffer<uint>			 mMaterialIndices;  // Per-primitive material index
		bool									 mPacked, mRequirePosBuffer;

		Mesh() : mPacked(false), mRequirePosBuffer(false) {}

	public:
		static MeshRef create() { return MeshRef( new Mesh() ); }
		~Mesh();

		void setPosBufferRequired(bool value) { mRequirePosBuffer = value; }
		 
		/**
		* @brief pack Cinder Mesh into LuisaCompute mesh strcture for ray tracing
		* @param mesh source TriMesh
		*/
		[[nodiscard]] luisa::compute::Mesh& pack(ci::TriMesh& mesh);

		/**
		* @brief pack Cinder Geom into LuisaCompute mesh strcture for ray tracing
		* @param geom source geometry
		*/
		[[nodiscard]] luisa::compute::Mesh& pack(ci::geom::Source& geom);

		/**
		* @brief Check if mesh has been packed
		*/
		[[nodiscard]] bool packed() const noexcept { return mPacked; }

		/**
		* @brief Get the LuisaCompute Mesh (for Accel building)
		*/
		[[nodiscard]] luisa::compute::Mesh& mesh() noexcept { return mMesh; }

		/**
		* @brief Get the full vertex buffer with all attributes (position, normal, uv)
		* @note Uses new Vertex format (32 bytes) from Vertex.h
		*/
		[[nodiscard]] luisa::compute::Buffer<Vertex>& vertexBuffer() noexcept { return mVertexBuffer; }

		/**
		* @brief Get legacy position-only vertex buffer
		* @deprecated Use vertexBuffer() for full attribute access
		*/
		[[nodiscard]] luisa::compute::Buffer<luisa::float3>& verts() noexcept { return mPositionBuffer; }

		//==========================================================================
		// Material Per-Primitive
		//==========================================================================

		/**
		* @brief Set material index for a specific primitive
		* @param primIdx Primitive index
		* @param materialIdx Material index from MaterialPool
		*/
		void setMaterialIndex(uint primIdx, uint materialIdx);

		/**
		* @brief Set material indices for all primitives (single material)
		* @param materialIdx Material index from MaterialPool
		*/
		void setAllMaterialIndices(uint materialIdx);

		/**
		* @brief Get per-primitive material indices buffer
		* @note Buffer is created on first call (lazy initialization)
		*/
		[[nodiscard]] luisa::compute::Buffer<uint>& materialIndices() noexcept { return mMaterialIndices; }

		/**
		* @brief Get material index count (same as triangle count)
		*/
		[[nodiscard]] uint materialIndexCount() const noexcept;
	};
}