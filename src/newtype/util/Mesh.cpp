#include "newtype/util/Mesh.h"
#include "cinder/CinderImGui.h"
#include "newtype/core/Renderer.h"
#include "cinder/Log.h"

namespace newtype::util {
	using namespace ci;
	using namespace luisa;
	using namespace luisa::compute;

	Mesh::~Mesh() {
		if (mPacked) {
			mMesh.release();
			mVertexBuffer.release();
			mPositionBuffer.release();
			mMaterialIndices.release();
		}
	}

	namespace {

		// Helper to convert vec3 to luisa::float3 (handles 16-byte alignment)
		luisa::float3 toFloat3(const ci::vec3& v) noexcept {
			return luisa::make_float3(v.x, v.y, v.z);
		}

		// Helper to convert vec2 to luisa::float2
		luisa::float2 toFloat2(const ci::vec2& v) noexcept {
			return luisa::make_float2(v.x, v.y);
		}

	} // namespace

	luisa::compute::Mesh& Mesh::pack(ci::TriMesh& triMesh) {
		auto& device = core::Renderer::device();

		uint32_t numVert = triMesh.getNumVertices();
		uint32_t numTri  = triMesh.getNumTriangles();

		CI_LOG_D("Packing mesh: " << numVert << " vertices, " << numTri << " triangles");

		//----------------------------------------------------------------------
		// Ensure required attributes exist
		//----------------------------------------------------------------------
		if (!triMesh.hasNormals()) {
			CI_LOG_D("No normals found, calculating...");
			triMesh.recalculateNormals(true, false);  // smooth=true, weighted=false
		}

		//----------------------------------------------------------------------
		// Get pointers to vertex attribute arrays
		//----------------------------------------------------------------------
		const ci::vec3* positions = triMesh.getPositions<3>();
		const std::vector<ci::vec3>& normals = triMesh.getNormals();
		const ci::vec2* texCoords = triMesh.hasTexCoords0() ? triMesh.getTexCoords0<2>() : nullptr;

		//----------------------------------------------------------------------
		// Create Vertex array using new Vertex format (32 bytes)
		//----------------------------------------------------------------------
		std::vector<Vertex> vertices(numVert);

		for (uint32_t i = 0; i < numVert; i++) {
			// Use Vertex::encode() for compact storage
			vertices[i] = Vertex::encode(
				toFloat3(positions[i]),      // position
				toFloat3(normals[i]),         // normal
				luisa::make_float4(1.f, 0.f, 0.f, 1.f),  // default tangent
				texCoords ? toFloat2(texCoords[i]) : luisa::make_float2(0.0f, 0.0f)  // uv
			);
		}

		//----------------------------------------------------------------------
		// Triangles from the index stream, then weld bit-identical duplicates
		// (A1) — bit-identical rendering, smaller buffers.
		//----------------------------------------------------------------------
		std::vector<Triangle> triangles(numTri);
		const auto& indices = triMesh.getIndices();
		for (uint32_t i = 0; i < numTri; i++) {
			triangles[i] = Triangle{ indices[3u * i + 0u], indices[3u * i + 1u], indices[3u * i + 2u] };
		}
		if (const size_t removed = weld_vertices(vertices, triangles); removed > 0u) {
			numVert = static_cast<uint32_t>(vertices.size());
			CI_LOG_I("Mesh::pack: welded " << removed << " duplicate vertices ("
				<< numVert << " remain)");
		}

		//----------------------------------------------------------------------
		// Create buffers
		//----------------------------------------------------------------------
		mVertexBuffer   = device.create_buffer<util::ActiveVertex>(numVert);
		compute::Buffer<Triangle> triangleBuffer = device.create_buffer<Triangle>(numTri);
		auto& stream = core::Renderer::stream();

		// A2: upload stages through the active GPU layout (fp32 identity).
		util::upload_vertex_buffer(
			mVertexBuffer, std::span<const Vertex>{vertices}, stream);

		// Extract positions for legacy buffer
		if (mRequirePosBuffer) {
			mPositionBuffer = device.create_buffer<luisa::float3>(numVert);  // Legacy support
			std::vector<luisa::float3> positionsOnly(numVert);
			for (uint32_t i = 0; i < numVert; i++)
				positionsOnly[i] = vertices[i].position();
			stream << mPositionBuffer.copy_from(positionsOnly.data())
				<< triangleBuffer.copy_from(triangles.data())
				<< synchronize();
		} else {
			stream << triangleBuffer.copy_from(triangles.data())
				<< synchronize();
		}

		//----------------------------------------------------------------------
		// Create LuisaCompute mesh with new Vertex format
		//----------------------------------------------------------------------
		mMesh = device.create_mesh(mVertexBuffer, triangleBuffer);
		stream << mMesh.build() << synchronize();

		mPacked = true;
		CI_LOG_D("Mesh packed successfully with " << sizeof(util::ActiveVertex) << " byte vertices");
		return mMesh;
	}

	luisa::compute::Mesh& Mesh::pack(ci::geom::Source& geom) {
		ci::TriMesh triMesh = ci::TriMesh(geom);
		return pack(triMesh);
	}

	//==========================================================================
	// Material Per-Primitive Methods
	//==========================================================================

	void Mesh::setMaterialIndex(uint primIdx, uint materialIdx) {
		if (!mMaterialIndices) {
			// Lazy initialization: create buffer on first use
			auto& device = core::Renderer::device();
			uint triangleCount = mMesh.triangle_count();
			mMaterialIndices = device.create_buffer<uint>(triangleCount);

			// Initialize with default material (0)
			std::vector<uint> indices(triangleCount, 0u);
			core::Renderer::stream() << mMaterialIndices.copy_from(indices.data()) << synchronize();
		}

		// Upload single index - need to copy entire buffer for this single update
		// TODO: Implement a more efficient approach (e.g., Command::copy_to_buffer)
		// For now, this is inefficient but functional
		uint triangleCount = mMesh.triangle_count();
		std::vector<uint> indices(triangleCount);
		mMaterialIndices.copy_to(indices.data());
		indices[primIdx] = materialIdx;
		mMaterialIndices.copy_from(indices.data());
		core::Renderer::stream() << synchronize();
	}

	void Mesh::setAllMaterialIndices(uint materialIdx) {
		if (!mMaterialIndices) {
			auto& device = core::Renderer::device();
			uint triangleCount = mMesh.triangle_count();
			mMaterialIndices = device.create_buffer<uint>(triangleCount);
		}

		uint triangleCount = mMesh.triangle_count();
		std::vector<uint> indices(triangleCount, materialIdx);
		core::Renderer::stream() << mMaterialIndices.copy_from(indices.data()) << synchronize();
	}

	uint Mesh::materialIndexCount() const noexcept {
		return mMesh.triangle_count();
	}
}
