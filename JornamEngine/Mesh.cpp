#include "headers.h"

namespace JornamEngine {

	/*
		Returns mesh index if hashed, 0 otherwise
	
		@param meshID	The string that identifies this mesh
		@return			The index of the mesh
	*/
	uint MeshMap::get(const char* meshID)
	{
		uint size = (uint)m_hashes.size();
		for (size_t i = 0; i < size; i++)
		{
			if (m_hashes[i] == meshID) return (uint)i;
		}
		return 0;
	}

	/*
		Adds the mesh to the heap on both host and device, and returns the index
		Also initializes and updates the prime Triangle Model

		@param meshID	The string that identifies this mesh
		@param mesh		The mesh object
		@return			The index of the mesh
	*/
	uint MeshMap::add(const char* meshID, std::vector<float> vertices, std::vector<uint> indices, std::vector<float> normals)
	{
		m_hashes.push_back(meshID);
		m_meshes->push_back(Mesh(indices, normals));
		uint meshIdx = (uint)m_meshes->size() - 1;
		CudaMesh cudaMesh = CudaMesh(indices, normals);
		cudaMemcpy(c_meshes + meshIdx, &cudaMesh, sizeof(CudaMesh), cudaMemcpyHostToDevice);

		optix::prime::Model newModel = m_context->createModel();
		newModel->setTriangles(
			indices.size() / 3, RTP_BUFFER_TYPE_HOST, indices.data(), // TODO correct buffer type?
			vertices.size() / 3, RTP_BUFFER_TYPE_HOST, vertices.data()
		);
		newModel->update(0);
		m_optixModels->push_back(newModel);

		return meshIdx;
	}

	/*
		Copies mesh data to host heap

		@param indices	A vector of 3 consecutive indices per triangle
		@param normals	A vector of 3 consecutive floats per vertex
	*/
	void Mesh::makeHostPtr(std::vector<uint> indices, std::vector<float> normals)
	{
		m_indices = (int3*)malloc(indices.size() * sizeof(uint));
		memcpy(m_indices, indices.data(), indices.size() * sizeof(uint));
		m_normals = (float3*)malloc(normals.size() * sizeof(float));
		memcpy(m_normals, normals.data(), normals.size() * sizeof(float));
	}

	/*
		Copies mesh data to device heap

		@param indices	A vector of 3 consecutive indices per triangle
		@param normals	A vector of 3 consecutive floats per vertex
	*/
	void CudaMesh::makeDevicePtr(std::vector<uint> indices, std::vector<float> normals)
	{
		cudaMalloc(&m_indices, indices.size() * sizeof(uint));
		cudaMalloc(&m_normals, normals.size() * sizeof(float));
		cudaMemcpy(m_indices, indices.data(), indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
		cudaMemcpy(m_normals, normals.data(), normals.size() * sizeof(float), cudaMemcpyHostToDevice);
	}

}