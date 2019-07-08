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
	@param attrib	The tinyobj attribute object containing vertices, normals, and texcoords
	@param mesh		The mesh object
	@param onDevice	A bool indicating where to load the mesh (CPU/GPU)
	@return			The index of the mesh
*/
uint MeshMap::add(const char* meshID, tinyobj::attrib_t attrib, tinyobj::mesh_t mesh, bool onDevice)
{
	// Adding mesh to hash map
	m_hashes.push_back(meshID);
	m_meshes->push_back(Mesh(mesh.indices, attrib.vertices, attrib.normals, attrib.texcoords, onDevice));
	uint meshIdx = (uint)m_meshes->size() - 1;

	// Extracting contiguous vertex indices from mesh.indices
	std::vector<int> indices;
	for (int i = 0; i < mesh.indices.size(); i++)
		indices.push_back(mesh.indices[i].vertex_index);

	// Adding model to Optix context
	optix::prime::Model newModel = m_context->createModel();
	newModel->setTriangles(
		indices.size() / 3, RTP_BUFFER_TYPE_HOST, indices.data(), // TODO correct buffer type?
		attrib.vertices.size() / 3, RTP_BUFFER_TYPE_HOST, attrib.vertices.data()
	);
	newModel->update(0);
	m_optixModels->push_back(newModel);

	return meshIdx;
}

/*
	Copies mesh data to host heap

	@param indices		A vector of 9 consecutive indices per triangle (vertex, normal, texture)
	@param vertices		A vector of 3 consecutive floats per vertex
	@param normals		A vector of 3 consecutive floats per vertex
	@param texcoords	A vector of 2 consecutive floats per vertex
*/
void Mesh::makeHostPtr(std::vector<tinyobj::index_t> indices, std::vector<float> vertices, std::vector<float> normals, std::vector<float> texcoords)
{
	m_indices = (Index*)malloc(indices.size() * sizeof(tinyobj::index_t));
	memcpy(m_indices, indices.data(), indices.size() * sizeof(tinyobj::index_t));
	m_vertices = (float3*)malloc(vertices.size() * sizeof(float));
	memcpy(m_vertices, vertices.data(), vertices.size() * sizeof(float));
	m_normals = (float3*)malloc(normals.size() * sizeof(float));
	memcpy(m_normals, normals.data(), normals.size() * sizeof(float));
	m_texcoords = (float2*)malloc(texcoords.size() * sizeof(float));
	memcpy(m_texcoords, texcoords.data(), texcoords.size() * sizeof(float));
}

/*
	Copies mesh data to device heap

	@param indices		A vector of 9 consecutive indices per triangle (vertex, normal, texture)
	@param vertices		A vector of 3 consecutive floats per vertex
	@param normals		A vector of 3 consecutive floats per vertex
	@param texcoords	A vector of 2 consecutive floats per vertex
*/
void Mesh::makeDevicePtr(std::vector<tinyobj::index_t> indices, std::vector<float> vertices, std::vector<float> normals, std::vector<float> texcoords)
{
	cudaMalloc(&m_indices, indices.size() * sizeof(tinyobj::index_t));
	cudaMemcpy(m_indices, indices.data(), indices.size() * sizeof(tinyobj::index_t), cudaMemcpyHostToDevice);
	cudaMalloc(&m_vertices, vertices.size() * sizeof(float));
	cudaMemcpy(m_vertices, vertices.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&m_normals, normals.size() * sizeof(float));
	cudaMemcpy(m_normals, normals.data(), normals.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&m_texcoords, texcoords.size() * sizeof(float));
	cudaMemcpy(m_texcoords, texcoords.data(), texcoords.size() * sizeof(float), cudaMemcpyHostToDevice);
}

}