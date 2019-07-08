#pragma once

namespace JornamEngine {

struct Index { int vertexIdx, normalIdx, textureIdx; };

/*
	Represents a mesh, containing pointers to the vertex indices and normals

	@param indices	A vector of 3 consecutive vertex indices per triangle
	@param normals	A vector of 3 consecutive floats per vertex
*/
class Mesh
{
public:
	Mesh() { m_indices = 0; m_normals = 0; };
	Mesh(std::vector<tinyobj::index_t> indices, std::vector<float> vertices, std::vector<float> normals, std::vector<float> texcoords, bool onDevice)
	{
		if (!onDevice) makeHostPtr(indices, vertices, normals, texcoords);
		if (onDevice) makeDevicePtr(indices, vertices, normals, texcoords);
	}
	//~Mesh() { if (m_indices) freeHostPtr(); };
	const Index* getIndices() const { return m_indices; }
	const float3* getVertices() const { return m_vertices; }
	const float3* getNormals() const { return m_normals; }
	const float2* getTexcoords() const { return m_texcoords; }
protected:
	Index* m_indices;
	float3* m_vertices;
	float3* m_normals;
	float2* m_texcoords;

	void makeHostPtr(std::vector<tinyobj::index_t> indices, std::vector<float> vertices, std::vector<float> normals, std::vector<float> texcoords);
	void makeDevicePtr(std::vector<tinyobj::index_t> indices, std::vector<float> vertices, std::vector<float> normals, std::vector<float> texcoords);
	inline void freeHostPtr() { free(m_indices); free(m_vertices); free(m_normals); free(m_texcoords); }
	inline void freeDevicePtr() { cudaFree(m_indices); cudaFree(m_vertices); cudaFree(m_normals); cudaFree(m_texcoords); }
};

/*
	Prevents Mesh duplicates by hashing the filenames
	The add function takes care of keeping the GPU and host in sync

	@param context	The optix context
	@param meshes	A pointer to the vector of host meshes
	@param c_meshes	A pointer to the array of device meshes
	@param initDeviceCapacity	The number of indices allocated on the device
*/
class MeshMap
{
public:
	MeshMap(optix::prime::Context context, std::vector<Mesh>* meshes, std::vector<optix::prime::Model>* optixModels)
		: m_context(context), m_meshes(meshes), m_optixModels(optixModels)
	{
		m_hashes.push_back("NULL HASH");
		m_meshes->push_back(Mesh());
		m_optixModels->push_back(optix::prime::Model());
	}
	uint get(const char* meshID);
	uint add(const char* meshID, tinyobj::attrib_t attrib, tinyobj::mesh_t mesh, bool onDevice);
private:
	optix::prime::Context m_context;
	std::vector<std::string> m_hashes;					// starts at idx 1
	std::vector<Mesh>* m_meshes;						// starts at idx 1
	std::vector<optix::prime::Model>* m_optixModels;	// starts at idx 1
};

} // namespace Engine