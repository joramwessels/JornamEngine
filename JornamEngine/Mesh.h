#pragma once

namespace JornamEngine {

/*
	Represents a mesh, containing vertex indices and normals
*/
class Mesh
{
public:
	Mesh() { m_indices = 0; m_normals = 0; };
	Mesh(std::vector<uint> indices, std::vector<float> normals) { makeHostPtr(indices, normals); }
	//~Mesh() { if (m_indices) freeHostPtr(); };
	int3* getIndices() { return m_indices; }
	float3* getNormals() { return m_normals; }
protected:
	int3* m_indices;
	float3* m_normals;

	void makeHostPtr(std::vector<uint> indices, std::vector<float> normals);
	void freeHostPtr() { free(m_indices); free(m_normals); }
};
__device__ class CudaMesh
{
public:
	CudaMesh() { m_indices = 0; m_normals = 0; };
	CudaMesh(std::vector<uint> indices, std::vector<float> normals) { makeDevicePtr(indices, normals); }
	//~Mesh() { if (m_indices) freeDevicePtr(); };
	int3* getIndices() { return m_indices; }
	float3* getNormals() { return m_normals; }
private:
	int3* m_indices;
	float3* m_normals;
	void makeDevicePtr(std::vector<uint> indices, std::vector<float> normals);
	void freeDevicePtr() { cudaFree(m_indices); cudaFree(m_normals); }
};

/*
	Prevents Mesh duplicates by hashing the filenames
*/
class MeshMap
{
public:
	MeshMap(optix::prime::Context context, uint initDeviceCapacity=10)
	{
		m_context = context;
		m_hashes.push_back("NULL HASH");
		m_meshes.push_back(Mesh());
		m_rtpModels.push_back(optix::prime::Model());
		cudaMalloc(&c_meshes, initDeviceCapacity+1 * sizeof(CudaMesh));
	}
	uint get(const char* meshID);
	uint add(const char* meshID, std::vector<float> positions, std::vector<uint> indices, std::vector<float> normals);
	optix::prime::Model getRTPModel(uint idx) { return m_rtpModels[idx]; }
	Mesh* getHostMeshes() { return m_meshes.data(); }
	CudaMesh* getDeviceMeshes() { return c_meshes; }
private:
	optix::prime::Context m_context;
	std::vector<const char*> m_hashes;
	std::vector<Mesh> m_meshes;
	CudaMesh* c_meshes;
	std::vector<optix::prime::Model> m_rtpModels;
};

} // namespace Engine