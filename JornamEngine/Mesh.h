#pragma once

namespace JornamEngine {

/*
	Represents a mesh, containing vertex indices and normals

	@param indices	A vector of 3 consecutive vertex indices per triangle
	@param normals	A vector of 3 consecutive floats per vertex
*/
class Mesh
{
public:
	Mesh() { m_indices = 0; m_normals = 0; };
	Mesh(std::vector<uint> indices, std::vector<float> normals) { makeHostPtr(indices, normals); }
	//~Mesh() { if (m_indices) freeHostPtr(); };
	const int3* getIndices() const { return m_indices; }
	const float3* getNormals() const { return m_normals; }
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
	const int3* getIndices() const { return m_indices; }
	const float3* getNormals() const { return m_normals; }
private:
	int3* m_indices;
	float3* m_normals;
	void makeDevicePtr(std::vector<uint> indices, std::vector<float> normals);
	void freeDevicePtr() { cudaFree(m_indices); cudaFree(m_normals); }
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
	MeshMap(optix::prime::Context context, std::vector<Mesh>* meshes, CudaMesh* c_meshes, std::vector<optix::prime::Model>* optixModels, uint initDeviceCapacity=10)
		: m_context(context), m_meshes(meshes), c_meshes(c_meshes), m_optixModels(optixModels)
	{
		m_hashes.push_back("NULL HASH");
		m_meshes->push_back(Mesh());
		m_optixModels->push_back(optix::prime::Model());
		cudaMalloc(&c_meshes, initDeviceCapacity+1 * sizeof(CudaMesh));
	}
	uint get(const char* meshID);
	uint add(const char* meshID, std::vector<float> positions, std::vector<uint> indices, std::vector<float> normals);
private:
	optix::prime::Context m_context;
	std::vector<std::string> m_hashes;					// starts at idx 1
	std::vector<Mesh>* m_meshes;						// starts at idx 1
	CudaMesh* c_meshes;									// starts at idx 1
	std::vector<optix::prime::Model>* m_optixModels;	// starts at idx 1
};

} // namespace Engine