#pragma once

namespace JornamEngine {

/*
	Represents a mesh, containing pointers to the vertex indices and normals

	@param indices	A vector of 3 consecutive vertex indices per triangle
	@param normals	A vector of 3 consecutive floats per vertex
*/
class Mesh
{
public:
	Mesh() { m_indices = 0; m_normals = 0; };
	Mesh(std::vector<uint> indices, std::vector<float> normals, bool onDevice)
	{
		if (!onDevice) makeHostPtr(indices, normals);
		if (onDevice) makeDevicePtr(indices, normals);
	}
	//~Mesh() { if (m_indices) freeHostPtr(); };
	const int3* getIndices() const { return m_indices; }
	const float3* getNormals() const { return m_normals; }
protected:
	int3* m_indices;
	float3* m_normals;

	void makeHostPtr(std::vector<uint> indices, std::vector<float> normals);
	void makeDevicePtr(std::vector<uint> indices, std::vector<float> normals);
	void freeHostPtr() { free(m_indices); free(m_normals); }
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
	MeshMap(optix::prime::Context context, std::vector<Mesh>* meshes, std::vector<optix::prime::Model>* optixModels)
		: m_context(context), m_meshes(meshes), m_optixModels(optixModels)
	{
		m_hashes.push_back("NULL HASH");
		m_meshes->push_back(Mesh());
		m_optixModels->push_back(optix::prime::Model());
	}
	uint get(const char* meshID);
	uint add(const char* meshID, std::vector<float> positions, std::vector<uint> indices, std::vector<float> normals, bool onDevice);
private:
	optix::prime::Context m_context;
	std::vector<std::string> m_hashes;					// starts at idx 1
	std::vector<Mesh>* m_meshes;						// starts at idx 1
	std::vector<optix::prime::Model>* m_optixModels;	// starts at idx 1
};

} // namespace Engine