/**
	file:			Scene.cpp
	last modified:	07-05-2019
	description:	Provides a Scene object that holds the triangles, lights,
					and skybox. Scenes can be loaded from custom .scene files.

@author Joram Wessels
@version 0.1
*/
#include "headers.h"

namespace JornamEngine {

/*
	Loads a scene from a .scene file
	
	@param filename	The path to the .scene file
	@param camera	(optional) A pointer to the camera object
*/
void Scene::loadScene(const char* filename, Camera *camera)
{
	SceneParser parser = SceneParser(this);
	parser.parseScene(filename, camera);

	m_model->setInstances(
		m_rtpModels.size(), RTP_BUFFER_TYPE_HOST, &m_rtpModels[0],
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, &m_transforms[0]
	);
	m_model->update(0);

	cudaMalloc(&c_lights, m_lights.size() * sizeof(Light));
	cudaMalloc(&c_meshes, m_meshes.size() * sizeof(Mesh));
	cudaMalloc(&c_objects, m_objects.size() * sizeof(Object3D));
	cudaMemcpy(c_lights, m_lights.data(), m_lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
	cudaMemcpy(c_meshes, m_meshes.data(), m_meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);
	cudaMemcpy(c_objects, m_objects.data(), m_objects.size() * sizeof(Object3D), cudaMemcpyHostToDevice);
}

/*
	Reads a mesh from a .obj file and adds it to the object queue

	@param filename		The path to the .obj file
	@param transform	A Transform struct with the initial position/rotation/scale
*/
void Scene::readMesh(const char* filename, Transform transform, Color color)
{
	uint meshIdx;
	if (!(meshIdx = m_meshMap.get(filename)))
	{
		// Reading .obj using tinyobjloader
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		tinyobj::LoadObj(shapes, materials, err, filename);
		if (!err.empty()) logDebug("Scene",
			(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
			JornamException::ERR);
		bool onDevice = (m_buffertype == RTP_BUFFER_TYPE_CUDA_LINEAR);
		meshIdx = m_meshMap.add(filename, shapes[0].mesh.positions, shapes[0].mesh.indices, shapes[0].mesh.normals, onDevice);
	}

	//addObject(shapes[0].mesh.positions, shapes[0].mesh.indices, N, transform, 0xBBBBBB);
	PhongMaterial material(0.3f, 0.3f, 1.0f, 10.0f);
	Object3D object(m_optixModels[meshIdx], meshIdx, transform, color, material);
	m_objects.push_back(object);
	m_rtpModels.push_back(object.getRTPmodel());
	m_transforms.push_back(transform.matrix);
}

/*
	Interpolates triangle surface normal given Barycentric coordinates

	@param trIdx	The triangle index
	@param obIdx	The object index
	@param u		The Barycentric u coordinate
	@param v		The barycentric v coordinate
	@return			The surface normal at the given coordinates
*/
vec3 Scene::interpolateNormal(uint o, uint t, float u, float v) const
{
	const int3* indices = m_meshes[m_objects[o].getMeshIdx()].getIndices();
	const float3* normals = m_meshes[m_objects[o].getMeshIdx()].getNormals();
	uint v0 = indices[t].x, v1 = indices[t].y, v2 = indices[t].z;
	vec3 n0 = normals[v0], n1 = normals[v1], n2 = normals[v2];
	return (n0 * u + n1 * v + n2 * (1 - u - v)).normalized();
}

} // namespace Engine