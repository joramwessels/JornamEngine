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

	// Moving assets to device
	if (m_buffertype == RTP_BUFFER_TYPE_CUDA_LINEAR)
	{
		cudaMalloc(&c_lights, m_lights.size() * sizeof(Light));
		cudaMalloc(&c_meshes, m_meshes.size() * sizeof(Mesh));
		cudaMalloc(&c_textures, m_textures.size() * sizeof(Texture));
		cudaMalloc(&c_objects, m_objects.size() * sizeof(Object3D));
		cudaMemcpy(c_lights, m_lights.data(), m_lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
		cudaMemcpy(c_meshes, m_meshes.data(), m_meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);
		cudaMemcpy(c_textures, m_textures.data(), m_textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
		cudaMemcpy(c_objects, m_objects.data(), m_objects.size() * sizeof(Object3D), cudaMemcpyHostToDevice);
	}
}

/*
	Reads a mesh from a .obj file and adds it to the object queue

	@param filename		The path to the .obj file
	@param transform	A Transform struct with the initial position/rotation/scale
*/
void Scene::readMesh(const char* filename, Transform transform, Color color)
{
	uint meshIdx, textureIdx;
	if (!(meshIdx = m_meshMap.get(filename)))
	{
		// Reading .obj using tinyobjloader
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		tinyobj::LoadObj(shapes, materials, err, filename);
		if (!err.empty()) logger.logDebug("Scene",
			(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
			JornamException::ERR);
		bool onDevice = (m_buffertype == RTP_BUFFER_TYPE_CUDA_LINEAR);
		meshIdx = m_meshMap.add(filename, shapes[0].mesh.positions, shapes[0].mesh.indices, shapes[0].mesh.normals, onDevice);
		logger.logDebug("Scene", ("Loaded mesh " + std::to_string(meshIdx) +
			": \"" + std::string(filename) + "\"").c_str(), JornamException::INFO);
	}
	char colorHex[8];
	sprintf(colorHex, "%x", color.hex);
	if (!(textureIdx = m_textureMap.get(colorHex)))
	{
		// Reading texture
		textureIdx = m_textureMap.add(colorHex, color);
		logger.logDebug("Scene", ("Loaded texture " + std::to_string(textureIdx) +
			": \"" + colorHex + "\"").c_str(), JornamException::INFO);
	}

	// Adding object
	PhongMaterial material(0.3f, 0.3f, 1.0f, 10.0f);
	Object3D object(m_optixModels[meshIdx], meshIdx, textureIdx, transform, material);
	m_objects.push_back(object);
	m_rtpModels.push_back(object.getRTPmodel());
	m_transforms.push_back(transform.matrix);
	logger.logDebug("Scene", ("Added object " + std::to_string(m_objects.size() - 1) +
		": (mesh: " + std::to_string(meshIdx) + ", texture: " + std::to_string(textureIdx) + ")").c_str(),
		JornamException::INFO);
}

/*
	Interpolates triangle surface normal given Barycentric coordinates

	@param obIdx	The object index
	@param trIdx	The triangle index
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

/*
	Interpolates textures given Barycentric coordinates

	@param obIdx	The object index
	@param trIdx	The triangle index
	@param u		The barycentric u coordinate
	@param v		The barycentric v coordinate
	@return			The texture color at this location
*/
Color Scene::interpolateTexture(uint o, uint t, float u, float v) const
{
	Texture texture = m_textures[m_objects[o].getTextureIdx()];
	if (texture.isSolidColor()) return texture.getColor();
}

} // namespace Engine