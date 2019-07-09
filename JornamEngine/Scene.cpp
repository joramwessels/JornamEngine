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
void Scene::addObject(uint meshIdx, uint textureIdx, Transform transform)
{
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
	Returns the mesh index of the given file
	Adds the mesh to the mesh queue if not already loaded

	@param filename			The path to the .obj file
	@returns				The index of the mesh
	@throws JornamException	When the file couldn't be read
*/
uint Scene::addMesh(const char* filename)
{
	uint meshIdx;
	if (!(meshIdx = m_meshMap.get(filename)))
	{
		// Reading .obj using tinyobjloader
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err, warn;
		tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);
		
		if (!err.empty()) logger.logDebug("Scene",
			(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
			JornamException::ERR);
		meshIdx = m_meshMap.add(filename, attrib, shapes[0].mesh, isOnDevice());
		logger.logDebug("Scene", ("Loaded mesh " + std::to_string(meshIdx) +
			": \"" + std::string(filename) + "\"").c_str(), JornamException::INFO);
	}
	return meshIdx;
}

/*
	Returns the texture index of the given file
	Adds the texture to the texture queue if not already loaded

	@param filename	The path to the image file
	@param color	A color for solid color textures, defaults to NOCOLOR
	@returns		The index of the texture
*/
uint Scene::addTexture(const char* filename, Color color)
{
	uint textureIdx;
	if (color.hex != COLOR::NOCOLOR)
	{
		// Adding solid color texture
		char colorHex[8];
		sprintf(colorHex, "%x", color.hex);
		if (!(textureIdx = m_textureMap.get(colorHex)))
		{
			textureIdx = m_textureMap.add(colorHex, color);
			logger.logDebug("Scene", ("Loaded texture " + std::to_string(textureIdx) +
				": \"" + colorHex + "\"").c_str(), JornamException::INFO);
		}
	}
	else
	{
		// Adding image texture
		if (!(textureIdx = m_textureMap.get(filename)))
		{
			textureIdx = m_textureMap.add(filename, isOnDevice());
			logger.logDebug("Scene", ("Loaded texture " + std::to_string(textureIdx) +
				": \"" + filename + "\"").c_str(), JornamException::INFO);
		}
	}
	return textureIdx;
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
	Mesh mesh = m_meshes[m_objects[o].getMeshIdx()];
	const Index* indices = mesh.getIndices();
	const float3* normals = mesh.getNormals();
	vec3 n0 = normals[indices[3 * t + 0].normalIdx];
	vec3 n1 = normals[indices[3 * t + 1].normalIdx];
	vec3 n2 = normals[indices[3 * t + 2].normalIdx];
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

	// Unpacking texture coordinates
	Mesh mesh = m_meshes[m_objects[o].getMeshIdx()];
	int idx1 = mesh.getIndices()[3 * t + 0].textureIdx;
	int idx2 = mesh.getIndices()[3 * t + 1].textureIdx;
	int idx3 = mesh.getIndices()[3 * t + 2].textureIdx;
	float2 texcoord1 = mesh.getTexcoords()[idx1];
	float2 texcoord2 = mesh.getTexcoords()[idx2];
	float2 texcoord3 = mesh.getTexcoords()[idx3];

	// Texture wrapping
	if (texcoord1.x > 1.0f || texcoord1.x < 0.0f) texcoord1.x -= floor(texcoord1.x);
	if (texcoord1.y > 1.0f || texcoord1.y < 0.0f) texcoord1.y -= floor(texcoord1.y);
	if (texcoord2.x > 1.0f || texcoord2.x < 0.0f) texcoord2.x -= floor(texcoord2.x);
	if (texcoord2.y > 1.0f || texcoord2.y < 0.0f) texcoord2.y -= floor(texcoord2.y);
	if (texcoord3.x > 1.0f || texcoord3.x < 0.0f) texcoord3.x -= floor(texcoord3.x);
	if (texcoord3.y > 1.0f || texcoord3.y < 0.0f) texcoord3.y -= floor(texcoord3.y);

	// Interpolating coordinates
	int x = (texcoord1.x * u + texcoord2.x * v + texcoord3.x * (1 - u - v)) * texture.getWidth();
	int y = (texcoord1.y * u + texcoord2.y * v + texcoord3.y * (1 - u - v)) * texture.getHeight();
	return texture.getBuffer()[x + y * texture.getWidth()];
}

} // namespace Engine