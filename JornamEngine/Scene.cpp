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

// Loads a scene from a .scene file
void Scene::loadScene(const char* filename, Camera *camera)
{
	SceneParser parser = SceneParser(this);
	parser.parseScene(filename, camera);

	m_model->setInstances(
		m_rtpModels.size(), m_buffertype, &m_rtpModels[0],
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, m_buffertype, &m_transforms[0]
	);
	m_model->update(0);
}

// Reads a mesh from a .obj file and adds it to the object queue
void Scene::readObject(const char* filename, Transform transform)
{
	// Reading .obj using tinyobjloader
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(shapes, materials, err, filename);
	if (!err.empty()) logDebug("Scene",
		(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
		JornamException::ERR);

	// Converting normals
	std::vector<vec3> N;
	std::vector<float> normals = shapes[0].mesh.normals;
	for (int i = 0; i < normals.size(); i += 3) N.push_back(vec3(normals[i], normals[i + 1], normals[i + 2]));

	addObject(shapes[0].mesh.positions, shapes[0].mesh.indices, N, transform, 0xBBBBBB);
}

/*
	Adds the object, RTPmodel, and transform to their queues

	@param vertices		A vector with 3 consecutive floats per vertex
	@param indices		A vector with 3 consecutive indices per triangle
	@param N			A vector of vertex normals
	@param transform	The initial transform
	@param color		The color of the model
*/
void Scene::addObject(std::vector<float> V, std::vector<uint> I, std::vector<vec3> N, Transform T, Color C)
{
	Object3D object(m_context->createModel(), I, N, T, C);
	object.setTriangles(I, V, m_buffertype);
	m_objects.push_back(object);
	//m_objects[m_objects.size() - 1].setTriangles(indices, vertices, m_buffertype);

	m_rtpModels.push_back(object.getRTPmodel());
	m_transforms.push_back(T.matrix);
}

} // namespace Engine