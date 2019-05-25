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
		m_objects.size(), RTP_BUFFER_TYPE_HOST, &m_objects[0],
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, &m_transforms[0]
	);
	m_model->update(0);
}

// Reads a mesh from a .obj file and adds it to the object queue
void Scene::readObject(const char* filename, TransformMatrix transform)
{
	// Reading .obj using tinyobjloader
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(shapes, materials, err, filename);
	if (!err.empty()) logDebug("Scene",
		(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
		JornamException::ERR);
	addObject(shapes[0].mesh.positions, shapes[0].mesh.indices, transform);
}

// Adds the object to the object queue as a triangle model and as a transformation matrix to the transform queue
void Scene::addObject(std::vector<float> vertices, std::vector<uint> indices, TransformMatrix transform)
{
	m_models.push_back(m_context->createModel());
	m_models[m_models.size() - 1]->setTriangles(
		indices.size() / 3, RTP_BUFFER_TYPE_HOST, indices.data(),
		vertices.size() / 3, RTP_BUFFER_TYPE_HOST, vertices.data()
	);
	m_models[m_models.size() - 1]->update(0);

	m_objects.push_back(m_models[m_models.size() - 1]->getRTPmodel());
	m_transforms.push_back(transform);
}

} // namespace Engine