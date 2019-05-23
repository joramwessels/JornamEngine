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
	//SceneParser parser = SceneParser(this);
	//parser.parseScene(filename, camera);
	initDebugModel();

	//m_model->setInstances(
	//	m_objects.size(), RTP_BUFFER_TYPE_HOST, &m_objects[0],
	//	RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, &m_transforms[0]
	//);
	//m_model->update(0);
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
	optix::prime::Model model = m_context->createModel();
	addObject(shapes[0].mesh.positions, shapes[0].mesh.indices, transform, model);
}

// Adds the object to the object queue as a triangle model and as a transformation matrix to the transform queue
void Scene::addObject(std::vector<float> vertices, std::vector<uint> indices, TransformMatrix transform, optix::prime::Model model)
{
	model->setTriangles(
		indices.size() / 3, RTP_BUFFER_TYPE_HOST, indices.data(),
		vertices.size() / 3, RTP_BUFFER_TYPE_HOST, vertices.data()
	);
	model->update(0);

	m_objects.push_back(model->getRTPmodel());
	m_transforms.push_back(transform);
}

void Scene::initDebugModel()
{
	RTPbuffertype buffertype = RTP_BUFFER_TYPE_HOST;

	vec3 v0(-2, 2, -2), v1(2, 2, -2), v2(2, -2, -2);
	std::vector<float> vertices({ v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z });
	std::vector<uint> indices({ 0, 1, 2 });
	optix::prime::Model model = m_context->createModel();
	addObject(vertices, indices, TransformMatrix(), model);

	m_model->setInstances(m_objects.size(), buffertype, &m_objects[0], RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, buffertype, &m_transforms[0]);
	m_model->update(0);
}

} // namespace Engine