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
		m_objects.size(), RTP_BUFFER_TYPE_CUDA_LINEAR, m_objects.data(),
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_CUDA_LINEAR, (void*)m_transforms.data()
	);
	m_model->update(0);
}

// Adds a new object to the GeometryGroup
void Scene::readObject(const char* filename, TransformMatrix transform, uint material)
{
	optix::prime::Model objectModel = m_context->createModel();
	readMesh(objectModel, filename, transform);
	// TODO materials and textures
}

// Reads a mesh from a .obj file and adds it to the object queue
void Scene::readMesh(optix::prime::Model model, const char* filename, TransformMatrix transform)
{
	// Reading .obj using tinyobjloader
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(shapes, materials, err, filename);
	if (!err.empty()) logDebug("Scene",
		(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
		JornamException::ERR);
	addObject(model, shapes[0].mesh.positions, shapes[0].mesh.indices, transform);
}

// Adds the object to the object queue as a triangle model and as a transformation matrix to the transform queue
void Scene::addObject(optix::prime::Model model, std::vector<float> vertices, std::vector<uint> indices, TransformMatrix transform)
{
	model->setTriangles(
		(RTPsize)(indices.size() / 3), RTP_BUFFER_TYPE_CUDA_LINEAR, indices.data(),
		(RTPsize)vertices.size(), RTP_BUFFER_TYPE_CUDA_LINEAR, vertices.data()
	);
	model->update(RTP_QUERY_HINT_ASYNC);
	model->finish();

	m_objects.push_back(model->getRTPmodel());
	m_transforms.push_back(transform);
}

} // namespace Engine