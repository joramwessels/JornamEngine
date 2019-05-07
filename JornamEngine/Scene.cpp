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
void Scene::loadScene(const char* filename, Camera *camera = 0)
{
	if (m_model != 0) rtpModelDestroy(m_model);
	SceneParser parser = SceneParser(this);
	parser.parseScene(filename, camera);

	RTPbufferdesc instances, transforms;
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_INSTANCE_MODEL, RTP_BUFFER_TYPE_HOST, m_objects.data, &instances);
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, m_transforms.data, &transforms);
	rtpBufferDescSetRange(instances, 0, m_objects.size());
	rtpBufferDescSetRange(transforms, 0, m_transforms.size());
	rtpModelSetInstances(m_model, instances, transforms);
	rtpModelUpdate(m_model, RTP_MODEL_HINT_NONE);
	rtpModelFinish(m_model);
}

// Adds a new object to the GeometryGroup
void Scene::readObject(const char* filename, TransformMatrix transform, uint material) // TODO reading material
{
	RTPmodel objectModel;
	rtpModelCreate(m_context, &objectModel);
	readMesh(objectModel, filename, transform);
	//RTgeometryinstance object = readMaterial(filename, material, mesh);
}

// Reads a mesh from a .obj file and adds it to the object queue
void Scene::readMesh(RTPmodel model, const char* filename, TransformMatrix transform)
{
	// Reading .obj using tinyobjloader
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(shapes, materials, err, filename);
	if (!err.empty()) logDebug("Scene",
		(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
		JornamException::ERR);
	addObject(model, shapes[0].mesh.positions.data, shapes[0].mesh.indices.data, transform);
}

// Adds the object to the object queue as a triangle model and as a transformation matrix to the transform queue
void Scene::addObject(RTPmodel model, std::vector<float> vertices, std::vector<int> indices, TransformMatrix transform)
{
	RTPbufferdesc indBuffer, verBuffer;
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_CUDA_LINEAR, indices.data, &indBuffer); // TODO is shapes[0] the whole mesh?
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_CUDA_LINEAR, indices.data, &verBuffer);
	rtpBufferDescSetRange(indBuffer, 0, indices.size());
	rtpBufferDescSetRange(verBuffer, 0, vertices.size());

	rtpModelSetTriangles(model, indBuffer, verBuffer);
	rtpModelUpdate(model, RTP_MODEL_HINT_NONE);
	rtpModelFinish(model);

	m_objects.push_back(model);
	m_transforms.push_back(transform);
}

} // namespace Engine