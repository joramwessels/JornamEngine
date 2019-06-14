#pragma once

namespace JornamEngine {

// A position and a color (16 bytes)
struct Light
{
	vec3 pos;
	Color color;

	Light(vec3 position) : pos(position), color(COLOR::WHITE) {}
	Light(vec3 position, Color color) : pos(position), color(color) {}
};
__device__ struct CudaLight { float3 pos; unsigned int color; };

// An image surrounding the scene (4/8 bytes; pointer)
struct Skybox
{
	Surface* image;
	Skybox() : image(0) {}
	Skybox(const char* filename) : image(new Surface(filename)) {}
	~Skybox() { }//delete image;}

	Color intersect(vec3 direction) const
	{
		if (!image) return COLOR::BLACK;
		float u = 0.5f + (atan2f(-direction.z, -direction.x) * INV2PI);
		float v = 0.5f - (asinf(-direction.y) * INVPI);
		uint x = (uint)((image->GetWidth() - 1) * u);
		uint y = (uint)((image->GetHeight() - 1) * v);
		return image->GetPixel(x, y);
	}
};

// Represents a 3D environment including triangles, lights, and a skybox (8 + 3*(4/8) bytes; 3 pointers)
class Scene
{
public:
	Scene(optix::prime::Context a_context, const char* filename, Camera* camera = 0)
		: m_context(a_context),
		m_meshMap(MeshMap(m_context)),
		m_model(m_context->createModel())
		{ loadScene(filename, camera); };
	~Scene() {}

	void addLight(Light light) { m_lights.push_back(light); }
	void readObject(const char* filename, Transform transform);
	void addObject(std::vector<float> vertices, std::vector<uint> indices, std::vector<vec3> normals, Transform transform, Color color);
	void loadScene(const char* filename, Camera* camera = 0);

	inline const optix::prime::Context	getContext() const { return m_context; }
	inline const optix::prime::Model	getSceneModel() const { return m_model; }
	inline Object3D						getModel(int index) const { return m_objects[index]; }
	inline const Light*					getLights() const { return m_lights.data(); }
	inline uint							getLightCount() const { return (uint)m_lights.size(); }
	inline uint							getObjectCount() const { return (uint)m_rtpModels.size(); }
	inline Color						getAmbientLight() const { return m_ambientLight; }
	inline Mesh*						getMeshes() { return m_meshMap.getHostMeshes(); }

	inline void setSkybox(Skybox skybox) { m_skybox = skybox; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

private:
	RTPbuffertype m_buffertype = RTP_BUFFER_TYPE_HOST;
	optix::prime::Context m_context;
	optix::prime::Model m_model;
	std::vector<Object3D> m_objects;
	std::vector<RTPmodel> m_rtpModels;
	std::vector<TransformMatrix> m_transforms;
	std::vector<Light> m_lights;
	CudaLight* c_lights;
	Skybox m_skybox;

	MeshMap m_meshMap;

	Color m_ambientLight = 0xAAAAAA;

	//void resetDimensions(uint lightSpace, uint triangleSpace);
};

class SceneParser
{
public:
	SceneParser(Scene* scene) : m_scene(scene) {}
	void parseScene(const char* filename, Camera* camera = 0);
	uint2 parseDimensions(const char* line);
	void parseTriangle(const char* line);
	void parsePlane(const char* line);
	void parseObject(const char* line);
	void parseLight(const char* line);
	void parseSkybox(const char* line);
	void parseCamera(const char* line, Camera* camera);
	vec3 parseVec3(const char* line, uint col);
	Color parseColor(const char* line, uint col);
	float parseFloat(const char* line, uint col);
	uint skipWhiteSpace(const char* line, uint i = 0);
	uint skipExpression(const char* line, uint i = 0);
private:
	Scene* m_scene;
};

} // namespace Engine