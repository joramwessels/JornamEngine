#pragma once

namespace JornamEngine {

// A position and a color (16 bytes)
struct Light
{
	vec3 pos;
	Color color;

	Light(vec3 position) : pos(position), color(COLOR::WHITE) {};
	Light(vec3 position, Color color) : pos(position), color(color) {};
};

// An image surrounding the scene (4/8 bytes; pointer)
struct Skybox
{
	Surface* image;
	Skybox() : image(0) {};
	Skybox(const char* filename) : image(new Surface(filename)) {};
	~Skybox() { delete image; }

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
	Scene(RTPcontext a_context, const char* filename, Camera* camera = 0)
		: m_context(a_context), m_skybox(Skybox())
	{
		rtpModelCreate(m_context, &m_model);
		loadScene(filename, camera);
	};
	~Scene() { delete m_lights; rtpModelDestroy(m_model); }

	void addLight(Light light) { m_lights[m_numLights++] = light; }
	void readObject(const char* filename, TransformMatrix transform, uint material);
	void readMesh(RTPmodel model, const char* filename, TransformMatrix transform);
	void addObject(RTPmodel model, std::vector<float> vertices, std::vector<uint> indices, TransformMatrix transform);
	RTgeometryinstance readMaterial(const char* filename, uint material, RTgeometrytriangles mesh);

	void loadScene(const char* filename, Camera* camera = 0);

	inline RTPcontext getContext() const { return m_context; }
	inline RTPmodel getModel() const { return m_model; }
	inline Light* getLights() const { return m_lights; }
	inline uint getLightCount() const { return m_numLights; }
	inline uint getObjectCount() const { return m_numObjects; }
	inline void setLightCount(uint numLights) { m_numLights = numLights; }
	inline void setObjectCount(uint numObjects) { m_numObjects = numObjects; }
	inline void setSkybox(Skybox skybox) { m_skybox = skybox; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

private:
	RTPcontext m_context;
	RTPmodel m_model;
	std::vector<RTPmodel> m_objects;
	std::vector<TransformMatrix> m_transforms;
	Light* m_lights;
	Skybox m_skybox;
	uint m_numLights, m_numObjects;

	void resetDimensions(uint lightSpace, uint triangleSpace);
};

class SceneParser
{
public:
	SceneParser(Scene* scene) : m_scene(scene) {}
	void parseScene(const char* filename, Camera* camera = 0);
	void parseDimensions(const char* line);
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