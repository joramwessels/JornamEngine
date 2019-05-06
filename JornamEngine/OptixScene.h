#pragma once

namespace JornamEngine {

struct Transform
{
	float matrix[16]; // row-dominant

	// Turns position, orientation, and scale into a row-dominant affine tranform matrix
	// TODO only translations are implemented so far
	Transform(vec3 pos, vec3 ori, vec3 scale)
	{
		matrix[0] = 1.0; matrix[5] = 1.0; matrix[10] = 1.0; // identity
		matrix[3] = pos.x; matrix[7] = pos.y; matrix[11] = pos.z; // translation
		matrix[12] = 0.0; matrix[13] = 0.0; matrix[14] = 0.0; matrix[15] = 1.0; // last row
	}
};

// Represents a 3D environment including triangles, lights, and a skybox (8 + 3*(4/8) bytes; 3 pointers)
class OptixScene : Scene
{
public:
	OptixScene(RTPcontext a_context, const char* filename, Camera* camera = 0, bool empty = false)
		: m_context(a_context), m_skybox(Skybox())
	{
		rtpModelCreate(m_context, &m_model);
		loadScene(filename, camera);
	};
	~OptixScene() { delete m_lights; rtpModelDestroy(m_model); }

	inline void addLight(Light light) { m_lights[m_numLights++] = light; }
	inline void readObject(const char* filename, vec3 pos, vec3 ori, vec3 scale, uint material);
	void readMesh(RTPmodel model, const char* filename, vec3 pos, vec3 ori, vec3 scale);
	void addObject(RTPmodel model, std::vector<float> vertices, std::vector<int> indices, Transform transform);
	RTgeometryinstance readMaterial(const char* filename, uint material, RTgeometrytriangles mesh);

	void loadScene(const char* filename, Camera* camera = 0);

	inline Light* getLights() const { return m_lights; }
	inline RTPmodel getModel() const { return m_model; }
	inline uint getLightCount() const { return m_numLights; }
	inline uint getObjectCount() const { return m_numObjects; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

private:
	RTPcontext m_context;
	RTPmodel m_model;
	std::vector<RTPmodel> m_objects;
	std::vector<float[16]> m_transforms;
	Light* m_lights;
	Skybox m_skybox;
	uint m_numLights, m_numObjects;

	void resetDimensions(uint lightSpace, uint triangleSpace);
	void parseDimensions(const char* line);
	void parseTriangle(const char* line);
	void parsePlane(const char* line);
	void parseObject(const char* line);
	void parseLight(const char* line);
	void parseSkybox(const char* line);
	void parseCamera(const char* line, Camera* camera);
	vec3 parseVec3(const char* line, uint col);
	Color parseColor(const char* line, uint col);
	uint skipWhiteSpace(const char* line, uint i = 0);
	uint skipExpression(const char* line, uint i = 0);
};

} // namespace Engine