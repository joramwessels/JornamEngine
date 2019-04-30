#pragma once

namespace JornamEngine {

// Represents a 3D environment including triangles, lights, and a skybox (8 + 3*(4/8) bytes; 3 pointers)
class OptixScene : Scene
{
public:
	OptixScene(const char* filename, Camera* camera = 0, bool empty = false) : m_skybox(Skybox()) { loadScene(filename, camera); };
	OptixScene(uint lightSpace, uint triangleSpace) : Scene(lightSpace, triangleSpace, Skybox()) {};
	OptixScene(uint lightSpace, uint triangleSpace, char* skybox) : Scene(lightSpace, triangleSpace, Skybox(skybox)) {};
	OptixScene(uint lightSpace, uint triangleSpace, Skybox skybox) :
		m_lights((Light*)malloc(lightSpace * sizeof(Light))),
		m_numLights(0), m_numObjects(0), m_skybox(skybox) {};
	~OptixScene() { delete m_lights; delete m_triangles; }

	inline void addLight(Light light) { m_lights[m_numLights++] = light; }
	inline void addObject(std::string filename, vec3 pos, vec3 ori, uint material);
	void loadScene(const char* filename, Camera* camera = 0);

	inline Light* getLights() const { return m_lights; }
	inline RTgeometrygroup getObjects() const { return m_objects; }
	inline uint getLightCount() const { return m_numLights; }
	inline uint getObjectCount() const { return m_numObjects; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

private:
	Light* m_lights;
	RTgeometrygroup m_objects;
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