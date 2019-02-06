#pragma once

namespace JornamEngine {

// Three vertices and a normal vector (48 bytes)
struct Triangle
{
	vec3 v0, v1, v2; // vertices
	//vec3 e0, e1, e2; // edges?
	vec3 N;  // normal vector
	//float D; // distance to origin

	// Only provides correct normal if provided in clockwise order
	Triangle(vec3 v0, vec3 v1, vec3 v2) : v0(v0), v1(v1), v2(v2), N((v2 - v0).cross(v1 - v0)) {};
	Triangle(vec3 v0, vec3 v1, vec3 v2, vec3 N) : v0(v0), v1(v1), v2(v2), N(N) {};
};

// A position and a color (16 bytes)
struct Light
{
	vec3 pos;
	Color col;

	Light(vec3 position) : pos(position), col(0xFFFFFFFF) {};
	Light(vec3 position, Color color) : pos(position), col(color) {};
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
		if (!image) return 0x00000000;
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
	Scene(uint lightSpace, uint triangleSpace) : Scene(lightSpace, triangleSpace, Skybox()) {};
	Scene(uint lightSpace, uint triangleSpace, char* skybox) : Scene(lightSpace, triangleSpace, Skybox(skybox)) {};
	Scene(uint lightSpace, uint triangleSpace, Skybox skybox) :
		m_lights((Light*)malloc(lightSpace * sizeof(Light))),
		m_triangles((Triangle*)malloc(triangleSpace * sizeof(Triangle))),
		m_numLights(0), m_numTriangles(0), m_skybox(skybox) {};
	~Scene() { delete m_lights; delete m_triangles; }

	inline void addLight(Light light) { m_lights[m_numLights++] = light; }
	inline void addTriangle(Triangle triangle) { m_triangles[m_numTriangles++] = triangle; }
	void loadScene(char* filename);

	inline Light* getLights() const { return m_lights; }
	inline Triangle* getTriangles() const { return m_triangles; }
	inline uint getLightCount() const { return m_numLights; }
	inline uint getTriangleCount() const { return m_numTriangles; }
	inline Color intersectSkybox(vec3 direction) const { return m_skybox.intersect(direction); }

private:
	Light* m_lights;
	Triangle* m_triangles;
	Skybox m_skybox;
	uint m_numLights, m_numTriangles;

	void parseTriangle(const char* line);
	void parsePlane(const char* line);
	void parseLight(const char* line);
	void parseSkybox(const char* line);
	vec3 parseVec3(const char* line);
	Color parseColor(const char* line);
	uint skipWhiteSpace(const char* line, uint i = 0);
	uint skipExpression(const char* line, uint i = 0);
};

} // namespace Engine