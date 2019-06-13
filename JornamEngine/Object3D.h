#pragma once

namespace JornamEngine {

// Reflection constants of the Phong shading model
struct PhongMaterial { float spec, diff, ambi, shin; };

class Object3D
{
public:
	Object3D(optix::prime::Model model, std::vector<uint> indices, std::vector<vec3> normals, Transform transform, Color color)
		: m_primeHandle(model), m_indices(indices), m_normals(normals), m_transform(transform), m_color(color) {}
	~Object3D() {}

	void setTriangles(std::vector<uint> indices, std::vector<float> vertices, RTPbuffertype type);

	vec3 interpolateNormal(uint trIdx, float u, float v);

	RTPmodel getRTPmodel() const { return m_primeHandle->getRTPmodel(); }
	Color getColor() const { return m_color; }
	TransformMatrix getInvTrans() const { return m_transform.inverse; }

private:
	// Model
	optix::prime::Model m_primeHandle;
	std::vector<uint> m_indices;
	std::vector<vec3> m_normals;
	Color m_color; // TODO change to textures
	PhongMaterial m_material;

	// Transform
	Transform m_transform;

};

} // namespace Engine