#pragma once

namespace JornamEngine {

// Reflection constants of the Phong shading model
struct PhongMaterial
{
	float spec, diff, ambi, shin;
	PhongMaterial() : spec(1.0f), diff(1.0f), ambi(1.0f), shin(1.0f) {}
	PhongMaterial(float spec, float diff, float ambi, float shin)
		: spec(spec), diff(diff), ambi(ambi), shin(shin) {}
};
static PhongMaterial MAT_DIFFUSE(0.3f, 0.3f, 1.0f, 10.0f);

class Object3D
{
public:
	Object3D(optix::prime::Model model, uint meshIdx, uint textureIdx, Transform transform, PhongMaterial material)
		: m_primeHandle(model), m_meshIdx(meshIdx), m_textureIdx(textureIdx), m_transform(transform), m_material(material) {}
	~Object3D() {}

	void setTriangles(std::vector<uint> indices, std::vector<float> vertices, RTPbuffertype type);

	inline RTPmodel			getRTPmodel() const { return m_primeHandle->getRTPmodel(); }
	inline TransformMatrix	getTransform() const { return m_transform.matrix; }
	inline TransformMatrix	getInvTrans() const { return m_transform.inverse; }
	inline uint				getMeshIdx() const { return m_meshIdx; }
	inline uint				getTextureIdx() const { return m_textureIdx; }
	inline PhongMaterial	getMaterial() const { return m_material; }

private:
	// Model
	optix::prime::Model m_primeHandle;
	uint m_meshIdx;
	uint m_textureIdx;
	Transform m_transform;
	PhongMaterial m_material;
};

} // namespace Engine