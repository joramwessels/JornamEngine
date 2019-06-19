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
__device__ struct CudaPhongMaterial { float spec, diff, ambi, shin; };
static PhongMaterial MAT_DIFFUSE(0.3f, 0.3f, 1.0f, 10.0f);

class Object3D
{
public:
	Object3D(optix::prime::Model model, uint meshIdx, Transform transform, Color color)
		: m_primeHandle(model), m_meshIdx(meshIdx), m_transform(transform), m_color(color) {}
	~Object3D() {}

	void setTriangles(std::vector<uint> indices, std::vector<float> vertices, RTPbuffertype type);

	inline RTPmodel			getRTPmodel() const { return m_primeHandle->getRTPmodel(); }
	inline Color			getColor() const { return m_color; }
	inline TransformMatrix	getTransform() const { return m_transform.matrix; }
	inline TransformMatrix	getInvTrans() const { return m_transform.inverse; }
	inline uint				getMeshIdx() const { return m_meshIdx; }

private:
	// Model
	optix::prime::Model m_primeHandle;
	uint m_meshIdx;
	Color m_color; // TODO change to textures
	PhongMaterial m_material;

	// Transform
	Transform m_transform;

};
__device__ struct CudaObject3D
{
	optix::prime::Model primeHandle;
	unsigned int meshIdx;
	struct material { float spec, diff, ambi, shin; };
	CudaTransformMatrix invTrans;
};

} // namespace Engine