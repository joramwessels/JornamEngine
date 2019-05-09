#pragma once

namespace JornamEngine {

enum RAYTYPE {PRIMARY, SHADOW}; // { PRIMARY, SHADOW }

struct OptixRay
{
	vec3 origin, direction;
	OptixRay() : origin(vec3(0.0f)), direction(vec3(0.0f)) {};
	OptixRay(vec3 ori, vec3 dir) : origin(ori), direction(dir) {};
};

struct OptixHit { float rayDistance; int triangleIdx; int instanceIdx; float u; float v; };

class OptixRenderer : public Renderer
{
public:
	OptixRenderer(Surface* screen, SCREENHALF renderhalf) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA, renderhalf) {};
	OptixRenderer(Surface* screen) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA) {};
	~OptixRenderer() { rtpContextDestroy(m_context); };
	virtual void init(uint SSAA) {}; // Called once at the start of the application
	virtual void tick() {};						   // Called at the start of every frame
	virtual void render(Camera* camera) {};		   // Called at the end of every frame
	inline RTPcontext getContext() const { return m_context; }

protected:
	RTPcontext m_context;
	RTPbufferdesc m_buffer;

	//void initializeMaterials();
};

} // namespace Engine