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
		Renderer(screen, JornamEngine::USE_GPU::CUDA, renderhalf) { initContext(); }
	OptixRenderer(Surface* screen) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA) { initContext(); }
	~OptixRenderer() { m_context.~Handle(); }
	virtual void init(Scene* scene) {}				// Called once at the start of the application
	virtual void tick() {}							// Called at the start of every frame
	virtual void render(Camera* camera) {}			// Called at the end of every frame
	inline optix::prime::Context getContext() const { return m_context; }

protected:
	optix::prime::Context m_context;

	// Initializes the context object
	void OptixRenderer::initContext(RTPcontexttype type = RTP_CONTEXT_TYPE_CPU)
	{
		m_context = optix::prime::Context::create(type);
		if (type == RTP_CONTEXT_TYPE_CPU)
			logDebug("RayTracer", "Using CPU context\n", JornamException::INFO);
		else
		{
			unsigned int device = 0;
			m_context->setCudaDeviceNumbers(1, &device);
			logDebug("RayTracer", "Using CUDA context\n", JornamException::INFO);
		}
	}
};

} // namespace Engine