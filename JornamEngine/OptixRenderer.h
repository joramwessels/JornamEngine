#pragma once

namespace JornamEngine {

enum RAYTYPE {PRIMARY, SHADOW}; // { PRIMARY, SHADOW }

class OptixRenderer : public Renderer
{
public:
	OptixRenderer(Surface* screen, SCREENHALF renderhalf) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA, renderhalf) { initContext(); }
	OptixRenderer(Surface* screen) :
		Renderer(screen, JornamEngine::USE_GPU::CUDA) { initContext(); }
	~OptixRenderer() { m_context.~Handle(); }
	virtual void init(Scene* scene) {}				// Called once at the start of the application
	virtual void render(Camera* camera) {}			// Called at the end of every frame
	inline optix::prime::Context getContext() const { return m_context; }

protected:
	optix::prime::Context m_context;
	RTPbuffertype m_buffertype;

	// Initializes the context object
	void OptixRenderer::initContext(RTPcontexttype type = RTP_CONTEXT_TYPE_CUDA)
	{
		m_context = optix::prime::Context::create(type);
		if (type == RTP_CONTEXT_TYPE_CPU)
		{
			m_buffertype = RTP_BUFFER_TYPE_HOST;
			logger.logDebug("RayTracer", "Using CPU context\n", JornamException::INFO);
		}
		else
		{
			unsigned int device = 0;
			m_context->setCudaDeviceNumbers(1, &device);
			m_buffertype = RTP_BUFFER_TYPE_CUDA_LINEAR;
			logger.logDebug("RayTracer", "Using CUDA context\n", JornamException::INFO);
		}
	}
};

} // namespace Engine