#pragma once

namespace JornamEngine {

class OptixRenderer : public Renderer
{
public:
	OptixRenderer(Surface* screen, SCREENHALF renderhalf, USE_GPU onDevice = USE_GPU::CUDA) :
		Renderer(screen, USE_GPU::CUDA, renderhalf) { initContext(onDevice); }
	OptixRenderer(Surface* screen, USE_GPU onDevice = USE_GPU::CUDA) :
		Renderer(screen, USE_GPU::CUDA) { initContext(onDevice); }
	~OptixRenderer() { m_context.~Handle(); }
	virtual void init(Scene* scene) {}				// Called once at the start of the application
	virtual void render(Camera* camera) {}			// Called at the end of every frame
	inline optix::prime::Context getContext() const { return m_context; }

protected:
	optix::prime::Context m_context;
	RTPbuffertype m_buffertype;

	// Initializes the context object
	void OptixRenderer::initContext(USE_GPU onDevice)
	{
		RTPcontexttype type = (onDevice == USE_GPU::CUDA ? RTP_CONTEXT_TYPE_CUDA : RTP_CONTEXT_TYPE_CPU);
		m_context = optix::prime::Context::create(type);
		if (type == RTP_CONTEXT_TYPE_CPU)
		{
			m_buffertype = RTP_BUFFER_TYPE_HOST;
			logger.logDebug("RayTracer", "Using CPU context", JornamException::INFO);
		}
		else
		{
			unsigned int device = 0;
			m_context->setCudaDeviceNumbers(1, &device);
			m_buffertype = RTP_BUFFER_TYPE_CUDA_LINEAR;
			logger.logDebug("RayTracer", "Using CUDA context", JornamException::INFO);
		}
	}
};

} // namespace Engine