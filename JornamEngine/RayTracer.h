#pragma once

namespace JornamEngine {

class RayTracer : public OptixRenderer
{
public:
	RayTracer(Surface* screen, SCREENHALF renderhalf, USE_GPU onDevice = USE_GPU::CUDA) :
		OptixRenderer(screen, renderhalf, onDevice) {}
	RayTracer(Surface* screen, USE_GPU onDevice = USE_GPU::CUDA) :
		OptixRenderer(screen, onDevice) {}
	~RayTracer() { m_rays->free(); m_hits->free(); }
	void init(Scene* scene);					// Called once at the start of the application
	void tick();								// Called at the start of every frame
	void render(Camera* camera);				// Called at the end of every frame

protected:
	inline void createPrimaryRays(Camera* camera);
	inline void traceRays();
	inline void shadeHits(Camera* camera);

	Buffer<OptixRay>* m_rays;
	Buffer<OptixHit>* m_hits;
	optix::prime::Query m_query;

	JECUDA::Color* c_buffer;
};

} // namespace Engine