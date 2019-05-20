#pragma once

namespace JornamEngine {

class RayTracer : public OptixRenderer
{
public:
	RayTracer(Surface* screen, SCREENHALF renderhalf) :
		OptixRenderer(screen, renderhalf) {}
	RayTracer(Surface* screen) :
		OptixRenderer(screen) {}
	~RayTracer() { m_rays->free(); m_hits->free(); }
	void init(Scene* scene);					// Called once at the start of the application
	void tick();								// Called at the start of every frame
	void render(Camera* camera);				// Called at the end of every frame

protected:
	inline void createPrimaryRays(Camera* camera);
	inline void traceRays();
	inline void shadeHits();

	Buffer<OptixRay>* m_rays;
	Buffer<OptixHit>* m_hits;
	optix::prime::Query m_query;
};

} // namespace Engine