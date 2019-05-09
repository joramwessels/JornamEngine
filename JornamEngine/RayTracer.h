#pragma once

namespace JornamEngine {

class RayTracer : public OptixRenderer
{
public:
	RayTracer(Surface* screen, SCREENHALF renderhalf) :
		OptixRenderer(screen, renderhalf) {};
	RayTracer(Surface* screen) :
		OptixRenderer(screen) {};
	~RayTracer() { rtpContextDestroy(m_context); };
	void init(uint SSAA); // Called once at the start of the application
	void tick();					    // Called at the start of every frame
	void render(Camera* camera);	    // Called at the end of every frame
	inline RTPcontext getContext() const { return m_context; }

protected:
	inline void createPrimaryRays(Camera* camera);
	inline void traceRays();
	inline void shadeHits();

	RTPbufferdesc m_rays;
	RTPbufferdesc m_hits;
	OptixRay* m_rayVector;
	OptixHit* m_hitsVector;
};

} // namespace Engine