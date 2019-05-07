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
	void init(Scene* scene, uint SSAA); // Called once at the start of the application
	void tick();					    // Called at the start of every frame
	void render(Camera* camera);	    // Called at the end of every frame

protected:
	void createBuffers();
	void createPrimaryRays();
	void traceRays();
	void shadeHits();
};

} // namespace Engine