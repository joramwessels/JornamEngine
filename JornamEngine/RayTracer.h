#pragma once

namespace JornamEngine {

class RayTracer : public OptixRenderer
{
public:
	RayTracer(Surface* screen, SCREENHALF renderhalf) :
		OptixRenderer(screen, renderhalf) {};
	RayTracer(Surface* screen, USE_GPU useGPU) :
		OptixRenderer(screen) {};
	~RayTracer() { rtContextDestroy(m_context); };
	void init(Scene* scene, uint SSAA); // Called once at the start of the application
	void tick();					    // Called at the start of every frame
	void render(Camera* camera);	    // Called at the end of every frame

protected:
	RTcontext m_context;
	RTbuffer m_buffer;
	
	Scene* m_scene;			// The collection of triangles and lights to be rendered
};

} // namespace Engine