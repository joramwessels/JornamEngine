#pragma once

namespace JornamEngine {

class RayTracer : public Renderer
{
public:
	RayTracer(Surface* screen, USE_GPU useGPU, SCREENHALF renderhalf) :
		Renderer(screen, useGPU, renderhalf) {};
	RayTracer(Surface* screen, USE_GPU useGPU) :
		Renderer(screen, useGPU) {};
	~RayTracer() {};
	void init(Scene* scene, uint SSAA);
	void tick();
	void render(Camera* camera);

protected:
	Color* m_buffer;		// The intermediate frame buffer to add ray values to
	Ray* m_rayQueue;		// The queue of rays to be extended
	Collision* m_colQueue;	// The queue of collisions to be shaded
	Ray* m_shadowRayQueue;	// The queue of shadow rays to be extended
	Scene* m_scene;			// The collection of triangles and lights to be rendered

	// Adds the color to the existing color at the given pixel id
	void addToBuffer(Color color, uint pixIdx) { m_buffer[pixIdx] += color; }
	void addRayToQueue(Ray ray);
	void addCollisionToQueue(Collision collision);
	void addShadowRayToQueue(Ray ray);

	void generateRays(vec3 location, ScreenCorners screenCorners);
	void extendRays();
	void generateShadowRays();
	void extendShadowRays();
	void plotScreenBuffer() { m_screen->Plot(m_buffer); }
};

} // namespace Engine