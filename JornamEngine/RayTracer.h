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
	void render(vec3 location, vec3 direction);
};

} // namespace Engine