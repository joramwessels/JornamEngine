#pragma once

namespace JornamEngine {

// A ray for streaming ray tracing (32 bytes)
union Ray {
	struct
	{
		vec3 origin;
		uint pixelIdx;
		vec3 direction;
		uint flags;
	};
	float cell[8];

	Ray(vec3 origin, uint pixelIdx, vec3 direction) :
		origin(origin), pixelIdx(pixelIdx), direction(direction) {};
};

// A collision between a ray and a triangle (32 bytes)
union Collision
{
	struct
	{
		vec3 position;
		float distance;
		vec3 N;
		Color colorAt;
	};
	float cell[8];

	Collision(vec3 position, float distance, vec3 normal, Color colorAt) :
		position(position), distance(distance), N(normal), colorAt(colorAt) {};
};

class RayTracer : public Renderer
{
public:
	RayTracer(Surface* screen, USE_GPU useGPU, SCREENHALF renderhalf) :
		Renderer(screen, useGPU, renderhalf) {};
	RayTracer(Surface* screen, USE_GPU useGPU) :
		Renderer(screen, useGPU) {};
	~RayTracer() {};
	void init(Scene* scene, uint SSAA);
	void render(Camera* camera);

protected:
	Color* m_buffer;		// The intermediate frame buffer to add ray values to
	Ray* m_rayQueue;		// The queue of rays to be extended
	Collision* m_colQueue;	// The queue of collisions to be shaded

	void addRayToQueue(Ray ray);
	void addCollisionToQueue(Collision collision);

	void generateRays(vec3 location, Corners screenCorners);
	void extendRays();
};

} // namespace Engine