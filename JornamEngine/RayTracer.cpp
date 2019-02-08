#include "headers.h"

namespace JornamEngine {

void RayTracer::init(Scene* a_scene, uint SSAA)
{

}

void RayTracer::render(Camera* camera)
{
	generateRays(camera->getForward(), camera->getScreenCorners());
	extendRays();
	//generateShadowRays();
	//extendShadowRays();
	//screen->Plot();
}

// Generates a queue of primary rays
void RayTracer::generateRays(vec3 location, Corners a_corners)
{
	for (int x=0; x < m_scrwidth; x++) for (int y=0; y < m_scrheight; y++)
	{
		// Find location on virtual screen
		float relX = x / (float)m_scrwidth;
		float relY = y / (float)m_scrheight;
		vec3 pixPos = a_corners.TL + (a_corners.TR - a_corners.TL) * relX + (a_corners.BL - a_corners.TL) * relY;

		// Add ray to queue
		vec3 direction = (pixPos - location).normalized();
		uint pixelIdx = x + y * m_scrwidth;
		addRayToQueue(Ray(location, pixelIdx, direction));
	}
}

void RayTracer::extendRays()
{

}

// Adds the ray to the ray queue and increments the ray count in the header
void RayTracer::addRayToQueue(Ray ray)
{
	// increments rayCount, checks for overflow, adds ray to incremented index to skip header
	uint *queueSize = (uint*)m_rayQueue, *rayCount = (uint*)m_rayQueue + 1;
	if (++*rayCount > *queueSize) printf("Ray queue overflow\n");
	else m_rayQueue[*rayCount] = ray;
}

// Adds the collision to the collision queue and increments the collision counter in the header
void RayTracer::addCollisionToQueue(Collision col)
{
	// increments colCount, checks for overflow, adds col to incremented index to skip header
	uint *queueSize = (uint*)m_colQueue, *colCount = (uint*)m_colQueue + 1;
	if (++*colCount > *queueSize) printf("Collision queue overflow\n");
	else m_colQueue[*colCount] = col;
}

} // namespace Engine