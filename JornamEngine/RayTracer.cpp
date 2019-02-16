#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* scene, uint SSAA)
{
	uint queuesize = m_scrwidth * m_scrheight;
	m_buffer = (Color*)malloc(queuesize * sizeof(Color));
	m_rayQueue = (Ray*)malloc((queuesize + 1) * sizeof(Ray));
	m_colQueue = (Collision*)malloc((queuesize + 1) * sizeof(Collision));
	((uint*)m_rayQueue)[0] = queuesize;
	memset(m_rayQueue + 1, 0, sizeof(Ray) - sizeof(uint));
	((uint*)m_colQueue)[0] = queuesize;
	memset(m_colQueue + 1, 0, sizeof(Collision) - sizeof(uint));
	m_scene = scene;
	m_SSAA = SSAA;
}

// Called at the start of every frame
void RayTracer::tick()
{
	memset(m_buffer, 0, m_scrwidth * m_scrheight * sizeof(Color));
	memset((uint*)m_rayQueue + 1, 0, sizeof(Ray) - sizeof(uint));
	memset((uint*)m_colQueue + 1, 0, sizeof(Collision) - sizeof(uint));
}

void RayTracer::render(Camera* camera)
{
	generateRays(camera->getLocation(), camera->getScreenCorners());
	extendRays();
	//generateShadowRays();
	//extendShadowRays();
	plotScreenBuffer();
}

// Generates a queue of primary rays
void RayTracer::generateRays(vec3 location, ScreenCorners a_corners)
{
	for (uint x=0; x < m_scrwidth; x++) for (uint y=0; y < m_scrheight; y++)
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

// Extends the ray and checks for triangle intersections
void RayTracer::extendRays()
{
	int rayCount = ((uint*)m_rayQueue)[1];
	int triCount = m_scene->getTriangleCount();
	Triangle* triangles = m_scene->getTriangles();
	for (int i = 1; i <= rayCount; i++)
	{
		Ray ray = m_rayQueue[i];
		m_colQueue[i] = intersectTriangles(triangles, triCount, ray);
		if (m_colQueue[i].N.isNonZero()) addToBuffer(m_colQueue[i].colorAt, i); // DEBUG
		else addToBuffer(0, i);													// DEBUG
	}
}

// Adds the ray to the ray queue and increments the ray count in the header
void RayTracer::addRayToQueue(Ray ray)
{
	// increments rayCount, checks for overflow, adds ray to incremented index to skip header
	uint *queueSize = (uint*)m_rayQueue, *rayCount = (uint*)m_rayQueue + 1;
	if (++*rayCount > *queueSize)
		throw JornamEngineException("RayTracer", "Ray queue overflow.\n", JornamEngineException::ERR);
	else m_rayQueue[*rayCount] = ray;
}

// Adds the collision to the collision queue and increments the collision counter in the header
void RayTracer::addCollisionToQueue(Collision col)
{
	// increments colCount, checks for overflow, adds col to incremented index to skip header
	uint *queueSize = (uint*)m_colQueue, *colCount = (uint*)m_colQueue + 1;
	if (++*colCount > *queueSize)
		throw JornamEngineException("RayTracer", "Collision queue overflow.\n", JornamEngineException::ERR);
	else m_colQueue[*colCount] = col;
}

} // namespace Engine