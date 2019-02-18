#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene, uint a_SSAA)
{
	uint queuesize = m_scrwidth * m_scrheight;
	m_buffer = (Color*)malloc(queuesize * sizeof(Color));
	m_rayQueue = (Ray*)malloc((queuesize + 1) * sizeof(Ray));
	m_colQueue = (Collision*)malloc((queuesize + 1) * sizeof(Collision));
	((uint*)m_rayQueue)[0] = queuesize;
	memset(m_rayQueue + 1, 0, sizeof(Ray) - sizeof(uint));
	((uint*)m_colQueue)[0] = queuesize;
	memset(m_colQueue + 1, 0, sizeof(Collision) - sizeof(uint));
	m_scene = a_scene;
	m_SSAA = a_SSAA;
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
void RayTracer::generateRays(vec3 a_location, ScreenCorners a_corners)
{
	for (uint x=0; x < m_scrwidth; x++) for (uint y=0; y < m_scrheight; y++)
	{
		// Find location on virtual screen
		float relX = (float)x / (float)m_scrwidth;
		float relY = (float)y / (float)m_scrheight;
		vec3 xPos = (a_corners.TR - a_corners.TL) * relX;
		vec3 yPos = (a_corners.BL - a_corners.TL) * relY;
		vec3 pixPos = a_corners.TL + xPos + yPos;

		// Add ray to queue
		vec3 direction = (pixPos - a_location).normalized();
		uint pixelIdx = x + y * m_scrwidth;
		addRayToQueue(Ray(a_location, pixelIdx, direction));
	}
}

// Extends the ray and checks for triangle intersections
void RayTracer::extendRays()
{
	Collision col;
	int rayCount = ((uint*)m_rayQueue)[1];
	int triCount = m_scene->getTriangleCount();
	Triangle* triangles = m_scene->getTriangles();
	for (int i = 1; i <= rayCount; i++)
	{
		Ray ray = m_rayQueue[i];
		col = intersectTriangles(triangles, triCount, ray);
		if (col.N.isNonZero()) addToBuffer(col.colorAt, i); // DEBUG
		else addToBuffer(COLOR::BLACK, i);					// DEBUG
		addCollisionToQueue(col);
	}
}

// Adds the ray to the ray queue and increments the ray count in the header
void RayTracer::addRayToQueue(Ray ray)
{
	// increments rayCount, checks for overflow, adds ray to incremented index to skip header
	uint *queueSize = (uint*)m_rayQueue, *rayCount = (uint*)m_rayQueue + 1;
	if (++*rayCount > *queueSize)
		throw JornamException("RayTracer", "Ray queue overflow.\n", JornamException::ERR);
	else m_rayQueue[*rayCount] = ray;
}

// Adds the collision to the collision queue and increments the collision counter in the header
void RayTracer::addCollisionToQueue(Collision col)
{
	// increments colCount, checks for overflow, adds col to incremented index to skip header
	uint *queueSize = (uint*)m_colQueue, *colCount = (uint*)m_colQueue + 1;
	if (++*colCount > *queueSize)
		throw JornamException("RayTracer", "Collision queue overflow.\n", JornamException::ERR);
	else m_colQueue[*colCount] = col;
}

} // namespace Engine