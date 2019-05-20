#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene)
{
	m_scene = a_scene;

	// Ray buffer, hit buffer, and query
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST; // TODO what type should this be?
	m_rays = new Buffer<OptixRay>(m_scrwidth * m_scrheight, bufferType, LOCKED);
	m_hits = new Buffer<OptixHit>(m_scrwidth * m_scrheight, bufferType, LOCKED);
	m_query = m_scene->getModel()->createQuery(RTP_QUERY_TYPE_CLOSEST);
}

// Called at the start of every frame
void RayTracer::tick()
{
	
}

// Renders the scene each frame
void RayTracer::render(Camera* camera)
{
	createPrimaryRays(camera);
	traceRays();
	shadeHits();
}

// Adds rays to the ray buffer
void RayTracer::createPrimaryRays(Camera* camera)
{
	vec3 eye = camera->getLocation();
	ScreenCorners corners = camera->getScreenCorners();
	for (uint x = 0; x < m_scrwidth; x++) for (uint y = 0; y < m_scrheight; y++)
	{
		// Find location on virtual screen
		float relX = (float)x / (float)m_scrwidth;
		float relY = (float)y / (float)m_scrheight;
		vec3 xPos = (corners.TR - corners.TL) * relX;
		vec3 yPos = (corners.BL - corners.TL) * relY;
		vec3 pixPos = corners.TL + xPos + yPos;

		// Add ray to queue
		vec3 direction = (pixPos - eye).normalized();
		uint pixelIdx = x + y * m_scrwidth;
		m_rays->ptr()[pixelIdx] = OptixRay(eye, direction);
	}
}

// Finds the closest hit for every primary ray
void RayTracer::traceRays()
{
	m_query->setRays(m_rays->count(), RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, m_rays->type(), m_rays->ptr());
	m_query->setHits(m_hits->count(), RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, m_hits->type(), m_hits->ptr());
	m_query->execute(0);
}

// Turns the hits into colors
void RayTracer::shadeHits()
{
	for (uint i = 0; i < m_scrwidth*m_scrheight; i++)
	{
		if (m_hits->ptr()[i].rayDistance >= 0) m_screen->GetBuffer()[i] = 0xFF0000;
	}
}

} // namespace Engine