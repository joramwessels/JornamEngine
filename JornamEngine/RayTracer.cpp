#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(uint a_SSAA)
{
	m_SSAA = a_SSAA;

	// Context creation
	m_context.create(RTP_CONTEXT_TYPE_CUDA);
	if (m_context.isValid())
	{
		uint* devices = { 0 };
		m_context->setCudaDeviceNumbers(1, devices);
		logDebug("RayTracer", "CUDA device found", JornamException::INFO);
	}
	else logDebug("RayTracer", "CUDA device not found", JornamException::FATAL);

	// Ray and collision buffers
	Buffer<OptixRay> m_rays(m_scrwidth * m_scrheight, RTP_BUFFER_TYPE_CUDA_LINEAR, LOCKED);
	Buffer<OptixHit> m_hits(m_scrwidth * m_scrheight, RTP_BUFFER_TYPE_CUDA_LINEAR, LOCKED);
}

// Called at the start of every frame
void RayTracer::tick()
{
	
}

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
		m_rays.ptr()[pixelIdx] = OptixRay(eye, direction);
	}
}

// Finds the closest hit for every primary ray
void RayTracer::traceRays()
{
	optix::prime::Query query = m_scene->getModel()->createQuery(RTP_QUERY_TYPE_CLOSEST);
	query->setRays(m_scrwidth * m_scrheight, RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, RTP_BUFFER_TYPE_CUDA_LINEAR, m_rays.ptr());
	query->setHits(m_scrwidth * m_scrheight, RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, m_hits.ptr());
	query->execute(0);
}

// Turns the hits into colors
void RayTracer::shadeHits()
{
	for (int i = 0; i < m_scrwidth*m_scrheight; i++)
	{
		if (m_hits.ptr()[i].rayDistance >= 0) m_screen->GetBuffer()[i] = 0xFF0000;
	}
}

} // namespace Engine