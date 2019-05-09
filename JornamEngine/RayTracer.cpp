#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(uint a_SSAA)
{
	m_SSAA = a_SSAA;

	if (rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &m_context) == RTP_SUCCESS)
	{
		const uint devicenumbers[] = { 0, 1 };
		rtpContextSetCudaDeviceNumbers(m_context, 2, devicenumbers);
	}
	else throw JornamException("RayTracer", "CUDA device not found", JornamException::FATAL);
	m_rayVector = new OptixRay[m_scrwidth * m_scrheight];
	m_hitsVector = new OptixHit[m_scrwidth * m_scrheight];
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, RTP_BUFFER_TYPE_CUDA_LINEAR, m_rayVector, &m_rays);
	rtpBufferDescCreate(m_context, RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_HOST, m_hitsVector, &m_hits);
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
		m_rayVector[pixelIdx] = OptixRay(eye, direction);
	}
}

// Finds the closest hit for every primary ray
void RayTracer::traceRays()
{
	RTPquery query;
	rtpQueryCreate(m_scene->getModel(), RTP_QUERY_TYPE_CLOSEST, &query);
	rtpQuerySetRays(query, m_rays);
	rtpQuerySetHits(query, m_hits);
	rtpQueryExecute(query, RTP_QUERY_HINT_NONE);
}

// Turns the hits into colors
void RayTracer::shadeHits()
{
	for (int i = 0; i < m_scrwidth*m_scrheight; i++)
	{
		if (m_hitsVector[i].rayDistance >= 0) m_screen->GetBuffer()[i] = 0xFF0000;
	}
}

} // namespace Engine