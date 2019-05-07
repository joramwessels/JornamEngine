#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene, uint a_SSAA)
{
	m_scene = a_scene;
	m_SSAA = a_SSAA;

	if (rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &m_context) == RTP_SUCCESS)
	{
		const uint devicenumbers[] = { 0, 1 };
		rtpContextSetCudaDeviceNumbers(m_context, 2, devicenumbers);
	}
	else throw JornamException("RayTracer", "CUDA device not found", JornamException::FATAL);
	createBuffers();
}

// Called at the start of every frame
void RayTracer::tick()
{
	
}

void RayTracer::render(Camera* camera)
{
	createPrimaryRays();
	traceRays();
	shadeHits();
}

// Creats the ray- and hits buffers
void RayTracer::createBuffers()
{

}

// Adds rays to the ray buffer
void RayTracer::createPrimaryRays()
{

}

// Finds the closest hit for every primary ray
void RayTracer::traceRays()
{

}

// Turns the hits into colors
void RayTracer::shadeHits()
{

}

} // namespace Engine