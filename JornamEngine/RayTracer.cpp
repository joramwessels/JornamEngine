#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(uint a_SSAA)
{
	debugInit(); // DEBUG
	return;		 // DEBUG
	m_SSAA = a_SSAA;

	// Context creation
	m_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
	if (m_context.isValid())
	{
		const uint* devices = { 0 };
		//m_context->setCudaDeviceNumbers(1, devices);
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
	debugRender(camera); // DEBUG
	//createPrimaryRays(camera);
	//traceRays();
	//shadeHits();
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
	optix::prime::Query query = m_scene->getModel()->createQuery(RTP_QUERY_TYPE_CLOSEST);
	query->setRays(m_scrwidth * m_scrheight, RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, RTP_BUFFER_TYPE_CUDA_LINEAR, m_rays->ptr());
	query->setHits(m_scrwidth * m_scrheight, RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, m_hits->ptr());
	query->execute(0);
}

// Turns the hits into colors
void RayTracer::shadeHits()
{
	for (uint i = 0; i < m_scrwidth*m_scrheight; i++)
	{
		if (m_hits->ptr()[i].rayDistance >= 0) m_screen->GetBuffer()[i] = 0xFF0000;
	}
}

void RayTracer::debugInit()
{
	RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;

	//
    // Create Context
    //
	m_context = optix::prime::Context::create(contextType);
	if (contextType == RTP_CONTEXT_TYPE_CPU) {
		std::cerr << "Using cpu context\n";
	}
	else {
		unsigned int device = 0;
		m_context->setCudaDeviceNumbers(1, &device);
		std::cerr << "Using cuda context\n";
	}

	//
	// Create the Model object
	//
	vec3 v0(-2, 2, -2), v1(2, 2, -2), v2(2, -2, -2);
	std::vector<float> vertices({ v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z });
	std::vector<uint> indices({ 0, 1, 2 });

	m_model = m_context->createModel();
	m_model->setTriangles(
		indices.size()/3, RTP_BUFFER_TYPE_HOST, indices.data(),
		vertices.size()/3, RTP_BUFFER_TYPE_HOST, vertices.data()
	);
	m_model->update(0);

	//
	// Create buffers for rays and hits
	//
	m_rays = new Buffer<OptixRay>(m_scrwidth * m_scrheight, bufferType, LOCKED);
	m_hits = new Buffer<OptixHit>(m_scrwidth * m_scrheight, bufferType, LOCKED);

	//
	// Execute query loop
	//
	m_query = m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
}

void RayTracer::debugRender(Camera* camera)
{
	createPrimaryRays(camera);
	m_query->setRays(m_rays->count(), RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, m_rays->type(), m_rays->ptr());
	m_query->setHits(m_hits->count(), RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, m_hits->type(), m_hits->ptr());
	m_query->execute(0);

	//
	// Shade the hit results to create image
	//
	shadeHits();
	printf("shaded\n");
}

} // namespace Engine