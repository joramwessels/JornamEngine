#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene)
{
	m_scene = a_scene;

	// Ray buffer, hit buffer, and query
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST; // TODO what type should this be?
	m_rays = new Buffer<OptixRay>(m_scrwidth * m_scrheight, RTP_BUFFER_TYPE_CUDA_LINEAR, LOCKED);
	m_hits = new Buffer<OptixHit>(m_scrwidth * m_scrheight, bufferType, LOCKED);
	m_query = m_scene->getSceneModel()->createQuery(RTP_QUERY_TYPE_CLOSEST);
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
	shadeHits(camera);
}

// Adds rays to the ray buffer
void RayTracer::createPrimaryRays(Camera* camera)
{
	// Calling CUDA kernel if rays are on device
	if (m_rays->type() == RTP_BUFFER_TYPE_CUDA_LINEAR)
	{
		ScreenCorners cor = camera->getScreenCorners();
		createPrimaryRaysOnDevice((float3*)m_rays->ptr(), m_scrwidth, m_scrheight,
			vtof3(camera->getLocation()), vtof3(cor.TR), vtof3(cor.TL), vtof3(cor.BL));
		return;
	}

	// Looping through pixels if rays are on host
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
void RayTracer::shadeHits(Camera* camera)
{
	float diffConst = 0.3f, specConst = 0.3f, ambConst = 1.0f, shinConst = 10.0f; // TODO move to renderer

	Color* buffer = m_screen->GetBuffer();
	const OptixRay* rays = m_rays->hostPtr();
	OptixHit* hits = m_hits->ptr();
	const Light* lights = m_scene->getLights();

	Color I, tricolor;
	vec3 loc, N, V, L, R;
	for (uint pixid = 0; pixid < m_scrwidth*m_scrheight; pixid++)
	{
		I = 0;
		OptixHit hit = hits[pixid];
		loc = rays[pixid].origin + rays[pixid].direction * hit.rayDistance;

		if (hit.rayDistance < 0)
		{
			I = 0x1A1ABB; // DEBUG
			// TODO Skybox intersection
		}
		else
		{
			// Phong reflection
			Object3D object = m_scene->getModel(hit.instanceIdx);
			vec3 eye = object.getInvTrans() * camera->getLocation();

			// Interpolating and transforming surface normal
			N = object.interpolateNormal(hit.triangleIdx, hit.u, hit.v);
			N = object.getInvTrans() * N;

			I += m_scene->getAmbientLight() * ambConst;
			tricolor = object.getColor();								// TODO change this using u and v to implement textures
			V = (eye - loc).normalized();								// Ray to viewer
			for (uint j = 0; j < m_scene->getLightCount(); j++)
			{
				L = (lights[j].pos - loc).normalized();						// Light source direction
				R = (-L - N * 2 * (-L).dot(N)).normalized();				// Perfect reflection
				I += (lights[j].color * tricolor) * diffConst * max(0, L.dot(N));		// Diffuse aspect
				I += lights[j].color * specConst * pow(max(0, R.dot(V)), shinConst);	// Specular aspect
			}
		}
		buffer[pixid] = I;
	}
}

} // namespace Engine