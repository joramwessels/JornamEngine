#include "headers.h"

namespace JornamEngine {

// Called at the start of the application to initialize the renderer
void RayTracer::init(Scene* a_scene)
{
	m_scene = a_scene;

	// Ray buffer, hit buffer, and query
	m_rays = new Buffer<OptixRay>(m_scrwidth * m_scrheight, m_buffertype, LOCKED);
	m_hits = new Buffer<OptixHit>(m_scrwidth * m_scrheight, m_buffertype, LOCKED);
	m_query = m_scene->getSceneModel()->createQuery(RTP_QUERY_TYPE_CLOSEST);

	cudaMalloc(&c_buffer, m_scrwidth * m_scrheight * sizeof(Color));
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
	if (m_rays->type() == RTP_BUFFER_TYPE_CUDA_LINEAR)
	{
		vec3 loc = camera->getLocation();
		shadeHitsOnDevice(
			c_buffer, m_rays->ptr(), m_hits->ptr(),
			m_scene->getDeviceObjects(), m_scene->getDeviceMeshes(), m_scene->getDeviceTextures(),
			m_scene->getDeviceLights(), m_scene->getLightCount(),
			make_float3(loc.x, loc.y, loc.z), m_scene->getAmbientLight().hex,
			m_scrheight, m_scrwidth
		);
		cudaMemcpy(m_screen->GetBuffer(), c_buffer, m_scrwidth * m_scrheight * sizeof(Color), cudaMemcpyDeviceToHost);
		return;
	}

	Color* buffer = m_screen->GetBuffer();
	const OptixRay* rays = m_rays->hostPtr();
	OptixHit* hits = m_hits->ptr();
	const Light* lights = m_scene->getHostLights();

	Color I, color;
	vec3 loc, N, V, L, R;
	for (uint pixid = 0; pixid < m_scrwidth*m_scrheight; pixid++)
	{
		I = 0;
		OptixHit hit = hits[pixid];
		loc = rays[pixid].origin + rays[pixid].direction * hit.rayDistance;

		if (hit.rayDistance < 0)
		{
			I = 0xAAAADD; // DEBUG
			// TODO Skybox intersection
		}
		else
		{
			// Phong reflection
			Object3D object = m_scene->getObject(hit.instanceIdx);
			PhongMaterial mat = object.getMaterial();
			vec3 eye = object.getInvTrans() * camera->getLocation();

			// Interpolating and transforming surface normal
			N = m_scene->interpolateNormal(hit.instanceIdx, hit.triangleIdx, hit.u, hit.v);
			N = (object.getInvTrans() * N).normalized();
			color = m_scene->interpolateTexture(hit.instanceIdx, hit.triangleIdx, hit.u, hit.v);

			I += m_scene->getAmbientLight() * mat.ambi * color;
			V = (eye - loc).normalized();								// Ray to viewer
			for (uint j = 0; j < m_scene->getLightCount(); j++)
			{
				L = (lights[j].pos - loc).normalized();						// Light source direction
				R = (-L - N * 2 * (-L).dot(N)).normalized();				// Perfect reflection
				I += (lights[j].color * color) * mat.diff * max(0.0f, L.dot(N));		// Diffuse aspect
				I += lights[j].color * mat.spec * pow(max(0.0f, R.dot(V)), mat.shin);		// Specular aspect
			}
		}
		buffer[pixid] = I;
	}
}

} // namespace Engine