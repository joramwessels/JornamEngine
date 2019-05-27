#include "headers.h"

inline int idivCeil(int x, int y)
{
	return (x + y - 1) / y;
}

__device__ float3 operator-(const float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3 &a, const float &b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 normalized(const float3 &a)
{
	float norm = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
	return make_float3(a.x / norm, a.y / norm, a.z / norm);
}

__device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ typedef BYTE CudaColor;

__device__ struct CudaOptixHit { float rayDistance; int triangleIdx; int instanceIdx; float u; float v; };
__device__ struct CudaOptixRay { float3 origin, direction; };

__global__ void cudaCreatePrimaryRays(float3* rays, unsigned int width, unsigned int height, float3 eye, float3 TR, float3 TL, float3 BL)
{
	// Get pixel ID
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	// Find location on virtual screen
	float relX = (float)x / (float)width;
	float relY = (float)y / (float)height;
	float3 xPos = (TR - TL) * relX;
	float3 yPos = (BL - TL) * relY;
	float3 pixPos = TL + xPos + yPos;

	// Add ray to queue
	float3 direction = normalized(pixPos - eye);
	unsigned int pixelIdx = (x + y * width) * 2;
	rays[pixelIdx] = eye;			// Origin
	rays[pixelIdx + 1] = direction;	// Direction
}

extern void createPrimaryRaysOnDevice(float3* rays, unsigned int width, unsigned int height, float3 eye, float3 TR, float3 TL, float3 BL, unsigned int blockX, unsigned int blockY)
{
	dim3 blockSize(blockX, blockY);
	dim3 gridSize(idivCeil(width, blockSize.x), idivCeil(height, blockSize.y));
	cudaCreatePrimaryRays<<<gridSize, blockSize>>>(rays, width, height, eye, TR, TL, BL);
}

//
// constants: {diffuse, specular, ambient, shinyness}
__global__ void cudaShadeHits(CudaOptixHit* hits, CudaOptixRay* rays, CudaColor* buffer, unsigned int width, unsigned int height, float3 eye, float4 constants)
{
	// Get pixel ID
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;
	int pixid = x + y * width;

	CudaColor I, tricolor;
	float3 loc, N, V, L, R;

	I = 0;
	CudaOptixHit hit = hits[pixid];
	loc = rays[pixid].origin + rays[pixid].direction * hit.rayDistance;

	if (hit.rayDistance < 0)
	{
		// TODO Skybox intersection
	}
	else
	{
		// Phong
		I += m_scene->getAmbientLight() * constants.z;
		tricolor = m_scene->getModel(hit.instanceIdx).color;	// TODO change this using u and v to implement textures
		N = m_scene->getModel(hit.instanceIdx).N[hit.triangleIdx];	// Surface normal
		V = (eye - loc).normalized();								// Ray to viewer
		for (unsigned int j = 0; j < m_scene->getLightCount(); j++)
		{
			L = (lights[j].pos - loc).normalized();			// Light source direction
			R = normalized(-L - N * 2 * dot(-L, N));		// Perfect reflection
			I += (lights[j].color * tricolor) * constants.x * dot(L, N);		// Diffuse aspect
			I += lights[j].color * constants.y * pow(dot(R, V), constants.w);	// Specular aspect
		}
	}
	buffer[pixid] = I;
}