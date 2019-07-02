#include "headers.h"

using namespace JECUDA;

inline int idivCeil(int x, int y)
{
	return (x + y - 1) / y;
}

__device__ float3 operator-(const float3 &a) {
	return make_float3(-a.x, -a.y, -a.z); }

__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

__device__ float3 operator*(const float3 &a, const float &b) {
	return make_float3(a.x * b, a.y * b, a.z * b); }

__device__ float3 normalized(const float3 &a) {
	float norm = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
	return make_float3(a.x / norm, a.y / norm, a.z / norm); }

__device__ float dot(const float3 &a, const float3 &b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z); }

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

/*
	Interpolates triangle surface normal given Barycentric coordinates

	@param meshes	A pointer to the mesh array on device
	@param objects	A pointer to the objects array on device
	@param trIdx	The triangle index
	@param obIdx	The object index
	@param u		The Barycentric u coordinate
	@param v		The barycentric v coordinate
	@return			The surface normal at the given coordinates
*/
__device__ float3 interpolateNormal(const Mesh* meshes, const Object3D* objects, int o, int t, float u, float v)
{
	Mesh mesh = meshes[objects[o].m_meshIdx];
	int3 vx = mesh.indices[t];
	float3 n0 = mesh.normals[vx.x], n1 = mesh.normals[vx.y], n2 = mesh.normals[vx.z];
	return normalized(n0 * u + n1 * v + n2 * (1 - u - v));
}

__global__ void cudaShadeHits( Color* buffer, OptixRay* rays, OptixHit* hits, const Object3D* objects,
	const Mesh* meshes, const Light* lights, int lightCount, float3 camera, Color ambiLight, int width)
{
	// Get pixel ID
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixid = x + y * width;

	Color I = 0;
	OptixHit hit = hits[pixid];

	if (hit.rayDistance < 0)
	{
		I = 0xAAAADD; // DEBUG
					  // TODO Skybox intersection
	}
	else
	{
		float3 N, V, L, R;

		// Phong reflection
		float3 loc = rays[pixid].origin + rays[pixid].direction * hit.rayDistance;
		Object3D object = objects[hit.instanceIdx];
		PhongMaterial mat = object.m_material;
		float3 eye = object.m_transform.inverse * camera;

		// Interpolating and transforming surface normal
		N = interpolateNormal(meshes, objects, hit.instanceIdx, hit.triangleIdx, hit.u, hit.v);
		N = normalized(object.m_transform.inverse * N);

		I += ambiLight * mat.ambi;
		Color tricolor = object.m_color; // TODO change this using u and v to implement textures
		V = normalized(eye - loc); // Ray to viewer
		for (int j = 0; j < lightCount; j++)
		{
			L = normalized(lights[j].pos - loc);	 // Light source direction
			R = normalized(-L - N * 2 * dot(-L, N)); // Perfect reflection
			I += (lights[j].color * tricolor) * mat.diff * max(0.0f, dot(L, N));   // Diffuse aspect
			I += lights[j].color * mat.spec * pow(max(0.0f, dot(R, V)), mat.shin); // Specular aspect
		}
	}
	buffer[pixid] = I;
}

extern void shadeHitsOnDevice( JECUDA::Color* buffer, void* rays, void* hits, const JECUDA::Object3D* objects, const JECUDA::Mesh* meshes,
	const JECUDA::Light* lights, int lightCount, float3 camera, JECUDA::Color ambiLight, int height, int width, unsigned int blockX, unsigned int blockY)
{
	dim3 blockSize(blockX, blockY);
	dim3 gridSize(idivCeil(width, blockSize.x), idivCeil(height, blockSize.y));
	cudaShadeHits<<<gridSize, blockSize>>>(buffer, (OptixRay*)rays, (OptixHit*)hits, objects, meshes, lights, lightCount, camera, ambiLight, width);
}