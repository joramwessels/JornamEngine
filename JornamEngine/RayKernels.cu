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

/*
	Fills the ray buffer with primary rays

	@param rays		A pointer to the Optix rays
	@param width	The width of the screen
	@param height	The height of the screen
	@param eye		The location of the camera object
	@param TR		The top right corner of the virtual screen
	@param TL		The top left corner of the virtual screen
	@param BL		The bottom left corner of the virtual screen
*/
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
	@param obIdx	The object index
	@param trIdx	The triangle index
	@param u		The Barycentric u coordinate
	@param v		The barycentric v coordinate
	@returns		The surface normal at the given coordinates
*/
__device__ float3 interpolateNormal(const Mesh* meshes, const Object3D object, int t, float u, float v)
{
	Mesh mesh = meshes[object.m_meshIdx];
	Index* indices = mesh.indices;
	float3 n0 = mesh.normals[indices[3 * t + 0].normalIdx];
	float3 n1 = mesh.normals[indices[3 * t + 1].normalIdx];
	float3 n2 = mesh.normals[indices[3 * t + 2].normalIdx];
	return normalized(n0 * u + n1 * v + n2 * (1 - u - v));
}

/*
	Interpolates the object texture given Barycentric coordinates

	@param meshes	A pointer to the texture array on device
	@param objects	A pointer to the objects array on device
	@param trIdx	The triangle index
	@param obIdx	The object index
	@param u		The Barycentric u coordinate
	@param v		The barycentric v coordinate
	@returns		The color at the given coordinates
*/
__device__ Color interpolateTexture(const Mesh* meshes, const Texture* textures, const Object3D object, int t, float u, float v)
{
	Texture texture = textures[object.m_textureIdx];
	Mesh mesh = meshes[object.m_meshIdx];
	if (texture.width <= 1 && texture.height <= 1) return Color(texture.color);

	// Unpacking texture coordinates
	int idx1 = mesh.indices[3 * t + 0].textureIdx;
	int idx2 = mesh.indices[3 * t + 1].textureIdx;
	int idx3 = mesh.indices[3 * t + 2].textureIdx;
	float2 texcoord1 = mesh.texcoords[idx1];
	float2 texcoord2 = mesh.texcoords[idx2];
	float2 texcoord3 = mesh.texcoords[idx3];

	// Texture wrapping
	if (texcoord1.x > 1.0f || texcoord1.x < 0.0f) texcoord1.x -= floor(texcoord1.x);
	if (texcoord1.y > 1.0f || texcoord1.y < 0.0f) texcoord1.y -= floor(texcoord1.y);
	if (texcoord2.x > 1.0f || texcoord2.x < 0.0f) texcoord2.x -= floor(texcoord2.x);
	if (texcoord2.y > 1.0f || texcoord2.y < 0.0f) texcoord2.y -= floor(texcoord2.y);
	if (texcoord3.x > 1.0f || texcoord3.x < 0.0f) texcoord3.x -= floor(texcoord3.x);
	if (texcoord3.y > 1.0f || texcoord3.y < 0.0f) texcoord3.y -= floor(texcoord3.y);

	// Interpolating coordinates
	int x = (texcoord1.x * u + texcoord2.x * v + texcoord3.x * (1 - u - v)) * texture.width;
	int y = (texcoord1.y * u + texcoord2.y * v + texcoord3.y * (1 - u - v)) * texture.height;
	return texture.buffer[x + y * texture.width];
}

/*
	Turns hits into pixel colors

	@param buffer		A pointer to the resulting pixel buffer
	@param rays			A pointer to the Optix rays
	@param hits			A pointer to the Optix hits
	@param objects		A pointer to the Object3D array on device
	@param meshes		A pointer to the Mesh array on device
	@param textures		A pointer to the Texture array on device
	@param lights		A poitner to the Light array on device
	@param lightCount	The number of lights in the scene
	@param camera		The location of the camera object
	@param ambiLight	The color of the ambilight in the scene
	@param width		The width of the screen
*/
__global__ void cudaShadeHits( Color* buffer, OptixRay* rays, OptixHit* hits, const Object3D* objects,
	const Mesh* meshes, const Texture* textures, const Light* lights, int lightCount, float3 camera, Color ambiLight, int width)
{
	// Get pixel ID
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixid = x + y * width;

	Color I = 0, color = 0;
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
		N = interpolateNormal(meshes, object, hit.triangleIdx, hit.u, hit.v);
		N = normalized(object.m_transform.inverse * N);
		color = interpolateTexture(meshes, textures, object, hit.triangleIdx, hit.u, hit.v);

		I += ambiLight * mat.ambi * color;
		V = normalized(eye - loc); // Ray to viewer
		for (int j = 0; j < lightCount; j++)
		{
			L = normalized(lights[j].pos - loc);	 // Light source direction
			R = normalized(-L - N * 2 * dot(-L, N)); // Perfect reflection
			I += (lights[j].color * color) * mat.diff * max(0.0f, dot(L, N));   // Diffuse aspect
			I += lights[j].color * mat.spec * pow(max(0.0f, dot(R, V)), mat.shin); // Specular aspect
		}
	}
	buffer[pixid] = I;
}

extern void shadeHitsOnDevice( JECUDA::Color* buffer, void* rays, void* hits, const JECUDA::Object3D* objects, const JECUDA::Mesh* meshes, const JECUDA::Texture* textures,
	const JECUDA::Light* lights, int lightCount, float3 camera, JECUDA::Color ambiLight, int height, int width, unsigned int blockX, unsigned int blockY)
{
	dim3 blockSize(blockX, blockY);
	dim3 gridSize(idivCeil(width, blockSize.x), idivCeil(height, blockSize.y));
	cudaShadeHits<<<gridSize, blockSize>>>(buffer, (OptixRay*)rays, (OptixHit*)hits, objects, meshes, textures, lights, lightCount, camera, ambiLight, width);
}