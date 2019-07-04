#pragma once

namespace JECUDA {

	struct Color
	{
		union { int hex; struct { unsigned char b, g, r, x; }; };
		__device__ Color(int a) : hex(a) {};
		__device__ inline void operator+=(const Color& c)
		{
			int ar = min(r + c.r, 0xFF), ag = min(g + c.g, 0xFF), ab = min(b + c.b, 0xFF);
			r = (unsigned char)ar; g = (unsigned char)ag; b = (unsigned char)ab;
		}
		__device__ inline Color operator*(const float& s) const
		{
			int ar = min((int)((float)((hex >> 16) & 0xFF) * s), 0xFF);
			int ag = min((int)((float)((hex >> 8) & 0xFF) * s), 0xFF);
			int ab = min((int)((float)(hex & 0xFF) * s), 0xFF);
			return ((ar << 16) & 0xFF0000) | ((ag << 8) & 0xFF00) | (ab & 0xFF);
		}
		__device__ inline Color operator*(const Color& c) const
		{
			int ar = min((((hex >> 16) & 0xFF) * ((c.hex >> 16) & 0xFF)) / 255, 0xFF);
			int ag = min((((hex >> 8) & 0xFF)  * ((c.hex >> 8) & 0xFF)) / 255, 0xFF);
			int ab = min(((hex & 0xFF)			* (c.hex & 0xFF)) / 255, 0xFF);
			return ((ar << 16) & 0xFF0000) | ((ag << 8) & 0xFF00) | (ab & 0xFF);
		}
	};
	struct Light { float3 pos; Color color; };
	struct OptixRay { float3 origin, direction; };
	struct OptixHit { float rayDistance; int triangleIdx; int instanceIdx; float u, v; };
	struct Mesh { int3* indices; float3* normals; };
	struct Texture { union { Color* buffer; long color; }; int width, height; };
	struct PhongMaterial {
		float spec, diff, ambi, shin;
		PhongMaterial(float spec, float diff, float ambi, float shin) {
			spec = spec; diff = diff; ambi = ambi; shin = shin; }
	};
	struct TransformMatrix
	{
		float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
		__device__ inline float3 operator*(float3 &a) const
		{
			return make_float3(
				t0 * a.x + t1 * a.y + t2 * a.z + t3,
				t4 * a.x + t5 * a.y + t6 * a.z + t7,
				t8 * a.x + t9 * a.y + t10 * a.z + t11
			);
		}
	};
	struct Transform { TransformMatrix matrix, inverse; };
	struct Object3D
	{
		//optix::prime::Model m_primeHandle; // 8 bytes
		void* placeholder; // 8 bytes
		int m_meshIdx;
		int m_textureIdx;
		PhongMaterial m_material;
		Transform m_transform;
	};

}

/*
	Initializes the primary rays on the GPU

	@param rays		A device pointer to an array of consecutive (origin, direction) pairs
	@param width	The width of the screen in pixels
	@param height	The height of the screen in pixels
*/
extern void createPrimaryRaysOnDevice(
	float3* rays, unsigned int width, unsigned int height,
	float3 eye, float3 TR, float3 TL, float3 BL, unsigned int blockX = 32, unsigned int blockY = 16
);

/*
	Shades the hits and fills the pixel buffer

	@param buffer		A device pointer to an allocated pixel buffer
	@param rays			A device pointer to the rays array
	@param hits			A divice pointer to the hits array
	@param objects		A device pointer to the 3D objects
	@param meshes		A device pointer to the meshes
	@param lights		A device pointer to the lights int the scene
	@param lightCount	The number of lights in the scene
	@param ambiLight	The color of the ambient light in the scene
	@param height		The height of the screen in pixels
	@param width		The width of the screen in pixels
*/
extern void shadeHitsOnDevice(
	JECUDA::Color* buffer, void* rays, void* hits, const JECUDA::Object3D* objects, const JECUDA::Mesh* meshes, const JECUDA::Texture* textures,
	const JECUDA::Light* lights, int lightCount, float3 camera, JECUDA::Color ambiLight, int height, int width,
	unsigned int blockX = 32, unsigned int blockY = 16
);