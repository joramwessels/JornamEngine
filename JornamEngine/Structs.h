#pragma once

#define JE_EPSILON 0.00025f

// Flags
#define JE_RAY_IS_SHADOWRAY 1
#define JE_SHADOWRAY_COLOR 0xFFFFFF00

namespace JornamEngine{

// A quaternion for rotations (16 bytes)
union Quaternion {
	struct
	{
		float r;
		vec3 v;
	};
	float cell[4];

	Quaternion(float real, vec3 imaginary) :
		r(real), v(imaginary.normalized())
	{
		float d = sqrt(r * r + v.x * v.x + v.y * v.y + v.z * v.z);
		r = r / d;
		v = v / d;
	};
	Quaternion(vec3 axis, float angle) :
		r(cos(angle / 2.0f)), v((axis * sin(angle / 2.0f)).normalized())
	{
		float d = sqrt(r * r + v.x * v.x + v.y * v.y + v.z * v.z);
		r = r / d;
		v = v / d;
	};

	inline Quaternion operator * (const Quaternion& a) const { return Quaternion(r*a.r - v.dot(a.v), a.v * r + v * a.r + v.cross(a.v)); }
};

// A ray for streaming ray tracing (32 bytes)
union Ray {
	struct
	{ // Primary ray
		uint flags;
		vec3 origin;
		uint pixelIdx;
		vec3 direction;
	};
	//struct
	//{ // Shadow ray
	//	uint flags;
	//	vec3 origin;
	//	uint pixelIdx;
	//	Color color;
	//	uint lightIdx;
	//	uint energy; // unused
	//};
	float cell[8];

	// Primary ray constructor
	Ray(vec3 origin, uint pixelIdx, vec3 direction, uint flags=0) :
		origin(origin), pixelIdx(pixelIdx), direction(direction), flags(flags) {};

	//// Shadow ray constructor
	//Ray(vec3 origin, uint pixelIdx, Color color, uint lightIdx, uint flags = 0) :
	//	origin(origin), pixelIdx(pixelIdx), color(color), lightIdx(lightIdx), flags(flags & JE_RAY_IS_SHADOWRAY & (color << 8)) {};
};

// A collision between a ray and a triangle (32 bytes)
union Collision
{
	struct
	{
		vec3 position;
		uint pixelIdx;
		vec3 N;
		Color colorAt;
	};
	float cell[8];

	Collision(vec3 position, uint pixelIdx, vec3 normal, Color colorAt) :
		position(position), pixelIdx(pixelIdx), N(normal), colorAt(colorAt) {};
	Collision() : N(vec3(0, 0, 0)), colorAt(0) {};

};

// Three vertices and a color (40 bytes)
union Triangle {
	struct
	{
		vec3 v0, v1, v2;
		Color color;
	};
	float cell[10];

	// Only provides correct normal if vertices are provided in clockwise order
	Triangle() :
		v0(vec3(0)), v1(vec3(0)), v2(vec3(0)), color(COLOR::WHITE) {};
	Triangle(vec3 v0, vec3 v1, vec3 v2, Color color) :
		v0(v0), v1(v1), v2(v2), color(color) {};

	float intersect(Ray ray);
};

Collision intersectTriangles(Triangle* triangles, int triCount, Ray ray);
bool checkOcclusion(Triangle* triangles, int triCount, Ray ray, float maxDistance);

} // namespace Engine