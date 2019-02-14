#pragma once

namespace JornamEngine{

// A ray for streaming ray tracing (32 bytes)
union Ray {
	struct
	{
		vec3 origin;
		uint pixelIdx;
		vec3 direction;
		uint flags;
	};
	float cell[8];

	Ray(vec3 origin, uint pixelIdx, vec3 direction) :
		origin(origin), pixelIdx(pixelIdx), direction(direction) {};
};

// A collision between a ray and a triangle (32 bytes)
union Collision
{
	struct
	{
		vec3 position;
		uint flags;
		vec3 N;
		Color colorAt;
	};
	float cell[8];

	Collision(vec3 position, uint flags, vec3 normal, Color colorAt) :
		position(position), flags(flags), N(normal), colorAt(colorAt) {};
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
	Triangle(vec3 v0, vec3 v1, vec3 v2, Color color) :
		v0(v0), v1(v1), v2(v2), color(color) {};
	Triangle(vec3 v0, vec3 v1, vec3 v2) :
		v0(v0), v1(v1), v2(v2), color(0x00FF0000) {}; // DEBUG

	float intersect(Ray ray);
};

Collision intersectTriangles(Triangle* triangles, int triCount, Ray ray);

} // namespace Engine