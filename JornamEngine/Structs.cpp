#include "headers.h"

namespace JornamEngine {

// Checks all triangles in the array for intersection with the ray
// Returns the closest triangle collision, or a collision with N = (0, 0, 0) if none was found
Collision intersectTriangles(Triangle* a_triangles, int a_triCount, Ray a_ray)
{
	Collision col = Collision();
	float closestDist = -1.0f;
	for (int i = 0; i <= a_triCount; i++)
	{
		Triangle tri = a_triangles[i];
		float dist = tri.intersect(a_ray);
		if (dist > 0.0f && (dist < closestDist || closestDist < 0.0f))
		{
			closestDist = dist;
			col.position = a_ray.origin + a_ray.direction * dist;
			col.pixelIdx = a_ray.pixelIdx;
			col.N = (tri.v2 - tri.v0).cross(tri.v1 - tri.v0).normalized();
			col.colorAt = tri.color;
		}
	}
	return col;
}

// Intersects the extended shadow ray with the scene but only returns whether it intersected
bool checkOcclusion(Triangle* a_triangles, int a_triCount, Ray a_ray, float a_maxDistance)
{
	Collision col = Collision();
	for (int i = 0; i <= a_triCount; i++)
	{
		Triangle tri = a_triangles[i];
		float dist = tri.intersect(a_ray);
		if (dist > 0.0f && dist < a_maxDistance) return true;
	}
	return false;
}

// Performs a Möller–Trumbore triangle intersection
// Returns the distance to the intersection, or -1.0f if there was none
float Triangle::intersect(Ray ray)
{
	const float EPSILON = 0.0000001f;
	
	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;
	vec3 tangent = ray.direction.cross(e2);		// the normal to the e2-ray plane
	float pitch = e1.dot(tangent);				// the orthogonal distance of v1 to the e2-ray plane
	if (pitch > -EPSILON && pitch < EPSILON) return -1.0f; // ray parallel to triangle
	
	float f = 1.0f / pitch;
	vec3 v0toRay = ray.origin - v0;
	float u = f * v0toRay.dot(tangent);
	if (u < 0.0f || u > 1.0f) return -2.0f;		// no intersection
	
	vec3 q = v0toRay.cross(e1);
	float v = f * ray.direction.dot(q);
	if (v < 0.0f || u + v > 1.0f) return -3.0f; // no intersection
	
	float t = f * e2.dot(q);
	if (t < EPSILON) return -4.0f;				// behind the camera
	return t;
}

} // namespace Engine