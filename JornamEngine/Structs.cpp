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
			col.N = (tri.v2 - tri.v0).cross(tri.v1 - tri.v0).normalized();
			col.colorAt = tri.color;
		}
	}
	return col;
}

// Performs a Möller–Trumbore triangle intersection
// Returns the distance to the intersection, or -1.0f if there was none
float Triangle::intersect(Ray ray)
{
	//const float EPSILON = 0.0000001f;
	//
	//vec3 e1 = v1 - v0;
	//vec3 e2 = v2 - v0;
	//vec3 tangent = ray.direction.cross(e2); // the normal to the e2-ray plane
	//float pitch = e1.dot(tangent);			// the orthogonal distance of v1 to the e2-ray plane
	//if (pitch > -EPSILON && pitch < EPSILON) return -1.0f; // ray parallel to triangle
	//
	//float f = 1.0f / pitch;
	//vec3 v0toRay = ray.origin - v0;
	//float u = f * v0toRay.dot(tangent);
	//if (u < 0.0f || u > 1.0f) return -2.0f; // no intersection
	//
	//vec3 q = v0toRay.cross(e1);
	//float v = f * ray.direction.dot(q);
	//if (v < 0.0f || u + v > 1.0f) return -3.0f; // no intersection
	//
	//float t = f * e2.dot(q);
	//if (t < EPSILON) return -4.0f; // behind the camera
	//return t;



	const float EPSILON = 0.0000001f;
	vec3 edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = v1 - v0;
	edge2 = v2 - v0;
	h = ray.direction.cross(edge2);
	a = edge1.dot(h);
	// DEBUG
	//printf("O: (%.1f, %.1f, %.1f)\nR: (%.1f, %.1f, %.1f)\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);
	//printf("v0: (%.1f, %.1f, %.1f)\nv1: (%.1f, %.1f, %.1f)\nv2: %.1f, %.1f, %.1f\n", v0.x, v0.y, v0.z, v1.x, v1.y, v2.z, v2.x, v2.y, v2.z);
	//printf("e1: (%.1f, %.1f, %.1f)f\ne2: (%.1f, %.1f, %.1f)\n", edge1.x, edge1.y, edge1.z, edge2.x, edge2.y, edge2.z);
	//printf("h: (%.1f, %.1f, %.1f)\na: %.1f\n", h.x, h.y, h.z, a);
	if (a > -EPSILON && a < EPSILON) return -1.0f;    // This ray is parallel to this triangle.

	f = 1.0f / a;
	s = ray.origin - v0;
	u = f * (s.dot(h));
	if (u < 0.0 || u > 1.0) return -2.0f;

	q = s.cross(edge1);
	v = f * ray.direction.dot(q);
	if (v < 0.0 || u + v > 1.0) return -3.0f;

	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * edge2.dot(q);
	if (t < EPSILON) return -4.0f;
	
	return t;
}

} // namespace Engine