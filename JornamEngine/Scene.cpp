// NOTE: .scene parsing
//
// Triangle (provide vertices in clockwise order)
// T v0.x,v0.y,v0.z v1.x,v1.y,v1.z v2.x,v2.y,v2.z
// 
// Plane (provide vertices in clockwise order)
// P v0.x,v0.y,v0.z v1.x,v1.y,v1.z v2.x,v2.y,v2.z v3.x,v3.y,v3.z
//
// Light
// L p.x,p.y,p.z color(hex)
//
// Skybox
// S filepath

#include "headers.h"

namespace JornamEngine {

void Scene::loadScene(char* filename)
{
	// check filename extension .scene
	// reset numlights and numtriangles
	// open file
	// if line startswith # or isempty continue
	// if first char is T parse triangle
	// if first char is P parse plane
	// if first char is L parse light
	// if first char is S load skybox
}

void Scene::parseTriangle(char* line)
{
	// skip first T
	// vec3 v0, v1, v2;
	// for v0, v1, v2; x, y, z
		// until you see a comma interpret value
	// addTriangle(Triangle(v0, v1, v2));
}

void Scene::parsePlane(char* line)
{
	// skip first P
	// vec3 v0, v1, v2, v3
	// for 4 loops
		// until you see comma, interpret value as vertex
	// addTriangle(Triangle(v0, v1, v2));
	// addTriangle(Triangle(v2, v3, v0));
}

void Scene::parseLight(char* line)
{
	// skip first L
	// vec3 p;
	// Pixel c;
	// read position
	// read color
	// addLight(Light(p, c));
}

// DEBUG unit test
void Scene::printTriangles()
{
	for (int i = 0; i < m_numTriangles; i++)
	{
		printf("T %.1f,%.1f,%.1f %.1f,%.1f,%.1f %.1f,%.1f,%.1f\n",
			m_triangles[i].v0.x, m_triangles[i].v0.y, m_triangles[i].v0.z,
			m_triangles[i].v1.x, m_triangles[i].v1.y, m_triangles[i].v1.z,
			m_triangles[i].v2.x, m_triangles[i].v2.y, m_triangles[i].v2.z);
		printf("Normal: %.1f,%.1f,%.1f\n",
			m_triangles[i].N.x, m_triangles[i].N.y, m_triangles[i].N.z);
	}
}

// DEBUG unit test
void Scene::printLights()
{
	for (int i = 0; i < m_numLights; i++)
	{
		printf("L %.1f,%.1f,%.1f %X\n",
			m_lights[i].pos.x, m_lights[i].pos.y, m_lights[i].pos.z,
			m_lights[i].col);
	}
}

} // namespace Engine