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
	try
	{
		if (!filenameHasExtention(filename, ".scene")) throw IOException("Scene.cpp", "23", "loadScene", filename, "NO_.SCENE", IOException::OPEN);
		m_numLights = m_numTriangles = 0;
		std::ifstream file(filename);
		std::string line;
		while (std::getline(file, line))
		{
			if (line[0] == '#') continue;
			if (line[0] == 'T') parseTriangle(line.c_str());
			if (line[0] == 'P') parsePlane(line.c_str());
			if (line[0] == 'L') parseLight(line.c_str());
			if (line[0] == 'S') parseSkybox(line.c_str());
		}

	}
	catch (IOException e)
	{
		printf("IOException in file \"%s\" while loading scene\n%s", filename, e.what());
	}
}

void Scene::parseTriangle(const char* line)
{
	// skip first T
	// vec3 v0, v1, v2;
	// for v0, v1, v2; x, y, z
		// until you see a comma interpret value
	// addTriangle(Triangle(v0, v1, v2));
}

void Scene::parsePlane(const char* line)
{
	// skip first P
	// vec3 v0, v1, v2, v3
	// for 4 loops
		// until you see comma, interpret value as vertex
	// addTriangle(Triangle(v0, v1, v2));
	// addTriangle(Triangle(v2, v3, v0));
}

// Parses a light definition (e.g. "L 0.0,0.0,0.0 0xFFFFFF")
void Scene::parseLight(const char* line)
{
	uint i = 1;
	while (line[i] == ' ' || line[i] == '\t') i++;
	if (line[i] == 0) throw IOException("Scene.cpp", "63", "parseLight", "", "END_OF_LINE", IOException::READ);
	// skip first L
	// vec3 p;
	// Pixel c;
	// read position
	// read color
	// addLight(Light(p, c));
}

// Parses a skybox definition (e.g. "S <path_to_skybox_image>")
void Scene::parseSkybox(const char* line)
{
	uint i = 1;
	while (line[i] == ' ' || line[i] == '\t') i++;
	if (line[i] == 0) throw IOException("Scene.cpp", "76", "parseSkybox", "", "EMPTY_SKYBOX_DEFINITION", IOException::READ);
	m_skybox = Skybox(line + i);
}

// Parses a vec3 definition (e.g. "(0.0,0.0,0.0)")
// char pointer should point at opening bracket
vec3 Scene::parseVec3(const char* line)
{
	vec3 vec = vec3();
	uint start = 1; // skip opening bracket
	for (uint i = 0; i < 3; i++)
	{
		uint end = start;
		uint iter = 0; // prevents endless loop when there's no end symbol
		while (line[end] != (i < 2 ? ',' : ')')) // while no end symbol has been found, keep incrementing float length
		{
			if (line[end] == 0) throw IOException("Scene.cpp", "96", "parseVec3", "", "END_OF_LINE", IOException::READ);
			end++;
			if (iter++ > 10) throw IOException("Scene.cpp", "98", "parseVec3", "", "NO_COMMA_FOUND", IOException::READ);
		}
		vec[i] = strtof(std::string(line, start, end - start).c_str(), 0);
		start = end + 1;
	}
	return vec;
}

// Parses a Color definition (e.g. "0xFF00FF")
// char pointer should point at leading 0
Color Scene::parseColor(const char* line)
{
	return std::stoul(line, 0, 16);
}

// DEBUG unit test
void Scene::testParsers()
{
	const char* vector = "(0.1,0.2,0.3)";
	printf("Testing vec3 definition \"%s\"\n", vector);
	vec3 vec = parseVec3(vector);
	printf("Result: %.2f, %.2f, %.2f\n", vec.x, vec.y, vec.z);
	const char* color = "0x00FF00";
	printf("Testing color definition \"%s\"\n", color);
	Color col = parseColor(color);
	printf("Result: %X\n", col);
}

// DEBUG unit test
void Scene::printTriangles()
{
	for (uint i = 0; i < m_numTriangles; i++)
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
	for (uint i = 0; i < m_numLights; i++)
	{
		printf("L %.1f,%.1f,%.1f %X\n",
			m_lights[i].pos.x, m_lights[i].pos.y, m_lights[i].pos.z,
			m_lights[i].col);
	}
}

} // namespace Engine