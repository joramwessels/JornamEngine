#include "headers.h"
// NOTE: .scene parsing formats
//
// Memory allocations (should be at the start of the .scene file)
// D 30 500
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

namespace JornamEngine {

// Loads a scene from a .scene file
void Scene::loadScene(const char* a_filename)
{
	try
	{
		if (!filenameHasExtention(a_filename, ".scene")) throw IOException("Scene.cpp", "23", "loadScene", a_filename, "NO_.SCENE", IOException::OPEN);
		m_numLights = m_numTriangles = 0;
		std::ifstream file(a_filename);
		std::string line;
		while (std::getline(file, line))
		{
			if (line[0] == '#' || line[0] == 0) continue;
			else if (line[0] == 'D') parseDimensions(line.c_str());
			else if (line[0] == 'T') parseTriangle(line.c_str());
			else if (line[0] == 'P') parsePlane(line.c_str());
			else if (line[0] == 'L') parseLight(line.c_str());
			else if (line[0] == 'S') parseSkybox(line.c_str());
			else throw IOException("Scene.cpp", "34", "loadScene", a_filename, "UNDEFINED_DESCRIPTOR", IOException::READ);
		}
	}
	catch (IOException e)
	{
		printf("IOException in file \"%s\" while loading scene\n\t%s", a_filename, e.what());
	}
}

// reallocates the Light and Triangle pointers given the new dimensions
void Scene::resetDimensions(uint ls, uint ts)
{
	delete m_lights;
	m_lights = (ls > 0 ? (Light*)malloc(ls * sizeof(Light)) : 0);
	delete m_triangles;
	m_triangles = (ts > 0 ? (Triangle*)malloc(ts * sizeof(Triangle)) : 0);
}

// Parses a dimension definition
void Scene::parseDimensions(const char* line)
{
	uint i = skipWhiteSpace(line, 1); // skip 'D' identifier and leading whitespace
	uint ld = std::stoul(line + i);
	i = skipExpression(line, i);
	i = skipWhiteSpace(line, i);
	uint td = std::stoul(line + i);
	resetDimensions(ld, td);
}

// Parses a triangle definition
void Scene::parseTriangle(const char* line)
{
	uint i = skipWhiteSpace(line, 1); // skip 'T' identifier and leading whitespace
	vec3 v0 = parseVec3(line + i);
	i = skipExpression(line, i);

	i = skipWhiteSpace(line, i);
	vec3 v1 = parseVec3(line + i);
	i = skipExpression(line, i);

	i = skipWhiteSpace(line, i);
	vec3 v2 = parseVec3(line + i);

	addTriangle(Triangle(v0, v1, v2));
}

// Parses a plane definition
void Scene::parsePlane(const char* line)
{
	uint i = skipWhiteSpace(line, 1); // skip 'P' identifier and leading whitespace
	vec3 v0 = parseVec3(line + i);
	i = skipExpression(line, i);

	i = skipWhiteSpace(line, i);
	vec3 v1 = parseVec3(line + i);
	i = skipExpression(line, i);

	i = skipWhiteSpace(line, i);
	vec3 v2 = parseVec3(line + i);
	i = skipExpression(line, i);

	i = skipWhiteSpace(line, i);
	vec3 v3 = parseVec3(line + i);

	addTriangle(Triangle(v0, v1, v2));
	addTriangle(Triangle(v2, v3, v0));
}

// Parses a light definition (e.g. "L 0.0,0.0,0.0 0xFFFFFF")
void Scene::parseLight(const char* line)
{
	uint i = skipWhiteSpace(line, 1); // skip 'L' identifier and leading whitespace
	vec3 pos = parseVec3(line + i);
	i = skipExpression(line, i);
	i = skipWhiteSpace(line, i);
	Color col = parseColor(line + i);
	addLight(Light(pos, col));
}

// Parses a skybox definition (e.g. "S <path_to_skybox_image>")
void Scene::parseSkybox(const char* line)
{
	uint i = skipWhiteSpace(line, 1); // skip 'S' identifier and leading whitespace
	m_skybox = Skybox(line + i);
}

// Parses a vec3 definition (e.g. "(0.0,0.0,0.0)")
// char pointer should point at opening bracket
vec3 Scene::parseVec3(const char* line)
{
	vec3 vec = vec3();
	if (line[0] != '(') throw IOException("Scene.cpp", "117", "parseVec3", "", "NO_BRACKET", IOException::READ);
	uint start = 1; // skip opening bracket
	for (uint i = 0; i < 3; i++)
	{
		uint end = start;
		uint iter = 0; // prevents endless loop when there's no end symbol
		while (line[end] != (i < 2 ? ',' : ')')) // while no end symbol has been found, keep incrementing float length
		{
			if (line[end] == 0) throw IOException("Scene.cpp", "111", "parseVec3", "", "END_OF_LINE", IOException::READ);
			end++;
			if (iter++ > 10) throw IOException("Scene.cpp", "114", "parseVec3", "", "NO_COMMA_FOUND", IOException::READ);
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

// Returns the index of the first character after the white space.
// Throws an IOException if there are no characters after the white space.
uint Scene::skipWhiteSpace(const char* line, uint col)
{
	uint i = col;
	while (line[i] == ' ' || line[i] == '\t') i++;
	if (line[i] == 0) throw IOException("Scene.cpp", "156", "skipWhiteSpace", "", "END_OF_LINE", IOException::READ);
	return i;
}

// Returns the index of the first whitespace character after the expression.
// Throws an IOException if there are no whitespace characters after the expression.
uint Scene::skipExpression(const char* line, uint col)
{
	uint i = col;
	while (line[i] != ' ' && line[i] != '\t' && line[i] != 0) i++;
	if (line[i] == 0) throw IOException("Scene.cpp", "154", "skipExpression", "", "END_OF_LINE", IOException::READ);
	return i;
}

} // namespace Engine