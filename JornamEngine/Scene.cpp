#include "headers.h"
// NOTE: .scene parsing formats
//
// Memory allocations (should be at the start of the .scene file)
// D 30 500
//
// Triangle (provide vertices in clockwise order)
// T (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) 0x00FFFFFF
// 
// Plane (provide vertices in clockwise order)
// P (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) (v3.x,v3.y,v3.z) 0x00FFFFFF
//
// Light
// L (p.x,p.y,p.z) color(hex)
//
// Skybox
// S <filepath>
//
// Camera
// C (pos.x, pos.y, pos.z) (dir.x, dir.y, dir.z) (left.x, left.y, left.z)

namespace JornamEngine {

// Loads a scene from a .scene file
void Scene::loadScene(const char* a_filename, Camera* a_camera)
{
	uint line_no = 0;
	if (!filenameHasExtention(a_filename, ".scene"))
		throw JornamEngineException("Scene",
			"The scene you're trying to load doesn't have the .scene extention.\n",
			JornamEngineException::ERR);
	//try {
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
			else if (line[0] == 'C') parseCamera(line.c_str(), a_camera);
			else throw JornamEngineException("Scene",
				"Undefined parse descriptor \"" + std::to_string(line[0]) + "\" encountered",
				JornamEngineException::ERR);
			line_no++;
		}
	//}
	//catch (JornamEngineException e)
	//{
	//	e.m_msg = e.m_msg + " in line " + std::to_string(line_no) + " of file \"" + a_filename + "\".\n";
	//	throw e;
	//}
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
	uint i, skip = 1; // skip 'D' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	uint ld = std::stoul(line + i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	uint td = std::stoul(line + i);

	resetDimensions(ld, td);
}

// Parses a triangle definition
void Scene::parseTriangle(const char* line)
{
	uint i, skip = 1; // skip 'T' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v0 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v1 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v2 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	Color col = parseColor(line, i);

	addTriangle(Triangle(v0, v1, v2, col));
}

// Parses a plane definition
void Scene::parsePlane(const char* line)
{
	uint i, skip = 1; // skip 'P' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v0 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v1 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v2 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 v3 = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	Color col = parseColor(line, i);

	addTriangle(Triangle(v0, v1, v2, col));
	addTriangle(Triangle(v2, v3, v0, col));
}

// Parses a light definition
void Scene::parseLight(const char* line)
{
	uint i, skip = 1; // skip 'L' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 pos = parseVec3(line, i);
	
	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	Color col = parseColor(line, i);

	addLight(Light(pos, col));
}

// Parses a skybox definition
void Scene::parseSkybox(const char* line)
{
	uint i, skip = 1; // skip 'S' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	m_skybox = Skybox(line + i);
}

// Parses a camera location and rotation
void Scene::parseCamera(const char* line, Camera* camera)
{
	if (camera == 0) throw JornamEngineException(
		"Scene", "No camera pointer provided",
		JornamEngineException::ERR);

	uint i, skip = 1; // skip 'C' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 pos = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 dir = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 lft = parseVec3(line, i);

	camera->setLocation(pos);
	if (lft.isNonZero()) camera->setRotation(dir, lft);
	else camera->setRotation(dir);
}

// Parses a vec3 definition (e.g. "(0.0,0.0,0.0)")
// char pointer should point at opening bracket
vec3 Scene::parseVec3(const char* line, uint col)
{
	if (line[col] != '(') throw JornamEngineException(
		"Scene", "Opening bracket expected at index " + std::to_string(col),
		JornamEngineException::ERR);

	vec3 vec = vec3();
	uint start = col + 1; // skip opening bracket
	for (uint i = 0; i < 3; i++)
	{
		uint end = start;
		uint iter = 0; // prevents endless loop when there's no end symbol
		while (line[end] != (i < 2 ? ',' : ')')) // while no end symbol has been found, keep incrementing float length
		{
			if (line[end] == 0) throw JornamEngineException("Scene",
				"Vec3 definition interrupted at index " + std::to_string(i),
				JornamEngineException::ERR);
			end++;
			if (iter++ > 10) throw JornamEngineException("Scene",
				"Missing comma before index " + std::to_string(i),
				JornamEngineException::ERR);
		}
		vec[i] = strtof(std::string(line, start, end - start).c_str(), 0);
		start = end + 1;
	}
	return vec;
}

// Parses a Color definition (e.g. "0xFF00FF")
// char pointer should point at leading 0
Color Scene::parseColor(const char* line, uint col)
{
	return std::stoul(line + col, 0, 16);
}

// Returns the index of the first character after the white space.
// Throws an IOException if there are no whitespace characters at the given index.
uint Scene::skipWhiteSpace(const char* line, uint col)
{
	uint i = col;
	if (line[i] == 0) throw JornamEngineException("Scene",
		"White space expected at index " + std::to_string(i),
		JornamEngineException::ERR);
	while (line[i] == ' ' || line[i] == '\t') i++;
	return i;
}

// Returns the index of the first whitespace character after the expression.
// Throws an IOException if there is no expression at the given index.
uint Scene::skipExpression(const char* line, uint col)
{
	uint i = col;
	if (line[i] == 0) throw JornamEngineException("Scene",
		"Expression expected at index " + std::to_string(i),
		JornamEngineException::ERR);
	while (line[i] != ' ' && line[i] != '\t' && line[i] != 0) i++;
	return i;
}

} // namespace Engine