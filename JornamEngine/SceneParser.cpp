/**
file:			SceneParser.cpp
last modified:	09-07-2019
description:	Parses .scene files using the following syntax:

	Memory allocations (should be at the start of the .scene file):
	D 30 500

	Triangle (provide vertices in clockwise order):
	T (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) 0x00FFFFFF

	Plane (provide vertices in clockwise order):
	P (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) (v3.x,v3.y,v3.z) 0x00FFFFFF

	Object
	O meshFile.obj (axis.x, axis.y, axis.z) angle (x, y, z) (scale.x, scale.y, scale.z) textureFile.jpg/png

	Light:
	L (p.x,p.y,p.z) 0x00FFFFFF

	Skybox:
	S <filepath>

	Camera:
	C (pos.x, pos.y, pos.z) (dir.x, dir.y, dir.z) (left.x, left.y, left.z)

@author Joram Wessels
@version 0.1
*/
#include "headers.h"

namespace JornamEngine {

/*
	Loads a scene from a .scene file.

	@param filename the name of the .scene file
	@param camera (optional) a pointer to the camera object;
	only required when the .scene file configures the camera
	@throws JornamException when there was an issue with the given file
*/
void SceneParser::parseScene(const char* a_filename, Camera* a_camera)
{
	uint line_no = 0;
	if (!filenameHasExtention(a_filename, ".scene"))
		logger.logDebug("Scene",
			"The scene you're trying to load doesn't have the .scene extention.",
			JornamException::ERR);
	try
	{
		std::ifstream file(a_filename);
		std::string line;
		while (std::getline(file, line))
		{
			if (line[0] == '#' || line[0] == 0) { line_no++; continue; }
			else if (line[0] == 'D') parseDimensions(line.c_str());
			else if (line[0] == 'O') parseObject(line.c_str());
			else if (line[0] == 'L') parseLight(line.c_str());
			else if (line[0] == 'S') parseSkybox(line.c_str());
			else if (line[0] == 'C') parseCamera(line.c_str(), a_camera);
			else logger.logDebug("Scene",
				("Undefined parse descriptor \"" + std::to_string(line[0]) + "\" encountered").c_str(),
				JornamException::ERR);
			line_no++;
		}
	}
	catch (JornamException e)
	{
		e.m_msg = e.m_msg + " in line " + std::to_string(line_no) + " of file \"" + a_filename + "\".";
		logger.logDebug(e.m_class.c_str(), e.m_msg.c_str(), e.m_severity);
	}
}

/*
	Parses a dimension definition

	@param line	The char pointer to the line
*/
uint2 SceneParser::parseDimensions(const char* line)
{
	uint i, skip = 1; // skip 'D' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	uint ld = std::stoul(line + i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	uint od = std::stoul(line + i);

	uint2 u2 = uint2();
	u2.x = ld; u2.y = od;

	return u2;
}

/*
	Parses an object definition
	Adds meshes and textures to memory when necessary

	@param line	The char pointer to the line
*/
void SceneParser::parseObject(const char* line)
{
	uint i, skip = 1; // skip 'O' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	std::string meshFile = std::string(line).substr(i, skip-i);
	uint meshIdx = m_scene->addMesh(meshFile.c_str());

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 axis = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	float angle = parseFloat(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 pos = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 scale = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	uint textureIdx;
	if (line[i] == *"0") // can parse color instead of texture
	{
		Color color = parseColor(line, i);
		textureIdx = m_scene->addTexture(NULL, color);
	}
	else
	{
		std::string textureFile = std::string(line).substr(i, skip - i);
		textureIdx = m_scene->addTexture(textureFile.c_str());
	}

	m_scene->addObject(meshIdx, textureIdx, Transform(axis, angle, pos, scale));
}

/*
	Parses a light definition

	@param line	The char pointer to the line
*/
void SceneParser::parseLight(const char* line)
{
	uint i, skip = 1; // skip 'L' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	vec3 pos = parseVec3(line, i);

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	Color col = parseColor(line, i);

	m_scene->addLight(Light(pos, col));
}

/*
	Parses a skybox definition

	@param line	The char pointer to the line
*/
void SceneParser::parseSkybox(const char* line)
{
	uint i, skip = 1; // skip 'S' identifier

	i = skipWhiteSpace(line, skip);
	skip = skipExpression(line, i);
	m_scene->setSkybox(Skybox(line + i));
}

/*
	Parses a camera location and rotation

	@param line		The char pointer to the line
	@param camera	A pointer to the camera object
*/
void SceneParser::parseCamera(const char* line, Camera* camera)
{
	if (camera == 0) throw JornamException(
		"Scene", "No camera pointer provided",
		JornamException::ERR);

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

/*
	Parses a vec3 definition (e.g. "(0.0,0.0,0.0)")
	col index should point at opening bracket

	@param line	The char pointer to the line
	@param col	The char index at which to start parsing
*/
vec3 SceneParser::parseVec3(const char* line, uint col)
{
	if (line[col] != '(') throw JornamException(
		"Scene", ("Opening bracket expected at index " + std::to_string(col)).c_str(),
		JornamException::ERR);

	vec3 vec = vec3();
	uint start = col + 1; // skip opening bracket
	for (uint i = 0; i < 3; i++)
	{
		uint end = start;
		uint iter = 0; // prevents endless loop when there's no end symbol
		while (line[end] != (i < 2 ? ',' : ')')) // while no end symbol has been found, keep incrementing float length
		{
			if (line[end] == 0) throw JornamException("Scene",
				("Vec3 definition interrupted at index " + std::to_string(i)).c_str(),
				JornamException::ERR);
			end++;
			if (iter++ > 10) throw JornamException("Scene",
				("Missing comma before index " + std::to_string(i)).c_str(),
				JornamException::ERR);
		}
		vec[i] = strtof(std::string(line, start, end - start).c_str(), 0);
		start = end + 1;
	}
	return vec;
}

/*
	Parses a Color definition (e.g. "0xFF00FF")
	col index should point at leading 0

	@param line	The char pointer to the line
	@param col	The char index at which to start parsing
*/
Color SceneParser::parseColor(const char* line, uint col)
{
	return std::stoul(line + col, 0, 16);
}

/*
	Parses a float definition
	col index should point at first digit

	@param line	The char pointer to the line
	@param col	The char index at which to start parsing
*/
float SceneParser::parseFloat(const char* line, uint col)
{
	return std::stof(line + col, 0);
}

/*
	Returns the index of the first character after the whitespace

	@param line				The char pointer to the line
	@param col				The char index pointing at the first whitespace char
	@returns				The index of the first character after the white space
	@throws	JornamException	When there are no whitespace char at the given index
*/
uint SceneParser::skipWhiteSpace(const char* line, uint col)
{
	uint i = col;
	if (line[i] == 0) throw JornamException("Scene",
		("White space expected at index " + std::to_string(i)).c_str(),
		JornamException::ERR);
	while (line[i] == ' ' || line[i] == '\t') i++;
	return i;
}

/*
	Returns the index of the first whitespace char after the expression

	@param line				The char pointer to the line
	@param col				The char index pointing at the first index of the expression
	@returns				The index of the first whitespace char after the expression
	@throws	JornamException	When there is no expression at the given index
*/
uint SceneParser::skipExpression(const char* line, uint col)
{
	uint i = col;
	if (line[i] == 0) throw JornamException("Scene",
		("Expression expected at index " + std::to_string(i)).c_str(),
		JornamException::ERR);
	while (line[i] != ' ' && line[i] != '\t' && line[i] != 0) i++;
	return i;
}

} // namespace Engine