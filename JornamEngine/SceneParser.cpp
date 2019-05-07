/**
file:			SceneParser.cpp
last modified:	07-05-2019
description:	Parses .scene files using the following syntax:

	Memory allocations (should be at the start of the .scene file):
	D 30 500

	Triangle (provide vertices in clockwise order):
	T (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) 0x00FFFFFF

	Plane (provide vertices in clockwise order):
	P (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) (v3.x,v3.y,v3.z) 0x00FFFFFF

	Object
	O filename.obj (axis.x, axis.y, axis.z) angle (x, y, z) (scale.x, scale.y, scale.z) material_id

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

	/**
	Loads a scene from a .scene file.

	@param filename the name of the .scene file
	@param camera (optional) a pointer to the camera object;
	only required when the .scene file configures the camera
	@throws JornamException when there was an issue with the given file
	*/
	void SceneParser::loadScene(const char* a_filename, Camera* a_camera)
	{
		uint line_no = 0;
		if (!filenameHasExtention(a_filename, ".scene"))
			logDebug("Scene",
				"The scene you're trying to load doesn't have the .scene extention.\n",
				JornamException::ERR);
		try
		{
			m_scene.setLightCount(0);
			m_scene.setObjectCount(0);
			std::ifstream file(a_filename);
			std::string line;
			while (std::getline(file, line))
			{
				if (line[0] == '#' || line[0] == 0) continue;
				else if (line[0] == 'D') parseDimensions(line.c_str());
				else if (line[0] == 'T') parseTriangle(line.c_str());
				else if (line[0] == 'P') parsePlane(line.c_str());
				else if (line[0] == 'O') parseObject(line.c_str());
				else if (line[0] == 'L') parseLight(line.c_str());
				else if (line[0] == 'S') parseSkybox(line.c_str());
				else if (line[0] == 'C') parseCamera(line.c_str(), a_camera);
				else logDebug("Scene",
					("Undefined parse descriptor \"" + std::to_string(line[0]) + "\" encountered").c_str(),
					JornamException::ERR);
				line_no++;
			}
		}
		catch (JornamException e)
		{
			e.m_msg = e.m_msg + " in line " + std::to_string(line_no) + " of file \"" + a_filename + "\".\n";
			throw e;
		}
	}

	// Parses a dimension definition
	void SceneParser::parseDimensions(const char* line)
	{
		uint i, skip = 1; // skip 'D' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		uint ld = std::stoul(line + i);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		uint od = std::stoul(line + i);

		//resetDimensions(ld, od);
	}

	// Parses a triangle definition
	void SceneParser::parseTriangle(const char* line)
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

		std::vector<float> vertices({ v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z });
		std::vector<int> indices({ 0, 1, 2 });

		RTPmodel triangle;
		rtpModelCreate(m_scene.getContext(), &triangle);
		m_scene.addObject(triangle, vertices, indices, TransformMatrix(vec3(0.0f), 0.0f));
	}

	// Parses a plane definition
	void SceneParser::parsePlane(const char* line)
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

		std::vector<float> vertices({ v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z });
		std::vector<int> indices({ 0, 1, 2, 2, 3, 0 });

		RTPmodel plane;
		rtpModelCreate(m_scene.getContext(), &plane);
		m_scene.addObject(plane, vertices, indices, TransformMatrix(vec3(0.0f), 0.0f));
	}

	// Parses an object
	void SceneParser::parseObject(const char* line)
	{
		uint i, skip = 1; // skip 'O' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		std::string filename = std::string(line).substr(i, skip);


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
		uint material = std::stoul(line + i, 0, 10);

		vec3 scale = vec3(1.0);

		m_scene.readObject(filename.c_str(), TransformMatrix(axis, angle, pos, scale), material);
	}

	// Parses a light definition
	void SceneParser::parseLight(const char* line)
	{
		uint i, skip = 1; // skip 'L' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		vec3 pos = parseVec3(line, i);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		Color col = parseColor(line, i);

		m_scene.addLight(Light(pos, col));
	}

	// Parses a skybox definition
	void SceneParser::parseSkybox(const char* line)
	{
		uint i, skip = 1; // skip 'S' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		m_scene.setSkybox(Skybox(line + i));
	}

	// Parses a camera location and rotation
	void SceneParser::parseCamera(const char* line, Camera* camera)
	{
		if (camera == 0) logDebug(
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

	// Parses a vec3 definition (e.g. "(0.0,0.0,0.0)")
	// char pointer should point at opening bracket
	vec3 SceneParser::parseVec3(const char* line, uint col)
	{
		if (line[col] != '(') logDebug(
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
				if (line[end] == 0) logDebug("Scene",
					("Vec3 definition interrupted at index " + std::to_string(i)).c_str(),
					JornamException::ERR);
				end++;
				if (iter++ > 10) logDebug("Scene",
					("Missing comma before index " + std::to_string(i)).c_str(),
					JornamException::ERR);
			}
			vec[i] = strtof(std::string(line, start, end - start).c_str(), 0);
			start = end + 1;
		}
		return vec;
	}

	// Parses a Color definition (e.g. "0xFF00FF")
	// char pointer should point at leading 0
	Color SceneParser::parseColor(const char* line, uint col)
	{
		return std::stoul(line + col, 0, 16);
	}

	// Parses a float definition
	float SceneParser::parseFloat(const char* line, uint col)
	{
		return std::stof(line + col, 0);
	}

	// Returns the index of the first character after the white space.
	// Throws an IOException if there are no whitespace characters at the given index.
	uint SceneParser::skipWhiteSpace(const char* line, uint col)
	{
		uint i = col;
		if (line[i] == 0) logDebug("Scene",
			("White space expected at index " + std::to_string(i)).c_str(),
			JornamException::ERR);
		while (line[i] == ' ' || line[i] == '\t') i++;
		return i;
	}

	// Returns the index of the first whitespace character after the expression.
	// Throws an IOException if there is no expression at the given index.
	uint SceneParser::skipExpression(const char* line, uint col)
	{
		uint i = col;
		if (line[i] == 0) logDebug("Scene",
			("Expression expected at index " + std::to_string(i)).c_str(),
			JornamException::ERR);
		while (line[i] != ' ' && line[i] != '\t' && line[i] != 0) i++;
		return i;
	}
} // namespace Engine