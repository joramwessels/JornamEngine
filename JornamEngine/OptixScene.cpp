/**
file:			Scene.cpp
last modified:	18-02-2019
description:	Provides a Scene object that holds the triangles, lights,
and skybox. Scenes can be loaded from custom .scene files
using the following syntax:

Memory allocations (should be at the start of the .scene file):
D 30 500

Triangle (provide vertices in clockwise order):
T (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) 0x00FFFFFF

Plane (provide vertices in clockwise order):
P (v0.x,v0.y,v0.z) (v1.x,v1.y,v1.z) (v2.x,v2.y,v2.z) (v3.x,v3.y,v3.z) 0x00FFFFFF

Object
O filename.obj (x, y, z) (ox, oy, oz) material_id

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
	void OptixScene::loadScene(const char* a_filename, Camera* a_camera)
	{
		uint line_no = 0;
		if (!filenameHasExtention(a_filename, ".scene"))
			logDebug("Scene",
				"The scene you're trying to load doesn't have the .scene extention.\n",
				JornamException::ERR);
		try {
			m_numLights = m_numObjects = 0;
			std::ifstream file(a_filename);
			std::string line;
			while (std::getline(file, line))
			{
				if (line[0] == '#' || line[0] == 0) continue;
				else if (line[0] == 'D') parseDimensions(line.c_str());
				//else if (line[0] == 'T') parseTriangle(line.c_str());
				//else if (line[0] == 'P') parsePlane(line.c_str());
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

	// reallocates the Light and Triangle pointers given the new dimensions
	void OptixScene::resetDimensions(uint ls, uint os)
	{
		delete m_lights;
		m_lights = (ls > 0 ? (Light*)malloc(ls * sizeof(Light)) : 0);
		if (m_objects != 0) rtGeometryGroupDestroy(m_objects);
		if (os > 0) rtGeometryGroupSetChildCount(m_objects, os);
	}

	// Parses a dimension definition
	void OptixScene::parseDimensions(const char* line)
	{
		uint i, skip = 1; // skip 'D' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		uint ld = std::stoul(line + i);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		uint od = std::stoul(line + i);

		resetDimensions(ld, od);
	}

	//// Parses a triangle definition
	//void OptixScene::parseTriangle(const char* line)
	//{
	//	uint i, skip = 1; // skip 'T' identifier
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v0 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v1 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v2 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	Color col = parseColor(line, i);
	//
	//	addTriangle(Triangle(v0, v1, v2, col));
	//}

	//// Parses a plane definition
	//void OptixScene::parsePlane(const char* line)
	//{
	//	uint i, skip = 1; // skip 'P' identifier
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v0 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v1 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v2 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	vec3 v3 = parseVec3(line, i);
	//
	//	i = skipWhiteSpace(line, skip);
	//	skip = skipExpression(line, i);
	//	Color col = parseColor(line, i);
	//
	//	addTriangle(Triangle(v0, v1, v2, col));
	//	addTriangle(Triangle(v2, v3, v0, col));
	//}

	// Parses an object
	void OptixScene::parseObject(const char* line)
	{
		uint i, skip = 1; // skip 'O' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		std::string filename = std::string(line).substr(i, skip);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		vec3 pos = parseVec3(line, i);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		vec3 ori = parseVec3(line, i);

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		uint material = std::stoul(line + i, 0, 10);

		addObject(filename.c_str(), pos, ori, material);
	}

	// Adds a new object to the GeometryGroup
	void OptixScene::addObject(const char* filename, vec3 pos, vec3 ori, uint material) // TODO pos, ori, material
	{
		// Reading .obj using tinyobjloader
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		tinyobj::LoadObj(shapes, materials, err, filename);
		if (!err.empty()) logDebug("Scene",
			(("Error reading object \"" + std::string(filename) + "\": ") + err).c_str(),
			JornamException::ERR);
	
		// Collecting vertex count
		int vertexCount = 0;
		for (std::vector<tinyobj::shape_t>::const_iterator it = shapes.begin(); it < shapes.end(); ++it)
		{
			const tinyobj::shape_t & shape = *it;
			vertexCount += static_cast<int32_t>(shape.mesh.positions.size()) / 3;
		}

		// Preparing vertex buffer
		RTbuffer vertices;
		rtBufferCreate(m_context, RT_BUFFER_INPUT, &vertices);
		rtBufferSetFormat(vertices, RT_FORMAT_USER);
		rtBufferSetElementSize(vertices, sizeof(float));
		rtBufferSetSize1D(vertices, vertexCount);
		void* bufferptr;
		rtBufferMap(vertices, &bufferptr);
		float* verticesptr = (float*)bufferptr;
		int verticesIndex = 0;

		// Copying vertices to buffer
		for (std::vector<tinyobj::shape_t>::const_iterator it = shapes.begin(); it < shapes.end(); ++it)
		{
			const tinyobj::shape_t & shape = *it;
			int shapeVertexCount = shape.mesh.positions.size();
			memcpy(&verticesptr[verticesIndex], shape.mesh.positions.data(), shapeVertexCount * sizeof(float));
			verticesIndex += shapeVertexCount;
		}
		rtBufferUnmap(vertices);

		// Initializing rtGeometry
		RTgeometrytriangles mesh;
		rtGeometryTrianglesCreate(m_context, &mesh);
		rtGeometryTrianglesSetVertices(mesh, vertexCount, vertices, 0, 3 * sizeof(float), RT_FORMAT_FLOAT3); // TODO is the stride right?
	}

	// Parses a light definition
	void OptixScene::parseLight(const char* line)
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
	void OptixScene::parseSkybox(const char* line)
	{
		uint i, skip = 1; // skip 'S' identifier

		i = skipWhiteSpace(line, skip);
		skip = skipExpression(line, i);
		m_skybox = Skybox(line + i);
	}

	// Parses a camera location and rotation
	void OptixScene::parseCamera(const char* line, Camera* camera)
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
	vec3 OptixScene::parseVec3(const char* line, uint col)
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
	Color OptixScene::parseColor(const char* line, uint col)
	{
		return std::stoul(line + col, 0, 16);
	}

	// Returns the index of the first character after the white space.
	// Throws an IOException if there are no whitespace characters at the given index.
	uint OptixScene::skipWhiteSpace(const char* line, uint col)
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
	uint OptixScene::skipExpression(const char* line, uint col)
	{
		uint i = col;
		if (line[i] == 0) logDebug("Scene",
			("Expression expected at index " + std::to_string(i)).c_str(),
			JornamException::ERR);
		while (line[i] != ' ' && line[i] != '\t' && line[i] != 0) i++;
		return i;
	}

} // namespace Engine