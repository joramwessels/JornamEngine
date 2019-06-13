#include "headers.h"

namespace JornamEngine {

/*
	Assigns the triangles to the Prime model

	@param indices	A vector with 3 consequtive vertex indices per triangle
	@param vertices	A vector with 3 consequtive floats per vertex
	@param type		The Prime buffer type
*/
void Object3D::setTriangles(std::vector<uint> indices, std::vector<float> vertices, RTPbuffertype type)
{
	m_primeHandle->setTriangles(
		indices.size() / 3, type, indices.data(),
		vertices.size() / 3, type, vertices.data()
	);
	m_primeHandle->update(0);
}

/*
	Interpolates triangle surface normal given Barycentric coordinates

	@param trIdx	The triangle index
	@param u		The Barycentric u coordinate
	@param v		The barycentric v coordinate
	@return			The surface normal at the given coordinates
*/
vec3 Object3D::interpolateNormal(uint t, float u, float v)
{
	uint v0 = m_indices[t * 3], v1 = m_indices[t * 3 + 1], v2 = m_indices[t * 3 + 2];
	vec3 n0 = m_normals[v0], n1 = m_normals[v1], n2 = m_normals[v2];
	return (n0 * u + n1 * v + n2 * (1 - u - v)).normalized();
}

} // namespace Engine