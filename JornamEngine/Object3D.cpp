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

} // namespace Engine