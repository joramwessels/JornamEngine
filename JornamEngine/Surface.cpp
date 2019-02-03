#include "headers.h"

namespace JornamEngine {

Surface::Surface(uint a_width, uint a_height, Pixel* a_buffer, uint a_pitch) :
	m_width(a_width),
	m_height(a_height),
	m_buffer(a_buffer),
	m_pitch(a_pitch)
{
	m_owner = false; // buffer was passed by reference
}

Surface::Surface(uint a_width, uint a_height) :
	m_width(a_width),
	m_height(a_height),
	m_pitch(a_width)
{
	m_buffer = (Pixel*)_aligned_malloc(a_width * a_height * sizeof(Pixel), 64);
	m_owner = true; // buffer was allocated by the instance
}

Surface::~Surface()
{
	// only free the buffer if it was allocated by the instance
	if (m_owner) _aligned_free(m_buffer);
}

void Surface::Clear(Pixel a_color)
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = a_color;
}

void Surface::Clear()
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = 0;
}

void Surface::Plot(uint x, uint y, Pixel p)
{
	if ((x < m_width) && (y < m_height)) m_buffer[x + y * m_pitch] = p;
}

} // namespace JornamEngine
