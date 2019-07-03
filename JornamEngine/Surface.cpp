#include "headers.h"

namespace JornamEngine {

Surface::Surface(uint a_width, uint a_height, Color* a_buffer, uint a_pitch) :
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
	m_buffer = (Color*)_aligned_malloc(a_width * a_height * sizeof(Color), 64);
	m_owner = true; // buffer was allocated by the instance
}

Surface::Surface(const char* a_filename)
{
	loadImage(a_filename);
}

void Surface::Clear(Color a_color)
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = a_color;
}

void Surface::Clear()
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = 0;
}

void Surface::Plot(uint x, uint y, Color p)
{
	if ((x < m_width) && (y < m_height)) m_buffer[x + y * m_pitch] = p;
}

void Surface::loadImage(const char* a_filename)
{
	if (!fopen(a_filename, "rb"))
		logger.logDebug("Surface", ("The given file \"" +
			std::string(a_filename) + "\" could not be found.\n").c_str(),
			JornamException::ERR);
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(a_filename, 0);
	if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename(a_filename);
	FIBITMAP* tmp = FreeImage_Load(fif, a_filename);
	FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
	FreeImage_Unload(tmp);
	m_width = m_pitch = FreeImage_GetWidth(dib);
	m_height = FreeImage_GetHeight(dib);
	m_buffer = (Color*)_aligned_malloc(m_width * m_height * sizeof(Color), 64);
	m_owner = true;
	for (uint y = 0; y < m_height; y++)
	{
		unsigned const char *line = FreeImage_GetScanLine(dib, m_height - 1 - y);
		memcpy(m_buffer + y * m_pitch, line, m_width * sizeof(Color));
	}
	FreeImage_Unload(dib);
}

} // namespace JornamEngine
