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
	FILE* file = fopen(a_filename, "rb");
	if (!file) printf("File \" %s \" not found.\n", a_filename);
	else loadImage();
}

Surface::~Surface()
{
	// only free the buffer if it was allocated by the instance
	if (m_owner) _aligned_free(m_buffer);
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

void Surface::loadImage()
{
	//FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	//fif = FreeImage_GetFileType(a_File, 0);
	//if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename(a_File);
	//FIBITMAP* tmp = FreeImage_Load(fif, a_File);
	//FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
	//FreeImage_Unload(tmp);
	//m_Width = m_Pitch = FreeImage_GetWidth(dib);
	//m_Height = FreeImage_GetHeight(dib);
	//m_Buffer = (Pixel*)MALLOC64(m_Width * m_Height * sizeof(Pixel));
	//m_Flags = OWNER;
	//for (int y = 0; y < m_Height; y++)
	//{
	//	unsigned const char *line = FreeImage_GetScanLine(dib, m_Height - 1 - y);
	//	memcpy(m_Buffer + y * m_Pitch, line, m_Width * sizeof(Pixel));
	//}
	//FreeImage_Unload(dib);
}

} // namespace JornamEngine
