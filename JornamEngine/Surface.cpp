#include "headers.h"

namespace JornamEngine {

/*
	Clears the pixel buffer

	@param color	The color each pixel will get
*/
void Surface::Clear(Color a_color)
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = a_color;
}

/*
	Clears the pixel buffer (black)
*/
void Surface::Clear()
{
	int size = m_width * m_height;
	for (int i = 0; i < size; i++) m_buffer[i] = 0;
}

/*
	Plots a single color value in the pixel buffer

	@param x	The x-coordinate
	@param y	The y-coordinate
	@param p	The color of the pixel
*/
void Surface::Plot(uint x, uint y, Color p)
{
	if ((x < m_width) && (y < m_height)) m_buffer[x + y * m_pitch] = p;
}

/*
	Loads an image into the buffer using FreeImage

	@param filename	The path to the image file
*/
void Surface::loadImage(const char* a_filename)
{
	if (!fopen(a_filename, "rb"))
		logger.logDebug("Surface", ("The given file \"" +
			std::string(a_filename) + "\" could not be found.").c_str(),
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
