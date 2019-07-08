#include "headers.h"

namespace JornamEngine {

/*
	Checks if the given file is already in memory and returns its idx if so

	@param filename	The path to the image file
	@returns		The texture idx if present, 0 otherwise
*/
uint TextureMap::get(const char* filename)
{
	size_t size = m_hashes.size();
	for (size_t i = 0; i < size; i++)
	{
		if (m_hashes[i] == filename) return (uint) i;
	}
	return 0;
}

/*
	Adds a solid color texture to the hashmap

	@param filename A string that identifies the color
	@param color	The solid color
	@returns		The new texture idx
*/
uint TextureMap::add(const char* filename, Color color)
{
	m_hashes.push_back(filename);
	m_textures->push_back(Texture(color));
	return (uint)m_hashes.size() - 1;
}

/*
	Loads a new texture and adds it to the hashmap

	@param filename	The path to the image file
	@param onDevice	Boolean indicating the texture context (host/device)
	@returns		The new texture idx
*/
uint TextureMap::add(const char* filename, bool onDevice)
{
	m_hashes.push_back(filename);
	m_textures->push_back(Texture(filename, onDevice));
	return (uint)m_hashes.size() - 1;
}

/*
	Loads the texture on local memory

	@param filename	The path to the image file
*/
void Texture::makeHostPtr(const char* filename)
{
	readTexture(filename);
}

/*
	Loads the texture locally, copies it to the GPU, and frees the local memory

	@param filename	The path to the image file
*/
void Texture::makeDevicePtr(const char* filename)
{
	readTexture(filename);
	Color* c_buffer;
	cudaMalloc(&c_buffer, m_width * m_height * sizeof(Color));
	cudaMemcpy(c_buffer, m_buffer, m_width * m_height * sizeof(Color), cudaMemcpyHostToDevice);
	//free(m_buffer);
	m_buffer = c_buffer;
}

/*
	Reads an image file from memory and decrypts it into m_buffer

	@param filename	The path to the image file
*/
void Texture::readTexture(const char* filename)
{
	if (!fopen(filename, "rb"))
		logger.logDebug("Texture", ("The given file \"" +
			std::string(filename) + "\" could not be found.").c_str(),
			JornamException::ERR);
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(filename, 0);
	if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename(filename);
	FIBITMAP* tmp = FreeImage_Load(fif, filename);
	FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
	FreeImage_Unload(tmp);
	m_width = FreeImage_GetWidth(dib);
	m_height = FreeImage_GetHeight(dib);
	m_buffer = (Color*)_aligned_malloc(m_width * m_height * sizeof(Color), 64);
	for (uint y = 0; y < m_height; y++)
	{
		unsigned const char *line = FreeImage_GetScanLine(dib, m_height - 1 - y);
		memcpy(m_buffer + y * m_width, line, m_width * sizeof(Color));
	}
	FreeImage_Unload(dib);
}

} // namespace Engine