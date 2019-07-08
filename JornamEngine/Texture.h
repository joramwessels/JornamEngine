#pragma once

namespace JornamEngine {

/*
	Represents a texture, containing a pointer to the pixel buffer
	If both the width and height equal 1, the Color pointer represents a color

	@param filename	The path to the image file
*/
class Texture
{
public:
	Texture() { m_buffer = NULL; }
	Texture(Color color) : m_width(0), m_height(0), m_color(color.hex) {}
	Texture(const char* filename, bool onDevice)
	{
		if (!onDevice) makeHostPtr(filename);
		if (onDevice) makeDevicePtr(filename);
	}
	~Texture() { if (!isSolidColor()) { if (m_buffer) freeHostPtr(); else freeDevicePtr(); } };
	const Color* getBuffer() { return m_buffer; }
	const Color getColor() { return m_color; }
	uint getWidth() { return m_width; }
	uint getHeight() { return m_height; }
	bool isSolidColor() { return (m_width <= 1 && m_height <= 1); }
protected:
	union { Color* m_buffer; long m_color; };
	uint m_width, m_height;

	void makeHostPtr(const char* filename);
	void makeDevicePtr(const char* filename);
	void freeHostPtr() { };//free(m_buffer); }
	void freeDevicePtr() { cudaFree(m_buffer); }
	void readTexture(const char* filename);
};

/*
	Prevents Texture duplicates by hashing the filenames
	The add function takes care of keeping the GPU and host in sync

	@param textures	A pointer to the textures vector
*/
class TextureMap
{
public:
	TextureMap(std::vector<Texture>* textures) : m_textures(textures)
	{
		m_hashes.push_back("NULL HASH");
		m_textures->push_back(Texture());
	}
	uint get(const char* filename);
	uint add(const char* filename, Color color);
	uint add(const char* filename, bool onDevice);
protected:
	std::vector<std::string> m_hashes; // starts at index 1
	std::vector<Texture>* m_textures;  // starts at index 1
};

} // namespace Engine