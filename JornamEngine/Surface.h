#pragma once

namespace JornamEngine {

// A grid of Colors
class Surface
{
public:
	Surface::Surface(uint a_width, uint a_height, Color* a_buffer, uint a_pitch) :
		m_width(a_width), m_height(a_height), m_buffer(a_buffer), m_pitch(a_pitch)
	{
		m_owner = false; // buffer was passed by reference
	}
	Surface::Surface(uint a_width, uint a_height) :
		m_width(a_width), m_height(a_height), m_pitch(a_width)
	{
		m_buffer = (Color*)_aligned_malloc(a_width * a_height * sizeof(Color), 64);
		m_owner = true; // buffer was allocated by the instance
	}
	Surface::Surface(const char* a_filename)
	{
		loadImage(a_filename);
	}
	~Surface() { if (m_owner) _aligned_free(m_buffer); }

	void loadImage(const char* filename);
	void Plot(uint x, uint y, Color p);
	void Plot(Color* buffer) { if (m_owner) _aligned_free(m_buffer); m_buffer = buffer; m_owner = false; }
	void Clear(Color p);
	void Clear();

	inline int GetWidth() { return m_width; }
	inline int GetHeight() { return m_height; }
	inline Color* GetBuffer() { return m_buffer; }
	inline Color GetPixel(uint x, uint y) const { if (x < m_width && y < m_height) return m_buffer[x + y * m_pitch]; else return 0x0; }

protected:
	Color* m_buffer;
	uint m_width, m_height, m_pitch;
	bool m_owner;
};

} // namespace JornamEngine
