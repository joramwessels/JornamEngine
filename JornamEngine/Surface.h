#pragma once

namespace JornamEngine {

// A grid of Colors
class Surface
{
public:
	Surface(uint width, uint height, Color* buffer, uint pitch);
	Surface(uint width, uint height);
	Surface(const char* file);
	~Surface();

	void loadImage(const char* filename);
	void Plot(uint x, uint y, Color p);
	void Clear(Color p);
	void Clear();

	inline int GetWidth() { return m_width; };
	inline int GetHeight() { return m_height; };
	inline Color* GetBuffer() { return m_buffer; };
	inline Color GetPixel(uint x, uint y) const { if (x < m_width && y < m_height) return m_buffer[x + y * m_pitch]; }

private:
	Color* m_buffer;
	uint m_width, m_height, m_pitch;
	bool m_owner;
};

} // namespace JornamEngine
