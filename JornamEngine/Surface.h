#pragma once

namespace JornamEngine {

typedef unsigned int Pixel; // uint assumed to be 32-bit

class Surface
{
public:
	Surface(uint width, uint height, Pixel* buffer, uint pitch);
	Surface(uint width, uint height);
	~Surface();

	void Plot(uint x, uint y, Pixel p);
	void Clear(Pixel p);
	void Clear();

	int GetWidth() { return m_width; };
	int GetHeight() { return m_height; };
	Pixel* GetBuffer() { return m_buffer; };

private:
	Pixel* m_buffer;
	uint m_width, m_height, m_pitch;
	bool m_owner;
};

} // namespace JornamEngine
