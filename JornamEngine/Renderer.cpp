#include "headers.h"

namespace JornamEngine {

/*
	Draws the world axis on the centre of the screen (XYZ represented by RGB respectively)

	@param camera		A pointer to the camera object
	@param unitLength	The pixel length of each vector
*/
void Renderer::drawWorldAxes(Camera* camera, float unitLength)
{
	// Axes are negated because m_left and m_up go in the negative axis directions
	vec3 right = camera->getLeft() * -unitLength;
	vec3 down = camera->getUp() * -unitLength;
	uint middlex = m_scrwidth / 2, middley = m_scrheight / 2;
	int xendx = (int) (vec3(1.0f, 0.0f, 0.0f).dot(right));
	int xendy = (int) (vec3(1.0f, 0.0f, 0.0f).dot(down));
	if (xendx != 0 || xendy != 0) drawLine(middlex, middley, middlex + xendx, middley + xendy, COLOR::RED);
	int yendx = (int) (vec3(0.0f, 1.0f, 0.0f).dot(right));
	int yendy = (int) (vec3(0.0f, 1.0f, 0.0f).dot(down));
	if (yendx != 0 || yendy != 0) drawLine(middlex, middley, middlex + yendx, middley + yendy, COLOR::GREEN);
	int zendx = (int) (vec3(0.0f, 0.0f, 1.0f).dot(right));
	int zendy = (int) (vec3(0.0f, 0.0f, 1.0f).dot(down));
	if (zendx != 0 || zendy != 0) drawLine(middlex, middley, middlex + zendx, middley + zendy, COLOR::BLUE);
}

/*
	Very basic line drawing algorithm; no thickness, no AA, not optimized

	@param startx	The x-coordinate of the starting pixel
	@param starty	The y-coordinate of the starting pixel
	@param endx		The x-coordinate of the final pixel
	@param endy		The y-coordinate of the final pixel
	@param color	The color of the line
*/
void Renderer::drawLine(uint startx, uint starty, uint endx, uint endy, Color color)
{
	int dx = endx - startx, dy = endy - starty;
	if (abs(dx) >= abs(dy))
	{
		if (startx > endx) { swap(&startx, &endx); swap(&starty, &endy); }
		for (uint x = startx; x <= endx; x++)
		{
			int y = starty + dy * (int)(x - startx) / dx;
			m_screen->Plot(x, y, color);
		}
	}
	else
	{
		if (starty > endy) { swap(&startx, &endx); swap(&starty, &endy); }
		for (uint y = starty; y <= endy; y++)
		{
			int x = startx + dx * (int)(y - starty) / dy;
			m_screen->Plot(x, y, color);
		}
	}
}

} // namespace Engine