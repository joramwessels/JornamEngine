#include "headers.h"

namespace JornamEngine {

void SideBySideRenderer::init(Scene* a_scene, uint SSAA)
{

}

void SideBySideRenderer::render(Camera* a_camera)
{

}

void Renderer::drawWorldAxes(Camera* camera, float unitLength)
{
	// unitLength is negated because m_left and m_up go in the negative axis directions
	uint middlex = m_scrwidth / 2, middley = m_scrheight / 2;
	int xendx = (int) (-unitLength * (vec3(1.0f, 0.0f, 0.0f).dot(camera->getLeft())));
	int xendy = (int) (-unitLength * (vec3(1.0f, 0.0f, 0.0f).dot(camera->getUp())));
	if (xendx != 0 || xendy != 0) drawLine(middlex, middley, middlex + xendx, middley + xendy, 0x00FF0000);
	int yendx = (int) (-unitLength * (vec3(0.0f, 1.0f, 0.0f).dot(camera->getLeft())));
	int yendy = (int) (-unitLength * (vec3(0.0f, 1.0f, 0.0f).dot(camera->getUp())));
	if (yendx != 0 || yendy != 0) drawLine(middlex, middley, middlex + yendx, middley + yendy, 0x0000FF00);
	int zendx = (int) (-unitLength * (vec3(0.0f, 0.0f, 1.0f).dot(camera->getLeft())));
	int zendy = (int) (-unitLength * (vec3(0.0f, 0.0f, 1.0f).dot(camera->getUp())));
	if (zendx != 0 || zendy != 0) drawLine(middlex, middley, middlex + zendx, middley + zendy, 0x000000FF);
}

void Renderer::drawLine(uint startx, uint starty, uint endx, uint endy, Color color)
{
	if (startx > endx) { swap(&startx, &endx); swap(&starty, &endy); }
	int dx = endx - startx, dy = endy - starty;
	if (dx >= abs(dy)) for (uint x = startx; x <= endx; x++)
	{
		int y = starty + dy * (int)(x - startx) / dx;
		m_screen->Plot(x, y, color);
	}
	else for (uint y = starty; y <= endy; y++)
	{
		int x = startx + dx * (int)(y - starty) / dy;
		m_screen->Plot(x, y, color);
	}
}

} // namespace Engine