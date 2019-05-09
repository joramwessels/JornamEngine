#pragma once

namespace JornamEngine {

enum USE_GPU { NO, CUDA, OPENCL }; // { NO, CUDA, OPENCL }
enum SCREENHALF { BOTH, LEFT, RIGHT }; // { BOTH, LEFT, RIGHT }

class Renderer
{
public:
	Renderer(Surface* screen, USE_GPU useGPU, SCREENHALF renderhalf) :
		m_screen(screen),
		m_scrwidth(screen->GetWidth()),
		m_scrheight(screen->GetHeight()),
		m_useGPU(useGPU),
		m_renderhalf(renderhalf)
	{};
	Renderer(Surface* screen, USE_GPU useGPU) : Renderer(screen, useGPU, SCREENHALF::BOTH) {};
	~Renderer() {};
	virtual void init(uint SSAA) {}; // Called once at the start of the application
	virtual void tick() {};						   // Called at the start of every frame
	virtual void render(Camera* camera) {};		   // Called at the end of every frame
	void drawWorldAxes(Camera* camera, float unitLength = 20.0f);
	void setScene(Scene* scene) { m_scene = scene; }
protected:
	const uint m_scrwidth;
	const uint m_scrheight;
	const USE_GPU m_useGPU;
	const SCREENHALF m_renderhalf;
	Surface* m_screen;
	Scene* m_scene;
	uint m_SSAA;
	
	inline void drawLine(uint startx, uint starty, uint endx, uint endy, Color color);
};

class SideBySideRenderer : public Renderer
{
public:
	SideBySideRenderer(Surface* screen, USE_GPU useGPU) :
		Renderer(screen, useGPU) {};
	~SideBySideRenderer() {};
	void init(Scene* scene, uint SSAA);
	void render(Camera* camera);
private:
	Renderer* m_leftRenderer;
	Renderer* m_rightRenderer;
};

} // namespace Engine