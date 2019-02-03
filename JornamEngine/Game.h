#pragma once

namespace JornamEngine {

class Game
{
public:
	Game(Surface* surface, bool* exitApp) : m_screen(surface), m_exitApp(exitApp) {};
	~Game() {};
	void init();
	void tick(float timeElapsed);
	void lateShutdown();
	void KeyDown(SDL_Scancode key);
	void KeyUp(SDL_Scancode key);
	void MouseMotion(Sint32 x, Sint32 y);
	void MouseUp(Uint8 button);
	void MouseDown(Uint8 button);
private:
	bool* m_exitApp;
	Surface* m_screen;
	Renderer* m_renderer;
	Scene* scene;

	void shutdown();
};

} // namespace Engine