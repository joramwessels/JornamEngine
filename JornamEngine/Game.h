#pragma once

namespace JornamEngine {

class Game
{
public:
	Game(Surface* surface, bool* exitApp) : m_screen(surface), m_exitApp(exitApp) {};
	~Game() {};
	void init();
	void tick(float timeElapsed);
	void quitGame();
	void shutdown();

	// Input Handling
	void keyEsc(bool down) { quitGame(); }
	void keyUp(bool down) { m_camera->moveForward(m_playerSpeed); }
	void keyDown(bool down) { m_camera->moveForward(-m_playerSpeed); }
	void keyLeft(bool down) { m_camera->moveLeft(m_playerSpeed); }
	void keyRight(bool down) { m_camera->moveLeft(-m_playerSpeed); }
	void leftClick(bool down) {}
	void rightClick(bool down) {}
	void MouseMotion(Sint32 x, Sint32 y);

private:
	bool* m_exitApp;
	Surface* m_screen;
	Renderer* m_renderer;
	Scene* m_scene;
	Camera* m_camera;

	uint m_currentTick = 0;
	uint m_maxTicks = 500;

	float m_mouseSensitivity = 0.01f;
	float m_playerSpeed = 1.0f;
};

} // namespace Engine