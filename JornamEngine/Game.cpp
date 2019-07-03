#include "headers.h"

namespace JornamEngine {

// Called once before starting the game loop
void Game::init()
{
	m_renderer = new RayTracer(m_screen);
	m_camera = new Camera(m_screen->GetWidth() / 800.0f, m_screen->GetHeight() / 800.0f);
	m_scene = new Scene(m_renderer->getContext(), "Assets\\Scenes\\floor.scene", m_camera);
	m_renderer->init(m_scene);

	logger.logDebug("Game", "initialized\n", JornamException::DEBUG);
}

// Called every game loop
void Game::tick(float a_timeElapsed)
{
	m_camera->tick();
	m_renderer->tick();

	m_renderer->render(m_camera);
	m_renderer->drawWorldAxes(m_camera, 50.0f);
}

// Exits the game loop
void Game::quitGame()
{
	// exit screen or something
	*m_exitApp = true;
	logger.logDebug("Game", "Quit\n", JornamException::DEBUG);
}

// Called when the game is exited by the engine (sdl_quit or error)
void Game::shutdown()
{
	// save progress or something
	logger.logDebug("Game", "Shutdown\n", JornamException::DEBUG);
}

// Input handling
void Game::MouseMotion(Sint32 x, Sint32 y)
{
	if (x != 0) m_camera->rotateY((float)x);
	if (y != 0) m_camera->rotateX((float)y);
}

} // namespace Engine