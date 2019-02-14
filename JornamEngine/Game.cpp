#include "headers.h"

namespace JornamEngine {

// Called once before starting the game loop
void Game::init()
{
	m_scene = new Scene("Assets\\Scenes\\floor.scene");
	m_renderer = new RayTracer(m_screen, USE_GPU::NO);
	m_renderer->init(m_scene, 0);
	m_camera = new Camera(m_screen->GetWidth(), m_screen->GetHeight());

	printf("Game initialized\n");
}

// Called every game loop
void Game::tick(float a_timeElapsed)
{
	m_camera->tick();
	m_renderer->tick();

	m_renderer->render(m_camera);
	printf("Game ticked\n");
	if (m_currentTick++ == m_maxTicks) quitGame();
}

// Exits the game loop
void Game::quitGame()
{
	// exit screen or something
	*m_exitApp = true;
	printf("Game shutdown\n");
}

// Called when the game is exited by the engine (sdl_quit or error)
void Game::shutdown()
{
	// save progress or something
	printf("Late shutdown\n");
}

// Input handling
void Game::KeyDown(SDL_Scancode key)
{
	m_camera->moveForward(-1.0f);
}
void Game::KeyUp(SDL_Scancode key)
{
	m_camera->moveForward();
}
void Game::MouseMotion(Sint32 x, Sint32 y)
{

}
void Game::MouseUp(Uint8 button)
{

}
void Game::MouseDown(Uint8 button)
{

}

} // namespace Engine