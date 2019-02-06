#include "headers.h"

namespace JornamEngine {

// Called once before starting the game loop
void Game::init()
{
	m_scene = new Scene(64, 64);
	m_scene->testParsers();
	m_renderer = new RayTracer(m_screen, USE_GPU::NO);
	m_renderer->init(m_scene, 0);
	printf("Game initialized\n");
}

// Called every game loop
void Game::tick(float a_timeElapsed)
{
	m_renderer->render(vec3{ 0, 0, 0 }, vec3{ 0, 0, 0 });
	printf("Game ticked\n");

	if (m_currentTick++ == m_maxTicks) quitGame();
}

// Exits the game loop
void Game::quitGame()
{
	printf("Game shutdown\n");
	*m_exitApp = true;
}

// Called when the game is exited by the engine (sdl_quit or error)
void Game::shutdown()
{
	printf("Late shutdown\n");
}

// Input handling
void Game::KeyDown(SDL_Scancode key)
{

}
void Game::KeyUp(SDL_Scancode key)
{

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