#include "headers.h"

namespace JornamEngine {

// Called once before starting the game loop
void Game::init()
{
	scene = 0;
	m_renderer = new RayTracer(m_screen, USE_GPU::NO);
	m_renderer->init(scene, 0);
	printf("Game initialized\n");
}

// Called every game loop
void Game::tick(float a_timeElapsed)
{
	m_renderer->render(vec3{ 0, 0, 0 }, vec3{ 0, 0, 0 });
	printf("Game ticked\n");
}

// Exits the game loop
void Game::shutdown()
{
	printf("Game shutdown\n");
	*m_exitApp = true;
}

// Called when the game is exited by the engine (sdl_quit or error)
void Game::lateShutdown()
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