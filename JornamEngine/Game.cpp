#include "headers.h"

namespace JornamEngine {

// Called once before starting the game loop
void Game::init()
{
	m_renderer = new RayTracer(m_screen, USE_GPU::NO);
	m_camera = new Camera(m_screen->GetWidth() / 800.0f, m_screen->GetHeight() / 800.0f);
	m_scene = new Scene("Assets\\Scenes\\floor.scene", m_camera);
	m_renderer->init(m_scene, 0);

	printf("Game initialized\n");
}

// Called every game loop
void Game::tick(float a_timeElapsed)
{
	printf("Game tick %.4f\n", a_timeElapsed);
	m_camera->tick();
	m_renderer->tick();

	m_renderer->render(m_camera);
	m_renderer->drawWorldAxes(m_camera, 50.0f);
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
	if (key == SDLK_ESCAPE) quitGame();
	if (key == SDLK_UP) m_camera->moveForward(10.0f);
	if (key == SDLK_DOWN) m_camera->moveForward(-1.0f);
	if (key == SDLK_LEFT) m_camera->moveLeft(1.0f);
	if (key == SDLK_RIGHT) m_camera->moveLeft(-1.0f);
}
void Game::KeyUp(SDL_Scancode key)
{
}
void Game::MouseMotion(Sint32 x, Sint32 y)
{
	if (x == 0 && y == 0) return;
	printf("Mouse motion x: %i, y: %i\n", x, y);
	m_camera->rotate(m_camera->getLeft(), y * m_mouseSensitivity);
	m_camera->rotate(m_camera->getUp(), x * m_mouseSensitivity);
}
void Game::MouseUp(Uint8 button)
{

}
void Game::MouseDown(Uint8 button)
{

}

} // namespace Engine