#include "headers.h"

namespace JornamEngine {

char* applicationName = "JornamEngine";
uint scrwidth = 512;
uint scrheight = 512;

int main(int argc, char* argv[])
{
	if (JE_DEBUG_MODE) openConsole(500);

	// Initializing SDL window
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* sdl_window = SDL_CreateWindow(applicationName, 100, 100, scrwidth, scrheight, SDL_WINDOW_SHOWN);
	SDL_Renderer* sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
	SDL_Texture* sdl_frameBuffer = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, scrwidth, scrheight);

	// Initializing game
	bool exit = false;
	JornamEngine::Timer*   timer   = new JornamEngine::Timer();
	JornamEngine::Surface* surface = new JornamEngine::Surface(scrwidth, scrheight);
	JornamEngine::Game*    game    = new JornamEngine::Game(surface, &exit);
	surface->Clear();
	game->init();

	// Game loop
	while (!exit)
	{
		handleSDLInput(game, &exit);
		game->tick(timer->elapsed());
		timer->reset();
		renderToScreen(sdl_frameBuffer, sdl_renderer, surface);
	}

	// Termination
	game->shutdown();
	SDL_Quit();
	return 0;
}

// Opens a console for debugging
void openConsole(short bufferSize)
{
	// Attaching a new console to the process
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	AllocConsole();

	// Setting the buffer size
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
	coninfo.dwSize.Y = bufferSize;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);
	
	// Redirecting stdout to console
	HANDLE h1 = GetStdHandle(STD_OUTPUT_HANDLE);
	int h2 = _open_osfhandle((intptr_t)h1, _O_TEXT);
	FILE* fp = _fdopen(h2, "w");
	*stdout = *fp;
	setvbuf(stdout, NULL, _IONBF, 0);
	
	// Redirecting stdin to console
	h1 = GetStdHandle(STD_INPUT_HANDLE);
	h2 = _open_osfhandle((intptr_t)h1, _O_TEXT);
	fp = _fdopen(h2, "r");
	*stdin = *fp;
	setvbuf(stdin, NULL, _IONBF, 0);
	
	// Redirecting stderr to console
	h1 = GetStdHandle(STD_ERROR_HANDLE);
	h2 = _open_osfhandle((intptr_t)h1, _O_TEXT);
	fp = _fdopen(h2, "w");
	*stderr = *fp;
	setvbuf(stderr, NULL, _IONBF, 0);
	
	// Reopening streams
	std::ios::sync_with_stdio();
	freopen("CON", "w", stdout);
	freopen("CON", "w", stderr);
}

// Handles the SDL events such as user input
void handleSDLInput(JornamEngine::Game* game, bool* exit)
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			*exit = true;
			break;
		case SDL_KEYDOWN:
			if (event.key.keysym.scancode == JE_SDLK_ESCAPE)	   game->keyEsc(true);
			else if (event.key.keysym.scancode == JE_SDLK_UP) 	   game->keyUp(true);
			else if (event.key.keysym.scancode == JE_SDLK_DOWN)   game->keyDown(true);
			else if (event.key.keysym.scancode == JE_SDLK_LEFT)   game->keyLeft(true);
			else if (event.key.keysym.scancode == JE_SDLK_RIGHT)  game->keyRight(true);
			else if (event.key.keysym.scancode == JE_SDLK_W)	  game->keyW(true);
			else if (event.key.keysym.scancode == JE_SDLK_A)	  game->keyA(true);
			else if (event.key.keysym.scancode == JE_SDLK_S)	  game->keyS(true);
			else if (event.key.keysym.scancode == JE_SDLK_D)	  game->keyD(true);
			else logDebug(
				"JornamEngine",
				("unknown key down scancode " + std::to_string(event.key.keysym.scancode)).c_str(),
				JornamException::DEBUG
			);
			break;
		case SDL_KEYUP:
			if (event.key.keysym.scancode == JE_SDLK_ESCAPE)	   game->keyEsc(false);
			else if (event.key.keysym.scancode == JE_SDLK_UP) 	   game->keyUp(false);
			else if (event.key.keysym.scancode == JE_SDLK_DOWN)   game->keyDown(false);
			else if (event.key.keysym.scancode == JE_SDLK_LEFT)   game->keyLeft(false);
			else if (event.key.keysym.scancode == JE_SDLK_RIGHT)  game->keyRight(false);
			else if (event.key.keysym.scancode == JE_SDLK_W)	  game->keyW(false);
			else if (event.key.keysym.scancode == JE_SDLK_A)	  game->keyA(false);
			else if (event.key.keysym.scancode == JE_SDLK_S)	  game->keyS(false);
			else if (event.key.keysym.scancode == JE_SDLK_D)	  game->keyD(false);
			break;
		case SDL_MOUSEBUTTONDOWN:
			if (event.button.button == SDL_BUTTON_LEFT)		   game->leftClick(true);
			else if (event.button.button == SDL_BUTTON_RIGHT)  game->rightClick(true);
			break;
		case SDL_MOUSEBUTTONUP:
			if (event.button.button == SDL_BUTTON_LEFT)		   game->leftClick(false);
			else if (event.button.button == SDL_BUTTON_RIGHT)  game->rightClick(false);
			break;
		case SDL_MOUSEMOTION:
			game->MouseMotion(event.motion.xrel, event.motion.yrel);
			break;
		default:
			break;
		}
	}
}

// Renders the frame buffer to the monitor
void renderToScreen(SDL_Texture* sdl_buff, SDL_Renderer* sdl_rend, JornamEngine::Surface* surface)
{
	void* target = 0;
	int pitch;
	SDL_LockTexture(sdl_buff, NULL, &target, &pitch);
	unsigned char* t = (unsigned char*)target;
	for (int i = 0; i < surface->GetHeight(); i++)
	{
		memcpy(t, surface->GetBuffer() + i * surface->GetWidth(), surface->GetWidth() * sizeof(Color));
		t += pitch;
	}
	SDL_UnlockTexture(sdl_buff);
	SDL_RenderCopy(sdl_rend, sdl_buff, NULL, NULL);
	SDL_RenderPresent(sdl_rend);
}

} // namespace Engine


// Routing application entry point into the namespace
int main(int argc, char* argv[]) { return JornamEngine::SDL_main(argc, argv); }