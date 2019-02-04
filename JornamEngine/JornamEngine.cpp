#include "headers.h"

char* applicationName = "JornamEngine";
uint scrwidth = 512;
uint scrheight = 512;

int main(int argc, char* argv[])
{
	openConsole(500);

	// SDL window
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* sdl_window = SDL_CreateWindow(applicationName, 100, 100, scrwidth, scrheight, SDL_WINDOW_SHOWN);
	SDL_Renderer* sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
	SDL_Texture* sdl_frameBuffer = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, scrwidth, scrheight);

	// Initializing game
	bool exit = false;
	JornamEngine::Timer* timer = new JornamEngine::Timer();
	JornamEngine::Surface* surface = new JornamEngine::Surface(scrwidth, scrheight);
	JornamEngine::Game* game = new JornamEngine::Game(surface, &exit);
	surface->Clear(0xff0000);
	game->init();

	while (!exit)
	{
		// Rendering
		void* target = 0;
		int pitch;
		SDL_LockTexture(sdl_frameBuffer, NULL, &target, &pitch);
		unsigned char* t = (unsigned char*)target;
		for (uint i = 0; i < scrheight; i++)
		{
			memcpy(t, surface->GetBuffer() + i * scrwidth, scrwidth * 4);
			t += pitch;
		}
		SDL_UnlockTexture(sdl_frameBuffer);
		SDL_RenderCopy(sdl_renderer, sdl_frameBuffer, NULL, NULL);
		SDL_RenderPresent(sdl_renderer);

		// Game tick
		game->tick(timer->elapsed());
		timer->reset();

		// SDL input handling
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			switch (event.type)
			{
			case SDL_QUIT:
				exit = true;
				break;
			case SDL_KEYDOWN:
				game->KeyDown(event.key.keysym.scancode);
			case SDL_KEYUP:
				game->KeyUp(event.key.keysym.scancode);
			case SDL_MOUSEMOTION:
				game->MouseMotion(event.motion.xrel, event.motion.yrel);
			case SDL_MOUSEBUTTONDOWN:
				game->MouseUp(event.button.button);
			case SDL_MOUSEBUTTONUP:
				game->MouseDown(event.button.button);
			default:
				break;
			}
		}
		exit = true;
	}
	game->lateShutdown();
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
	
	std::ios::sync_with_stdio();
	freopen("CON", "w", stdout);
	freopen("CON", "w", stderr);
}
