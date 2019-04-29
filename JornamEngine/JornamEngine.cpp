#include "headers.h"

namespace JornamEngine {

char* applicationName = "JornamEngine";
uint scrwidth = 512;
uint scrheight = 512;

optix::Context context;

// GLUT callbacks
void initializeGLUT(int* argc, char** argv);
void glutDisplay();
void destroyContext() { if (context) { context->destroy(); context = 0; } }
optix::Buffer getOutputBuffer() { return context["output_buffer"]->getBuffer(); }
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

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

void createContext()
{
	GLuint outputBuffer = 0;
	glGenBuffers(1, &outputBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, outputBuffer);
	glBufferData(GL_ARRAY_BUFFER, 4 * scrwidth * scrheight, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ------------------------------------------------------
// FreeGLUT functions
// ------------------------------------------------------

// Initializes an OpenGL window
void initializeGLUT(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(scrwidth, scrheight);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(applicationName);
	glutHideWindow();
}

// Registers callbacks and runs the GLUT main loop
void runGLUT()
{
	// Initializing GL state
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, scrwidth, scrheight);

	glutShowWindow();
	glutReshapeWindow(scrwidth, scrheight);

	// Regestering GLUT callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	//glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);	
	glutCloseFunc(destroyContext);

	glutMainLoop();
}

void displayBuffer(optix::Buffer buffer)
{
	optix::Buffer buffer = optix::Buffer::take(buffer->get());

	// Query buffer information
	RTsize buffer_width_rts, buffer_height_rts;
	buffer->getSize(buffer_width_rts, buffer_height_rts);
	RTformat buffer_format = buffer->getFormat();

	GLboolean use_SRGB = GL_FALSE;
	if (buffer_format == RT_FORMAT_FLOAT4 || buffer_format == RT_FORMAT_FLOAT3)
	{
		glGetBooleanv(GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &use_SRGB);
		if (use_SRGB)
			glEnable(GL_FRAMEBUFFER_SRGB_EXT);
	}

	static unsigned int gl_tex_id = 0;
	if (!gl_tex_id)
	{
		glGenTextures(1, &gl_tex_id);
		glBindTexture(GL_TEXTURE_2D, gl_tex_id);

		// Change these to GL_LINEAR for super- or sub-sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	glBindTexture(GL_TEXTURE_2D, gl_tex_id);

	// send PBO or host-mapped image data to texture
	const unsigned pboId = buffer->getGLBOId();
	GLvoid* imageData = 0;
	if (pboId)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
	else
		imageData = buffer->map(0, RT_BUFFER_MAP_READ);

	RTsize elmt_size = buffer->getElementSize();
	if (elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
	else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
	else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	GLenum pixel_format = GL_BGRA;

	if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, scrwidth, scrheight, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
	else if (buffer_format == RT_FORMAT_FLOAT4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, scrwidth, scrheight, 0, pixel_format, GL_FLOAT, imageData);
	else if (buffer_format == RT_FORMAT_FLOAT3)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, scrwidth, scrheight, 0, pixel_format, GL_FLOAT, imageData);
	else if (buffer_format == RT_FORMAT_FLOAT)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, scrwidth, scrheight, 0, pixel_format, GL_FLOAT, imageData);

	if (pboId)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	else
		buffer->unmap();

	// 1:1 texel to pixel mapping with glOrtho(0, 1, 0, 1, -1, 1) setup:
	// The quad coordinates go from lower left corner of the lower left pixel
	// to the upper right corner of the upper right pixel.
	// Same for the texel coordinates.

	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);

	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	if (use_SRGB) glDisable(GL_FRAMEBUFFER_SRGB_EXT);
}

void glutDisplay()
{
	//updateCamera();

	context->launch(0, scrwidth, scrheight);

	displayBuffer(getOutputBuffer());

	glutSwapBuffers();
}

void glutKeyboardPress(unsigned char k, int x, int y)
{
	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	}
}

void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
}


void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(scrwidth);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(scrheight);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = fminf(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / scrwidth, from.y / scrheight };
		const float2 b = { to.x / width, to.y / height };

		camera_rotate = arcball.rotate(b, a);
	}

	mouse_prev_pos = make_int2(x, y);
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
			if (event.key.keysym.scancode == SDLK_ESCAPE)	   game->keyEsc(true);
			else if (event.key.keysym.scancode == SDLK_UP) 	   game->keyUp(true);
			else if (event.key.keysym.scancode == SDLK_DOWN)   game->keyDown(true);
			else if (event.key.keysym.scancode == SDLK_LEFT)   game->keyLeft(true);
			else if (event.key.keysym.scancode == SDLK_RIGHT)  game->keyRight(true);
			break;
		case SDL_KEYUP:
			if (event.key.keysym.scancode == SDLK_ESCAPE)	   game->keyEsc(false);
			else if (event.key.keysym.scancode == SDLK_UP) 	   game->keyUp(false);
			else if (event.key.keysym.scancode == SDLK_DOWN)   game->keyDown(false);
			else if (event.key.keysym.scancode == SDLK_LEFT)   game->keyLeft(false);
			else if (event.key.keysym.scancode == SDLK_RIGHT)  game->keyRight(false);
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

} // namespace Engine


// Routing application entry point into the namespace
int main(int argc, char* argv[]) { return JornamEngine::SDL_main(argc, argv); }