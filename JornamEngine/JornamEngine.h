#pragma once

// Engine axis system
#define JE_AXIS_RIGHT vec3(1, 0, 0)
#define JE_AXIS_DOWN vec3(0, 1, 0)
#define JE_AXIS_FORWARD vec3(0, 0, 1)

namespace JornamEngine {

typedef std::chrono::high_resolution_clock Clock;
typedef Clock::time_point TimePoint;
typedef std::chrono::microseconds MicroSeconds;

// Keeps track of a start time to compare the current time to
struct Timer
{
	TimePoint start;
	inline Timer() : start(get()) {}
	static inline TimePoint get() { return Clock::now(); }
	inline void reset() { start = get(); }
	inline float elapsed() const
	{
		auto diff = get() - start;
		auto duration = std::chrono::duration_cast<MicroSeconds>(diff);
		return static_cast<float>(duration.count()) / 1000.0f;
	}
};

// Checks if the given string ends with the specified extention
inline bool filenameHasExtention(const char* filename, const char* extention)
{
	std::string filename_str = std::string(filename);
	std::string extention_str = std::string(extention);
	return 0 == filename_str.compare(filename_str.length() - extention_str.length(), extention_str.length(), extention);
}

inline void openConsole(short bufferSize);
inline void handleSDLInput(JornamEngine::Game* game, bool* exit);
inline void renderToScreen(SDL_Texture* sdl_frameBuffer, SDL_Renderer* sdl_renderer, JornamEngine::Surface* surface);

} // namespace Engine