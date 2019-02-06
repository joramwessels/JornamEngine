#pragma once

namespace JornamEngine {

typedef std::chrono::high_resolution_clock Clock;
typedef Clock::time_point TimePoint;
typedef std::chrono::microseconds MicroSeconds;

// Timer keeps track of a _start_ time point.
// Timer.reset() sets _start_ to current time
// Timer.elapsed() returns the difference between _start_ and the current time
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

class IOException : public std::exception
{
public:
	enum TYPE {OPEN, READ, WRITE, CLOSE};
	const std::string m_sourcefile;
	const std::string m_line;
	const std::string m_func;
	const std::string m_file;
	const TYPE m_type;
	IOException(const std::string sourcefile, const std::string line, const std::string func, const std::string file, TYPE type) :
		std::exception(("IOException: " + std::string((type == OPEN ? "OPEN" : (type == READ ? "READ" : (type == WRITE ? "WRITE" : "CLOSE")))) +\
			" error with filename \"" + file + "\"\n\tin " + func + " in file \"" + sourcefile + "\" line " + line + "\n").c_str()),
		m_type(type), m_sourcefile(sourcefile), m_line(line), m_func(func), m_file(file) {};
};

void openConsole(short bufferSize);
void handleSDLInput(JornamEngine::Game* game, bool* exit);
void renderToScreen(SDL_Texture* sdl_frameBuffer, SDL_Renderer* sdl_renderer, JornamEngine::Surface* surface);

} // namespace Engine