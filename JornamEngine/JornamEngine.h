#pragma once

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

// Exception class regarding files, reading, writing, and parsing
class IOException : public std::exception
{
public:
	enum TYPE {OPEN, READ, WRITE, CLOSE};
	const std::string m_sourcefile;
	const std::string m_line;
	const std::string m_func;
	const std::string m_file;
	const std::string m_msg;
	const TYPE m_type;
	IOException(const std::string sourcefile, const std::string line, const std::string func, const std::string file, const std::string msg, TYPE type) :
		std::exception(("IOException: " + errorType2String(type) + " error with filename \"" + file + "\" - " +\
			msg + "\n\tin " + func + " in file \"" + sourcefile + "\" line " + line + "\n").c_str()),
		m_type(type), m_sourcefile(sourcefile), m_line(line), m_func(func), m_msg(msg), m_file(file) {};
private:
	std::string errorType2String(TYPE type)
	{
		return std::string((type == OPEN ? "OPEN" : (type == READ ? "READ" : (type == WRITE ? "WRITE" : "CLOSE"))));
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