#pragma once

#define X_AXIS vec3(1, 0, 0)
#define Y_AXIS vec3(0, 1, 0)
#define Z_AXIS vec3(0, 0, 1)

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

// Exception class for classes in the engine
class JornamException : public std::exception
{
public:
	enum LEVEL {DEBUG, INFO, WARN, ERR, FATAL};
	LEVEL m_severity;
	std::string m_class;
	std::string m_msg;
	JornamException(const std::string a_class, std::string a_msg, const LEVEL a_severity) :
		m_class(a_class), m_msg(a_msg), m_severity(a_severity) {};
	const char* what()
	{
		return ("JornamException in " + m_class + " class: " + m_msg).c_str();
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