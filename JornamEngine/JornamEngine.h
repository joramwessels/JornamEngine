#pragma once

//#define printf(x) fprintf(stdout, x)
//#define _CRT_SECURE_NO_WARNINGS

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

} // namespace Engine

void openConsole(short bufferSize);