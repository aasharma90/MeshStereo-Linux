#include "Timer.h"

namespace bs {

	void Timer::Tic()
	{
		timeStamps.push(clock());
	}
	void Timer::Tic(const char *msg)
	{
		printf("Processing %s ...\n", msg);
		Tic();
	}
	void Timer::Toc()
	{
		clock_t tic = timeStamps.top();
		clock_t toc = clock();
		float timeElapsed = (toc - tic) / 1000.f;
		printf("%.2fs\n", timeElapsed);
		timeStamps.pop();
	}

	std::stack<clock_t> Timer::timeStamps;

}

