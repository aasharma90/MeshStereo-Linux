#pragma once
#ifndef __TIMER_H__
#define __TIMER_H__

#include <stack>
#include <ctime>
#include <cstdio>


namespace bs {
	class Timer
	{
	public:
		static void Tic();
		static void Tic(const char *msg);
		static void Toc();

	private:
		static std::stack<clock_t> timeStamps;
	};
}


#endif 
