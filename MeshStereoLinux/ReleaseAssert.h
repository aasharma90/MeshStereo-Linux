#pragma once

#ifndef __RELEASEASSERT_H__
#define __RELEASEASSERT_H__

#define ASSERT(condition)								\
	if (!(condition)) {									\
		printf("ASSERT %s VIOLATED AT LINE %d, %s\n",	\
			#condition, __LINE__, __FILE__);			\
		exit(-1);										\
	}

#endif