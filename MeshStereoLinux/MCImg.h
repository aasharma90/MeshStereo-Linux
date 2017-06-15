#include <cstdio>
#pragma once

#ifndef __MCIMG_H__
#define __MCIMG_H__

template<class T> class MCImg 
{
	// The class name stands for "Multi-Channel Image".
public:
	T *data;
	int w, h, n;
	bool is_shared;
	T *get(int y, int x) { return &data[(y*w + x)*n]; }			/* Get patch (y, x). */
	T *line(int y) { return &data[y*w]; }						/* Get line y assuming n=1. */
	MCImg() { w = h = n = 0; data = NULL; }
	MCImg(const MCImg& obj)
	{
		// This constructor is very necessary for returning an object in a function,
		// in the case that the Name Return Value Optimization (NRVO) is turned off.
		w = obj.w; h = obj.h; n = obj.n; is_shared = obj.is_shared;
		if (is_shared) { data = obj.data; }
		else { data = new T[w*h*n]; memcpy(data, obj.data, w*h*n*sizeof(T)); }
	}
	MCImg(int h_, int w_, int n_ = 1, T* data_ = NULL)
	{
		w = w_; h = h_; n = n_;
		if (!data_) { data = new T[w*h*n]; is_shared = false; memset(data, 0, w*h*n*sizeof(T)); }
		else	    { data = data_;        is_shared = true; }
	}
	void create(int _h, int _w, int _n = 1)
	{
		h = _h; w = _w; n = _n;
		if (!is_shared && data) { delete[] data; }
		is_shared = false;
		data = new T[w*h*n];
		memset(data, 0, w*h*n*sizeof(T));
	}
	MCImg& operator=(const MCImg& m)
	{
		// printf("= operator invoked.\n");
		// FIXME: it's not suggested to overload assignment operator, should declare a copyTo() function instead.
		// However, if the assignment operator is not overloaded, do not invoke it (e.g. a = b), it is dangerous.
		if (data) { delete[] data; }
		w = m.w; h = m.h; n = m.n; is_shared = m.is_shared;
		if (m.is_shared) { data = m.data; }
		else { data = new T[w*h*n]; memcpy(data, m.data, w*h*n*sizeof(T)); }
		return *this;
	}
	~MCImg() { if (!is_shared) delete[] data; }
	T *operator[](int y) { return &data[y*w]; }

	void SaveToBinaryFile(std::string filename)
	{
		FILE *fid = fopen(filename.c_str(), "wb");
		assert(fid != NULL);
		fwrite(data, sizeof(T), w*h*n, fid);
		fclose(fid);
	}
	void LoadFromBinaryFile(std::string filename)
	{
		FILE *fid = fopen(filename.c_str(), "rb");
		assert(fid != NULL);
		fread(data, sizeof(T), w*h*n, fid);
		fclose(fid);
	}
};

#endif
