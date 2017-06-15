#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <sstream>


#include "StereoAPI.h"
#include "Timer.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_video248d.lib")
#pragma comment(lib, "opencv_flann248d.lib")
#pragma comment(lib, "opencv_nonfree248d.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_video248.lib")
#pragma comment(lib, "opencv_flann248.lib")
#pragma comment(lib, "opencv_nonfree248.lib")
#endif



int main(int argc, char **argv)
{

	if (argc != 5) {
		printf("%s.exe filePathImageL filePathImageR filePathDispOut numDisps\n", argv[0]);
		exit(-1);
	}

	std::string filePathImageL	= argv[1];
	std::string filePathImageR	= argv[2];
	std::string filePathDispOut = argv[3];
	int numDisps				= atoi(argv[4]);

	void RunMeshStereo(std::string filePathImageL, std::string filePathImageR, std::string filePathDispOut, int numDisps);
	bs::Timer::Tic("MeshStereo");
	RunMeshStereo(filePathImageL, filePathImageR, filePathDispOut, numDisps);
	bs::Timer::Toc();
}



