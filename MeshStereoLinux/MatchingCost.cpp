#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>

#include "MCImg.h"
#include "SlantedPlane.h"
#include "StereoAPI.h"
#include "ReleaseAssert.h"
#include "Timer.h"






extern int						PATCHRADIUS;
extern int						PATCHWIDTH;
extern float					GRANULARITY;
extern MCImg<unsigned short>	gDsiL;
extern MCImg<unsigned short>	gDsiR;




bool InBound(int y, int x, int numRows, int numCols)
{
	return 0 <= y && y < numRows && 0 <= x && x < numCols;
}

bool InBound(cv::Point &p, int numRows, int numCols)
{
	return 0 <= p.y && p.y < numRows && 0 <= p.x && p.x < numCols;
}

cv::Mat ComputeCensusImage(cv::Mat &img, int vpad, int hpad)
{
	// Assumes input is a BGR image or a 3-channel gray image.
	int numRows = img.rows, numCols = img.cols;
	cv::Mat imgGray;
	cv::cvtColor(img, imgGray, CV_BGR2GRAY);

	// We use long long for each element of censusImg, 
	// but CV_64FC1 is the only suitable size we can choose.
	cv::Mat censusImg(numRows, numCols, CV_64FC1);
	ASSERT(censusImg.isContinuous())
		long long *census = (long long*)censusImg.data;


	memset(census, 0, imgGray.rows * imgGray.cols * sizeof(long long));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {

			int uu = std::max(i - vpad, 0);
			int dd = std::min(i + vpad, numRows - 1);
			int ll = std::max(j - hpad, 0);
			int rr = std::min(j + hpad, numCols - 1);

			int idx = 0;
			long long feature = 0;
			uchar center = imgGray.at<unsigned char>(i, j);

			for (int ii = uu; ii <= dd; ii++) {
				for (int jj = ll; jj <= rr; jj++) {
					feature |= ((long long)(imgGray.at<unsigned char>(ii, jj) > center) << idx);
					idx++;
				}
			}

			census[i * numCols + j] = feature;
		}
	}

	return censusImg;
}

cv::Mat ComputeGradientImage(cv::Mat &img)
{
	int numRows = img.rows, numCols = img.cols;
	cv::Mat grayImg;
	cv::Mat gradientImg(numRows, numCols, CV_32FC4);
	cv::Mat grayfImg(numRows + 2, numCols + 2, CV_32FC1);
	grayfImg.setTo(0.f);

	cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	grayImg.convertTo(grayImg, CV_32FC1, 1 / 255.f);
	grayImg.copyTo(grayfImg(cv::Rect(1, 1, numCols, numRows)));


	#define BUF(y, x) grayfImg.at<float>(y + 1, x + 1)
	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			float dx = (BUF(y + 0, x + 1) - BUF(y + 0, x - 1)) * 0.5
					 + (BUF(y - 1, x + 1) - BUF(y - 1, x - 1)) * 0.25
					 + (BUF(y + 1, x + 1) - BUF(y + 1, x - 1)) * 0.25;

			float dy = (BUF(y + 1, x + 0) - BUF(y - 1, x + 0)) * 0.5
					 + (BUF(y + 1, x - 1) - BUF(y - 1, x - 1)) * 0.25
					 + (BUF(y + 1, x + 1) - BUF(y - 1, x + 1)) * 0.25;

			float dxy = BUF(y + 1, x + 1) - BUF(y - 1, x - 1);
			float dyx = BUF(y + 1, x - 1) - BUF(y - 1, x + 1);

			gradientImg.at<cv::Vec4f>(y, x) = cv::Vec4f(dx, dy, dxy, dyx);
		}
	}

	return gradientImg;
}

cv::Mat ComputeCappedSobelImage(cv::Mat &imgIn, int sobelCapValue)
{

	cv::Mat img;
	if (imgIn.channels() == 3) {
		cv::cvtColor(imgIn, img, CV_BGR2GRAY);
	}
	else {
		img = imgIn.clone();
	}
	//cv::imshow("img", img);
	//cv::waitKey(0);

	int numRows = img.rows, numCols = img.cols;
	cv::Mat sobelImage(numRows, numCols, CV_8UC1);
	ASSERT(sobelImage.isContinuous());
	memset(sobelImage.data, sobelCapValue, numRows * numCols * sizeof(unsigned char));

	for (int y = 1; y < numRows - 1; ++y) {
		for (int x = 1; x < numCols - 1; ++x) {
			int sobelValue
				= (1.f * img.at<unsigned char>(y - 1, x + 1) + 2.f * img.at<unsigned char>(y, x + 1) + 1.f * img.at<unsigned char>(y + 1, x + 1))
				- (1.f * img.at<unsigned char>(y - 1, x - 1) + 2.f * img.at<unsigned char>(y, x - 1) + 1.f * img.at<unsigned char>(y + 1, x - 1));
			if (sobelValue > sobelCapValue) sobelValue = 2 * sobelCapValue;
			else if (sobelValue < -sobelCapValue) sobelValue = 0;
			else sobelValue += sobelCapValue;
			sobelImage.at<unsigned char>(y, x) = sobelValue;
		}
	}



	//unsigned char *image = img.data;
	//for (int y = 1; y < numRows - 1; ++y) {
	//	for (int x = 1; x < numCols - 1; ++x) {
	//		int sobelValue = (image[numCols*(y - 1) + x + 1] + 2 * image[numCols*y + x + 1] + image[numCols*(y + 1) + x + 1])
	//			- (image[numCols*(y - 1) + x - 1] + 2 * image[numCols*y + x - 1] + image[numCols*(y + 1) + x - 1]);
	//		if (sobelValue > sobelCapValue) sobelValue = 2 * sobelCapValue;
	//		else if (sobelValue < -sobelCapValue) sobelValue = 0;
	//		else sobelValue += sobelCapValue;
	//		sobelImage.data[numCols*y + x] = sobelValue;
	//	}
	//}

	return sobelImage;
}

cv::Mat ComputeGradxyImage(cv::Mat& img)
{
	int numRows = img.rows, numCols = img.cols;
	int sobelScale = 1, sobelDelta = 0;
	cv::Mat gray, grad_x, grad_y;

	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3, sobelScale, sobelDelta, cv::BORDER_DEFAULT);
	cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3, sobelScale, sobelDelta, cv::BORDER_DEFAULT);
	grad_x = grad_x / 8.f;
	grad_y = grad_y / 8.f;

	cv::Mat grad(numRows, numCols, CV_32FC2);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			grad.at<cv::Vec2f>(y, x)[0] = grad_x.at<float>(y, x);
			grad.at<cv::Vec2f>(y, x)[1] = grad_y.at<float>(y, x);
		}
	}

	return grad / 255.f;
}

float L2Dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
	return std::sqrt(
		  ((float)a[0] - b[0]) * ((float)a[0] - b[0])
		+ ((float)a[1] - b[1]) * ((float)a[1] - b[1])
		+ ((float)a[2] - b[2]) * ((float)a[2] - b[2])
		);
}

int L1Dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
	return std::abs((int)a[0] - (int)b[0])
		 + std::abs((int)a[1] - (int)b[1])
		 + std::abs((int)a[2] - (int)b[2]);
}

float L1Dist(const cv::Vec2f &a, const cv::Vec2f &b)
{
	return std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]);
}

float L1Dist(const cv::Vec3f &a, const cv::Vec3f &b)
{
	return std::abs(a[0] - b[0]) 
		 + std::abs(a[1] - b[1]) 
		 + std::abs(a[2] - b[2]);
}

float L1Dist(const cv::Vec4f &a, const cv::Vec4f &b)
{
	return std::abs(a[0] - b[0])
		 + std::abs(a[1] - b[1])
		 + std::abs(a[2] - b[2])
		 + std::abs(a[3] - b[3]);
}

int HammingDist(const long long x, const long long y)
{
	int dist = 0;
	long long val = x ^ y; // XOR
	// Count the number of set bits
	while (val)
	{
		++dist;
		val &= val - 1;
	}
	return dist;
}

MCImg<float> Compute9x7CensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat censusImgL = ComputeCensusImage(imL, 4, 3);
	cv::Mat censusImgR = ComputeCensusImage(imR, 4, 3);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)				xm += numCols;
				if (xm > numCols - 1)	xm -= numCols;

				long long censusL = censusImgL.at<long long>(y, x);
				long long censusR = censusImgR.at<long long>(y, xm + 0.5);

				float censusDiff = HammingDist(censusL, censusR);
				dsiL.get(y, x)[level] = censusDiff;
			}
		}
	}

	return dsiL;
}

MCImg<float> ComputeAdCensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	#define AD_LAMBDA		600.f
	#define CENSUS_LAMBDA	10.f

	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat censusImgL = ComputeCensusImage(imL);
	cv::Mat censusImgR = ComputeCensusImage(imR);
	cv::Mat bgrL;	imL.convertTo(bgrL, CV_32FC3, 1.f);
	cv::Mat bgrR;	imR.convertTo(bgrR, CV_32FC3, 1.f);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)			xm += numCols;
				if (xm > numCols - 1) xm -= numCols;

				float xmL, xmR, wL, wR;
				float cost_col, cost_grad;
				xmL = (int)(xm);
				xmR = (int)(xm + 0.99);
				wL = xmR - xm;
				wR = 1.f - wL;

				cv::Vec3f &colorL	= bgrL.at<cv::Vec3f>(y, x);
				cv::Vec3f &colorRmL = bgrR.at<cv::Vec3f>(y, xmL);
				cv::Vec3f &colorRmR = bgrR.at<cv::Vec3f>(y, xmR);
				cv::Vec3f colorR	= wL * colorRmL + wR * colorRmR;
				long long censusL	= censusImgL.at<long long>(y, x);
				long long censusR	= censusImgR.at<long long>(y, xm + 0.5);

				float adDiff = 0.3333f * L1Dist(colorL, colorR);
				float censusDiff = HammingDist(censusL, censusR);
				dsiL.get(y, x)[level] = 2.f - exp(-adDiff / AD_LAMBDA) - exp(-censusDiff / CENSUS_LAMBDA);
			}
		}
	}

	return dsiL;


}

MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	#define COLORGRADALPHA		0.05f
	#define COLORMAXDIFF		0.04f
	#define GRADMAXDIFF			0.01f

	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat gradientL = ComputeGradientImage(imL);
	cv::Mat gradientR = ComputeGradientImage(imR);
	cv::Mat bgrL;	imL.convertTo(bgrL, CV_32FC3, 1 / 255.f);
	cv::Mat bgrR;	imR.convertTo(bgrR, CV_32FC3, 1 / 255.f);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)				xm += numCols;
				if (xm > numCols - 1)	xm -= numCols;

				float xmL, xmR, wL, wR;
				xmL = (int)(xm);
				xmR = (int)(xm + 0.99);
				wL = xmR - xm;
				wR = 1.f - wL;

				cv::Vec3f &colorL	= bgrL.at<cv::Vec3f>(y, x);
				cv::Vec3f &colorRmL	= bgrR.at<cv::Vec3f>(y, xmL);
				cv::Vec3f &colorRmR	= bgrR.at<cv::Vec3f>(y, xmR);
				cv::Vec4f &gradL	= gradientL.at<cv::Vec4f>(y, x);
				cv::Vec4f &gradRmL	= gradientR.at<cv::Vec4f>(y, xmL);
				cv::Vec4f &gradRmR	= gradientR.at<cv::Vec4f>(y, xmR);
				cv::Vec3f colorR	= wL * colorRmL + wR * colorRmR;
				cv::Vec4f gradR		= wL * gradRmL + wR * gradRmR;


				float costColor = std::min(COLORMAXDIFF, L1Dist(colorL, colorR));
				float costGrad  = std::min(GRADMAXDIFF,  L1Dist(gradL, gradR));
				dsiL.get(y, x)[level] = COLORGRADALPHA * costColor + (1 - COLORGRADALPHA) * costGrad;
			}
		}
	}

	return dsiL;
}

cv::Mat WinnerTakesAll(MCImg<float> &dsi, float granularity)
{
	int numRows = dsi.h, numCols = dsi.w, ndisps = dsi.n;
	cv::Mat disp(numRows, numCols, CV_32FC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int minidx = 0;
			float *cost = dsi.get(y, x);
			for (int k = 1; k < ndisps; k++) {
				if (cost[k] < cost[minidx]) {
					minidx = k;
				}
			}
			disp.at<float>(y, x) = minidx * granularity;
		}
	}
	return disp;
}

template<typename T> static inline T BilinearInterp(cv::Mat &im, int y, float x)
{
	int numRows = im.rows, numCols = im.cols;
	ASSERT(0 <= y && y < numRows);
	
	x = std::max(0.f, std::min(numCols - 1.f, x));
	int xL = floor(x);
	int xR = ceil(x);
	float wL = std::abs(x - xR);
	float wR = 1.f - wL;
	return wL * im.at<T>(y, xL) + wR * im.at<T>(y, xR);
}

float gExpTable[256 * 3];

static void InitExpTable(float *table)
{
	for (int dist = 0; dist < 256 * 3; dist++) {
		gExpTable[dist] = exp(-dist / 30.f);
	}
}

std::vector<cv::Vec3b> gMeanLabColorsL, gMeanLabColorsR;

float PatchMatchSlantedPlaneCost(int yc, int xc, SlantedPlane &slantedPlane, int sign)
{
	
	static int initExpTable = false;
	if (!initExpTable) {
		InitExpTable(gExpTable);
		initExpTable = true;
	}

	if (std::abs(slantedPlane.nx) > 0.3 && std::abs(slantedPlane.ny) > 0.3) {
		return 1e+7;
	}
	if (std::abs(slantedPlane.nz < 0.5)) {
		return 1e+7;
	}

	extern int INTERP_ONLINE;
	extern std::string ROOTFOLDER;
	extern cv::Mat gImLabL, gImLabR;

	if (gDsiL.data != NULL) {

		

		MCImg<unsigned short> &dsi = (sign == -1 ? gDsiL : gDsiR);
		cv::Mat &img = (sign == -1 ? gImLabL : gImLabR);
		int numRows = dsi.h, numCols = dsi.w, maxLevel = dsi.n - 1;

		cv::Vec3b center = img.at<cv::Vec3b>(yc, xc);
		float totalCost = 0.f, accWeight = 0.f;

		const int MATCHINGCOST_STRIDE = 1;
		for (int STRIDE = 1; STRIDE <= MATCHINGCOST_STRIDE; STRIDE++) {
			for (int y = yc - STRIDE * PATCHRADIUS; y <= yc + STRIDE * PATCHRADIUS; y += STRIDE) {
				for (int x = xc - STRIDE * PATCHRADIUS; x <= xc + STRIDE * PATCHRADIUS; x += STRIDE) {
					if (InBound(y, x, numRows, numCols)) {
						cv::Vec3b &c = img.at<cv::Vec3b>(y, x);
						float w = gExpTable[L1Dist(c, center)];
						float d = slantedPlane.ToDisparity(y, x);
						int level = 0.5 + d / GRANULARITY;
						level = std::max(0, std::min(maxLevel, level));

						totalCost += w * dsi.get(y, x)[level];
						accWeight += w;
					}
				}
			}
		}
		return totalCost / accWeight;


		
	}
	else {		
		extern cv::Mat gLabelMapL, gLabelMapR;
		extern cv::Mat gSobelImgL, gSobelImgR, gCensusImgL, gCensusImgR;

		cv::Mat &labImg		= (sign == -1 ? gImLabL : gImLabR);
		cv::Mat &labelMap	= (sign == -1 ? gLabelMapL : gLabelMapR);
		cv::Mat &sobelImgL	= (sign == -1 ? gSobelImgL : gSobelImgR);
		cv::Mat &sobelImgR	= (sign == -1 ? gSobelImgR : gSobelImgL);
		cv::Mat &censusImgL = (sign == -1 ? gCensusImgL : gCensusImgR);
		cv::Mat &censusImgR = (sign == -1 ? gCensusImgR : gCensusImgL);
		//cv::Mat &labImgL	= (sign == -1 ? gImLabL : gImLabR);
		//cv::Mat &labImgR	= (sign == -1 ? gImLabR : gImLabL);

		int numRows = labImg.rows, numCols = labImg.cols;
		cv::Vec3b center = labImg.at<cv::Vec3b>(yc, xc);
		float totalCost = 0.f, accWeight = 0.f;
		const int MATCHINGCOST_STRIDE = 1;
		for (int STRIDE = 1; STRIDE <= MATCHINGCOST_STRIDE; STRIDE++) {
			for (int y = yc - STRIDE * PATCHRADIUS; y <= yc + STRIDE * PATCHRADIUS; y += STRIDE) {
				for (int x = xc - STRIDE * PATCHRADIUS; x <= xc + STRIDE * PATCHRADIUS; x += STRIDE) {
					if (InBound(y, x, numRows, numCols)) {
						cv::Vec3b &c = labImg.at<cv::Vec3b>(y, x);
						/*	float w = 1.f;
							if (labelMap.at<int>(yc, xc) != labelMap.at<int>(y, x)) {
							w = gExpTable[L1Dist(c, center)];
							}*/
						float w = gExpTable[L1Dist(c, center)];
						float d = slantedPlane.ToDisparity(y, x);
						int xm = x + sign * d + 0.5;
						xm = std::max(0, std::min(numCols - 1, xm));

						unsigned char &sobelPixelL = sobelImgL.at<unsigned char>(y, x);
						unsigned char &sobelPixelR = sobelImgR.at<unsigned char>(y, xm);
						long long &censusPixelL = censusImgL.at<long long>(y, x);
						long long &censusPixelR = censusImgR.at<long long>(y, xm);

						float cost = std::abs((float)sobelPixelL - sobelPixelR)
							+ 2.f * HammingDist(censusPixelL, censusPixelR);
						//cv::Vec3b &rawPixelL = labImgL.at<cv::Vec3b>(y, x);
						//cv::Vec3b &rawPixelR = labImgR.at<cv::Vec3b>(y, xm);
						//float cost = std::min(30, L1Dist(rawPixelL, rawPixelR))
						//	+ 2.f * HammingDist(censusPixelL, censusPixelR);

						totalCost += w * cost;
						accWeight += w;
					}
				}
			}
		}

		return totalCost / accWeight;
	}
}

void InitGlobalColorGradientFeatures(cv::Mat &imL, cv::Mat &imR)
{
	extern cv::Mat gImLabL, gImLabR, gImGradL, gImGradR;
	imL.convertTo(gImLabL, CV_32FC3, 1.f / 255.f);
	imR.convertTo(gImLabR, CV_32FC3, 1.f / 255.f);
	gImGradL = ComputeGradientImage(imL);
	gImGradR = ComputeGradientImage(imR);
}

void InitGlobalDsiAndSimWeights(cv::Mat &imL, cv::Mat &imR, int numDisps)
{
#if 0
	extern cv::Mat gImLabL, gImLabR;
	gImLabL = imL.clone();
	gImLabR = imR.clone();
	//cv::cvtColor(imL, gImLabL, CV_BGR2Lab);
	//cv::cvtColor(imR, gImLabR, CV_BGR2Lab);

	extern float SIMILARITY_GAMMA;
	int numRows = imL.rows, numCols = imL.cols;
	if (gMatchingCostType == ADCENSUS) {
		gDsiL = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdCensusCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}
	else if (gMatchingCostType == ADGRADIENT) {
		gDsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}

	std::vector<SimVector> simVecsStdL;
	std::vector<SimVector> simVecsStdR;

	if (gCostAggregationType == GRID) {
		bs::Timer::Tic("Precompute Similarity Weights");
		//MCImg<float> PrecomputeSimilarityWeights(cv::Mat &img, int patchRadius, int simGamma);
		//gSimWeightsL = PrecomputeSimilarityWeights(imL, PATCHRADIUS, SIMILARITY_GAMMA);
		//gSimWeightsR = PrecomputeSimilarityWeights(imR, PATCHRADIUS, SIMILARITY_GAMMA);
		bs::Timer::Toc();
	}
	else if (gCostAggregationType == TOP50) {
		bs::Timer::Tic("Begin SelfSimilarityPropagation");
		SelfSimilarityPropagation(imL, simVecsStdL);
		InitSimVecWeights(imL, simVecsStdL);
		gSimVecsL = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdL[0]);
		SelfSimilarityPropagation(imR, simVecsStdR);
		InitSimVecWeights(imR, simVecsStdR);
		gSimVecsR = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdR[0]);
		bs::Timer::Toc();
	}
#endif
}
