#pragma once
#ifndef __STEREOAPI_H__
#define __STEREOAPI_H__

#include <opencv2/core/core.hpp>
#include "MCImg.h"
#include "SlantedPlane.h"




#define SIMVECTORSIZE	50

struct SimVector
{
	cv::Point2i pos[SIMVECTORSIZE];
	float w[SIMVECTORSIZE];
};

bool InBound(int y, int x, int numRows, int numCols);

bool InBound(cv::Point &p, int numRows, int numCols);

cv::Mat ComputeCensusImage(cv::Mat &img, int vpad = 3, int hpad = 4);

cv::Mat ComputeGradientImage(cv::Mat &img);

int L1Dist(const cv::Vec3b &a, const cv::Vec3b &b);

float L1Dist(const cv::Vec3f &a, const cv::Vec3f &b);

float L1Dist(const cv::Vec4f &a, const cv::Vec4f &b);

int HammingDist(const long long x, const long long y);

MCImg<float> Compute9x7CensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity);

MCImg<float> ComputeAdCensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity);

MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity);

cv::Mat WinnerTakesAll(MCImg<float> &dsi, float granularity);

void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp,
	std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap); // = cv::Mat());

void SetupStereoParameters(std::string rootFolder, int &numDisps, int &maxDisp, int &visualizeScale);

void EvaluateDisparity(std::string rootFolder, cv::Mat &dispL, float eps = 1.f,
	void *auxParamsPtr = NULL, std::string mouseCallbackName = "OnMouseEvaluateDisparity");

void RunPatchMatchOnPixels(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR);

void InitSimVecWeights(cv::Mat &img, std::vector<SimVector> &simVecs);

void SelfSimilarityPropagation(cv::Mat &img, std::vector<SimVector> &simVecs);

void Triangulate2DImage(cv::Mat& img, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int> > &triVertexInds);

cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int> > &triVertexInds);

float PatchMatchSlantedPlaneCost(int yc, int xc, SlantedPlane &slantedPlane, int sign);

void PatchMatchOnPixelPostProcess(MCImg<SlantedPlane> &slantedPlanesL, MCImg<SlantedPlane> &slantedPlanesR,
	cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR);

cv::Mat SlantedPlaneMapToDisparityMap(MCImg<SlantedPlane> &slantedPlanes);

cv::Mat CrossCheck(cv::Mat &dispL, cv::Mat &dispR, int sign, float thresh = 1.f);

int SLICSegmentation(const cv::Mat &img, const int numPreferedRegions, const int compactness, cv::Mat& labelMap, cv::Mat& contourImg);

void InitGlobalColorGradientFeatures(cv::Mat &imL, cv::Mat &imR);

void InitGlobalDsiAndSimWeights(cv::Mat &imL, cv::Mat &imR, int numDisps);
#endif
