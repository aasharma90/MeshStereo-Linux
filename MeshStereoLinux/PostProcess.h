#pragma once
#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__

#include <opencv2/core/core.hpp>
#include <vector>
#include "SlantedPlane.h"
#include "MCImg.h"




cv::Mat CrossCheck(cv::Mat &dispL, cv::Mat &dispR, int sign, float thresh);

void DisparityHoleFilling(cv::Mat &disp, MCImg<SlantedPlane> &slantedPlanes, cv::Mat &validPixelMap);

float SelectWeightedMedianFromPatch(cv::Mat &disp, int yc, int xc, float *w);

void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img);

std::vector<float> DetermineConfidence(cv::Mat &validPixelMap, std::vector<std::vector<cv::Point2i> > &segPixelLists);

cv::Mat DrawSegmentConfidenceMap(int numRows, int numCols, std::vector<float> &confidence,
	std::vector<std::vector<cv::Point2i> > &segPixelLists);

void SegmentOcclusionFilling(std::vector<SlantedPlane> &slantedPlanes, std::vector<cv::Point2f> &baryCenters,
	std::vector<std::vector<int> > &nbIndices, std::vector<float> &confidence);

void SegmentOcclusionFilling(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<int> > &nbIndices,
	std::vector<float> &confidence, std::vector<std::vector<cv::Point2i> > &segPixelLists);

#endif
