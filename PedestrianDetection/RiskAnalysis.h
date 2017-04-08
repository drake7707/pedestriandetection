#pragma once
#include "DataSet.h"
#include "opencv2/opencv.hpp"
#include "MarchingSquares.h"

namespace RiskAnalysis {
	double getStoppingDistance(double vehicleSpeedForRating, double tireroadFriction);

	double getRemainingTimeToHitPedestrian(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction = 0.7, int t = 1);

	std::vector<std::string> getRiskCategories();

	std::string getRiskCategory(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction = 0.7, int t = 1);

	cv::Mat getTopDownImage(int imgWidth, int imgHeight, std::vector<DataSetLabel>& labels, float maxDepth, float vehicleSpeedKMh, float tireroadFriction, float t = 1);

	cv::Mat getRGBImage(cv::Mat& mRGB, cv::Mat& mDepth, std::vector<DataSetLabel>& labels, float vehicleSpeedKMh, float tireroadFriction, float t = 1);
}