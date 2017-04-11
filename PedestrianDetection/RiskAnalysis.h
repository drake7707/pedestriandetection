#pragma once
#include "DataSet.h"
#include "opencv2/opencv.hpp"
#include "MarchingSquares.h"

namespace RiskAnalysis {

	/// <summary>
	/// Calculates the stopping distance based on the vehicle speed (m/s) and tire road friction
	/// </summary>
	double getStoppingDistance(double vehicleSpeedForRating, double tireroadFriction);

	/// <summary>
	/// Calculates the remaining time to hit a pedestrian (in seconds) with given speed and pedestrian location
	/// </summary>
	double getRemainingTimeToHitPedestrian(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction = 0.7, int t = 1);

	/// <summary>
	/// Returns the available risk categories
	/// </summary>
	std::vector<std::string> getRiskCategories();

	/// <summary>
	/// Determines which risk category the pedestrian belongs to
	/// </summary>
	std::string getRiskCategory(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction = 0.7, int t = 1);

	/// <summary>
	/// Creates a top down view of the scene with car and pedestrians and annotates the various risk categories
	/// </summary>
	cv::Mat getTopDownImage(int imgWidth, int imgHeight, std::vector<DataSetLabel>& labels, float maxDepth, float vehicleSpeedKMh, float tireroadFriction, float t = 1);

	/// <summary>
	/// Draws the various risk categories and depth iso lines on the given image
	/// </summary>
	cv::Mat getRGBImage(cv::Mat& mRGB, cv::Mat& mDepth, std::vector<DataSetLabel>& labels, float vehicleSpeedKMh, float tireroadFriction, float t = 1);
}