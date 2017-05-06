#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include <functional>
#include "IntegralHistogram.h"
#include "Helper.h"

class ROIManager
{

private:
	IntegralHistogram thermalRegions;
	std::vector<cv::Rect2d> candidates;
	cv::Mat thermalMask;


public:
	ROIManager();
	~ROIManager();


	/// <summary>
	/// Does the necessary preparations for the current given images so needToEvaluate can be checked quickly
	/// </summary>
	void prepare(cv::Mat& mRGB, cv::Mat& mDepth, cv::Mat& mThermal);

	/// <summary>
	/// Checks whether the given window bounding box needs to be evaluated, based on either depth correlation or thermal ROI.
	/// </summary>
	bool needToEvaluate(const cv::Rect2d& bbox, const cv::Mat& mRGB, const cv::Mat & mDepth, const cv::Mat& mThermal, std::function<bool(double, double)> isValidDepthRangeFunc) const;
};

