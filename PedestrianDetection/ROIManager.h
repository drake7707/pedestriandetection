#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include <functional>
#include "IntegralImage.h"
class ROIManager
{

private:
	IntegralHistogram thermalRegions;

public:
	ROIManager();
	~ROIManager();


	void prepare(cv::Mat& mRGB, cv::Mat& mDepth, cv::Mat& mThermal);

	bool needToEvaluate(const cv::Rect2d& bbox, const cv::Mat& mRGB, const cv::Mat & mDepth, const cv::Mat& mThermal, std::function<bool(double, double)> isValidDepthRangeFunc) const;
};

