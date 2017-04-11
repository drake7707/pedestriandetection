#pragma once
#include <opencv2/opencv.hpp>
#include "Histogram.h"
#include <functional>


class IntegralHistogram {

private:
	std::vector<cv::Mat> ihist;
	int binSize;
public:
	/// <summary>
	/// Creates a integral histogram from the given bin value at each pixel
	/// </summary>
	void create(int width, int height, int binSize, std::function<void(int x, int y, std::vector<cv::Mat>& ihist)> setBinValues);
	
	/// <summary>
	/// Calculates the histogram for a specific region from the integral histogram
	/// </summary>
	Histogram calculateHistogramIntegral(int x, int y, int w, int h) const;

};