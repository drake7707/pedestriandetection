#pragma once
#include <opencv2/opencv.hpp>
#include "Histogram2D.h"
#include <functional>


class IntegralHistogram2D {

private:
	std::vector<std::vector<cv::Mat>> ihist;
	int binSize;
public:
	/// <summary>
	/// Creates a integral histogram from the given bin value at each pixel
	/// </summary>
	void create(int width, int height, int binSize, std::function<void(int x, int y, std::vector<std::vector<cv::Mat>>& ihist)> setBinValues);


	/// <summary>
	/// Calculates the histogram for a specific region from the integral histogram, uses the given histogram to fill in data to prevent an allocation and copy on return
	/// </summary>
	void calculateHistogramIntegral(int x, int y, int w, int h, Histogram2D& outHist) const;

	/// <summary>
	/// Calculates the histogram for a specific region from the integral histogram
	/// </summary>
	Histogram2D calculateHistogramIntegral(int x, int y, int w, int h) const;

};