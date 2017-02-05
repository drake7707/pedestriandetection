#pragma once
#include "opencv2/opencv.hpp"

typedef std::vector<float> Histogram;

struct HoGResult {
	int width;
	int height;

	std::vector<std::vector<Histogram>> data;

	cv::Mat hogImage;
};
HoGResult getHistogramsOfOrientedGradient(cv::Mat& mat, int patchSize, int binSize, bool createImage = false);