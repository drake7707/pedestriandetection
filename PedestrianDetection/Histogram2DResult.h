#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"

namespace hog {


	struct Histogram2DResult {
		int width;
		int height;

		std::vector<std::vector<Histogram>> data;

		cv::Mat hogImage;

		FeatureVector getFeatureArray() const;

		FeatureVector getHistogramVarianceFeatures()const;

		cv::Mat combineHOGImage(cv::Mat& img) const;
	};


}