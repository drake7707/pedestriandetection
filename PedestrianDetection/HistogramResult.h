#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"

namespace hog {


	struct HistogramResult {
		int width;
		int height;

		std::vector<std::vector<Histogram>> data;

		cv::Mat hogImage;

		/// <summary>
		/// Creates a feature vector from the histogram data
		/// </summary>
		FeatureVector getFeatureArray() const;

		/// <summary>
		/// Returns a feature vector containing the S^2 variance of each 8x8 cell
		/// </summary>
		FeatureVector getHistogramVarianceFeatures()const;

		/// <summary>
		/// Combines the HOG image of the result with the given rgb image
		/// </summary>
		cv::Mat combineHOGImage(cv::Mat& img) const;
	};


}