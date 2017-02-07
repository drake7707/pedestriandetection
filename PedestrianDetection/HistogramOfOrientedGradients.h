#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"


struct HoGResult {
	int width;
	int height;

	std::vector<std::vector<Histogram>> data;

	cv::Mat hogImage;


	std::vector<float> getFeatureArray() {
		std::vector<float> arr;
		arr.reserve(width * height * 10);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (float el : data[j][i])
					arr.push_back(el);
			}
		}
		return arr;
	}

	cv::Mat getFeatureMat() {
		auto featureArray = getFeatureArray();

		double maxVal = *std::max_element(featureArray.begin(), featureArray.end());

		cv::Mat imgSample(1, featureArray.size(), CV_32FC1);
		for (int i = 0; i < featureArray.size(); i++)
			imgSample.at<float>(0, i) = featureArray[i];
		return imgSample;
	}
};

HoGResult getHistogramsOfOrientedGradient(cv::Mat& mat, int patchSize, int binSize, bool createImage = false);