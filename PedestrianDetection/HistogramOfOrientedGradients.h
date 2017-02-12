#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "Histogram.h"

struct HoGResult {
	int width;
	int height;

	std::vector<std::vector<Histogram>> data;

	cv::Mat hogImage;


	std::vector<float> getFeatureArray(bool addS2) {
		std::vector<float> arr;
		arr.reserve(width * height * 10);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (float el : data[j][i])
					arr.push_back(el);

				if (addS2)
					arr.push_back(data[j][i].getS2());
			}
		}
		return arr;
	}

	cv::Mat getFeatureMat(bool addS2) {
		auto featureArray = getFeatureArray(addS2);
		cv::Mat imgSample(1, featureArray.size(), CV_32FC1);
		for (int i = 0; i < featureArray.size(); i++)
			imgSample.at<float>(0, i) = featureArray[i];
		return imgSample;
	}
};

cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize);


HoGResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, bool createImage = false, bool l2normalize = true);


HoGResult getHistogramsOfX(cv::Mat& imgValues, cv::Mat& imgBinningValues, int patchSize, int binSize, bool createImage);