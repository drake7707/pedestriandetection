#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"


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


				double avg = 0;
				for (float el : data[j][i]) {
					avg += el;
					arr.push_back(el);
				}
				avg /= data[j][i].size();

				if (addS2) {
					double sumvar = 0;
					for (float el : data[j][i]) {
						sumvar += (el - avg) * (el - avg);
					}
					arr.push_back(sumvar);
				}

			}
		}
		return arr;
	}

	cv::Mat getFeatureMat(bool addS2) {
		auto featureArray = getFeatureArray(addS2);

		double maxVal = *std::max_element(featureArray.begin(), featureArray.end());

		cv::Mat imgSample(1, featureArray.size(), CV_32FC1);
		for (int i = 0; i < featureArray.size(); i++)
			imgSample.at<float>(0, i) = featureArray[i];
		return imgSample;
	}
};

cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize);


HoGResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, bool createImage = false, bool l2normalize = true);


HoGResult getHistogramsOfX(cv::Mat& imgValues, cv::Mat& imgBinningValues, int patchSize, int binSize, bool createImage);