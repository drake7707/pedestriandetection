#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "Histogram.h"
#include "FeatureVector.h"

namespace hog {
	struct HOGResult {
		int width;
		int height;

		std::vector<std::vector<Histogram>> data;

		cv::Mat hogImage;


		FeatureVector getFeatureArray() {
			FeatureVector arr;
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

		FeatureVector getHistogramVarianceFeatures() {
			FeatureVector arr;
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
					arr.push_back(data[j][i].getS2());
			}
			return arr;
		}

		cv::Mat combineHOGImage(cv::Mat& img) {

			cv::Mat result;
			if (img.channels() == 1)
				cv::cvtColor(img, result, CV_GRAY2BGR);
			else
				result = img.clone();

			for (int j = 0; j < result.rows; j++)
			{
				for (int i = 0; i < result.cols; i++) {

					cv::Vec3b hog = hogImage.at<cv::Vec3b>(j, i);
					if (hog[2] > 0) { // red
						cv::Vec3b res = result.at<cv::Vec3b>(j, i);
						result.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0,hog[2]);
					}
				}
			}
			return result;
		}
	};

	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize);

	int getNumberOfFeatures(int imgWidth, int imgHeight, int patchSize, int binSize);

	std::string explainHOGFeature(int featureIndex, double featureValue, int imgWidth, int imgHeight, int patchSize, int binSize);

	HOGResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, bool createImage = false, bool l2normalize = true);

	HOGResult getHistogramsOfDepthDifferences(cv::Mat& img, int patchSize, int binSize, bool createImage, bool l2normalize);


	HOGResult getHistogramsOfX(cv::Mat& imgValues, cv::Mat& imgBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize);
}
