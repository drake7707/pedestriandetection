#include "HistogramResult.h"
#include "opencv2/opencv.hpp"

namespace hog {


	FeatureVector HistogramResult::getFeatureArray() const {
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

	FeatureVector HistogramResult::getHistogramVarianceFeatures() const {
		FeatureVector arr;
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
				arr.push_back(data[j][i].getS2());
		}
		return arr;
	}

	cv::Mat HistogramResult::combineHOGImage(cv::Mat& img)  const {

		cv::Mat result;
		if (img.channels() == 1) {
			if (img.type() == CV_32FC1) {
				img.convertTo(result, CV_8UC1, 255, 0);
				cv::cvtColor(result, result, CV_GRAY2BGR);
			}
			else
				cv::cvtColor(img, result, CV_GRAY2BGR);
		}
		else
			result = img.clone();

		for (int j = 0; j < result.rows; j++)
		{
			for (int i = 0; i < result.cols; i++) {

				cv::Vec3b hog = hogImage.at<cv::Vec3b>(j, i);
				if (hog[2] > 0) { // red
					cv::Vec3b res = result.at<cv::Vec3b>(j, i);
					result.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, hog[2]);
				}
			}
		}
		return result;
	}
}