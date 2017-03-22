#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DataSetLabel.h"
#include <functional>

class DataSet
{
public:
	DataSet();
	~DataSet();

	virtual std::vector<DataSetLabel> getLabels() const = 0;

	virtual std::vector<cv::Mat> getImagesForNumber(int number) const = 0;

	virtual int getNrOfImages() const = 0;

	void DataSet::iterateDataSetWithSlidingWindow(std::vector<cv::Size>& windowSizes, int baseWindowStride,
		int refWidth, int refHeight,
		std::function<bool(int number)> canSelectFunc,
		std::function<void(int imageNumber)> onImageStarted,
		std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTP)> func,
		std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
		int parallization) const;

	virtual bool isWithinValidDepthRange(int height, float depthAverage) const {
		return true;
	}
};

