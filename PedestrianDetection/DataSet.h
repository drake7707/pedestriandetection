#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DataSetLabel.h"
#include <functional>
#include "ROIManager.h"

class DataSet
{
public:
	DataSet();
	~DataSet();

	/// <summary>
	/// Returns the name of the data set
	/// </summary>
	virtual std::string getName() const = 0;

	/// <summary>
	/// Returns all the labels of the data set
	/// </summary>
	virtual std::vector<DataSetLabel> getLabels() const = 0;

	/// <summary>
	/// Returns the data set labels per image number
	/// </summary>
	std::vector<std::vector<DataSetLabel>> getLabelsPerNumber() const;

	/// <summary>
	/// Returns the actual images for the given image number, in order: RGB/Depth/Thermal
	/// </summary>
	virtual std::vector<cv::Mat> getImagesForNumber(int number) const = 0;

	/// <summary>
	/// Returns the number of images in the data set
	/// </summary>
	virtual int getNrOfImages() const = 0;

	/// <summary>
	/// Returns if depth information is available or is approximated so risk analysis can be done
	/// </summary>
	virtual bool canDoRiskAnalysis() const = 0;

	/// <summary>
	/// Iterates the data set with the given window sizes on each image that can be selected and given stride
	/// </summary>
	void DataSet::iterateDataSetWithSlidingWindow(const std::vector<cv::Size>& windowSizes, int baseWindowStride,
		int refWidth, int refHeight,
		std::function<bool(int number)> canSelectFunc,
		std::function<void(int imgNr)> onImageStarted,
		std::function<void(int imgNr, double scale, cv::Mat& fullRGBScale, cv::Mat& fullDepthScale, cv::Mat& fullThermalScale)> onScaleStarted,
		std::function<void(int idx, int resultClass, int imageNumber, int scale, cv::Rect2d& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTruePositive)> func,
		std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
		int parallization) const;


	/// <summary>
	/// Checks whether the height of a window and the depth average in the middle of the window can be a true positive
	/// </summary>
	virtual bool isWithinValidDepthRange(int height, float depthAverage) const;

	/// <summary>
	/// Returns the various categories possible
	/// </summary>
	virtual std::vector<std::string> getCategories() const = 0;

	/// <summary>
	/// Returns the category of the data set label
	/// </summary>
	virtual std::string getCategory(DataSetLabel* label) const = 0;

	/// <summary>
	/// Returns which requirements the data set fullfills: RGB,Depth,Thermal
	/// </summary>
	virtual std::vector<bool> getFullfillsRequirements() const = 0;
};

