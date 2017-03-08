#pragma once
#include <string>
#include "opencv2/opencv.hpp"
#include <map>
#include <functional>
#include "KITTIDataSet.h"

struct TrainingRegion {
	cv::Rect region;
	int regionClass;
};

struct TrainingImage {

	int number;
	std::vector<TrainingRegion> regions;

};

class TrainingDataSet
{
	std::string baseDataSetPath;
	std::map<int, TrainingImage> images;

	int refWidth = 64;
	int refHeight = 128;

	KITTIDataSet dataSet;

public:
	int nrOfSamples = 0;

	TrainingDataSet(std::string& baseDataSetPath);

	TrainingDataSet(const TrainingDataSet& dataSet);

	~TrainingDataSet();

	void addTrainingImage(TrainingImage& img);
	void addTrainingRegion(int imageNumber, TrainingRegion& region);


	void save(std::string& path);
	void load(std::string& path);

	int getNumberOfImages() const;

	void iterateDataSetImages(std::function<void(int imageNumber, cv::Mat&rgb, cv::Mat&depth, const std::vector<TrainingRegion>& regions)> func) const;

	void iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth)> func) const;

	void iterateDataSetWithSlidingWindow(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTP)> func, int parallization = 8) const;

	std::string getBaseDataSetPath() const;

	bool isWithinValidDepthRange(int height, float depthAverage) const;
};

