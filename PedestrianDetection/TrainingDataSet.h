#pragma once
#include <string>
#include "opencv2/opencv.hpp"
#include <map>
#include <functional>
#include "KITTIDataSet.h"
#include <memory>

struct TrainingRegion {
	cv::Rect region;
	int regionClass;
	std::string category;
};

struct TrainingImage {

	int number;
	std::vector<TrainingRegion> regions;

};

class TrainingDataSet
{
	std::string baseDataSetPath;
	std::map<int, TrainingImage> images;

	DataSet* dataSet;

public:
	

	TrainingDataSet(DataSet* dataSet);

	TrainingDataSet(const TrainingDataSet& dataSet);

	~TrainingDataSet();

	void addTrainingImage(TrainingImage& img);
	
	void addTrainingRegion(int imageNumber, TrainingRegion& region);

	int getNumberOfImages() const;

	bool isWithinValidDepthRange(int height, float depthAverage) const;

	DataSet* getDataSet() const;

	void save(std::string& path);
	void load(std::string& path);

	void iterateDataSetImages(std::function<void(int imageNumber, cv::Mat&rgb, cv::Mat&depth, const std::vector<TrainingRegion>& regions)> func) const;

	void iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal)> func, bool addFlipped, int refWidth, int refHeight) const;	
};

