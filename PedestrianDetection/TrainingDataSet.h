#pragma once
#include <string>
#include "opencv2/opencv.hpp"
#include <map>
#include <functional>

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

public:
	TrainingDataSet(std::string& baseDataSetPath);
	~TrainingDataSet();

	void addTrainingImage(TrainingImage& img);

	void save(std::string& path);
	void load(std::string& path);



	void iterateDataSet(std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth)> func) const;


	std::string getBaseDataSetPath() const;
};

