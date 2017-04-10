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

	/// <summary>
	/// Adds a training image that contains 0 or more regions to the training set
	/// </summary>
	void addTrainingImage(TrainingImage& img);
	
	/// <summary>
	/// Adds a region to the existing training image corresponding to the given image nr
	/// </summary>
	void addTrainingRegion(int imageNumber, TrainingRegion& region);

	/// <summary>
	/// Returns the number of images in underlying dataset of the training set
	/// </summary>
	int getNumberOfImages() const;

	/// <summary>
	/// Determines whether a pedestrian with given depth average can be in a window with given height
	/// </summary>
	bool isWithinValidDepthRange(int height, float depthAverage) const;

	/// <summary>
	/// Returns the underlying data set
	/// </summary>
	DataSet* getDataSet() const;

	/// <summary>
	/// Saves the training set to the given path
	/// </summary>
	void save(std::string& path);

	/// <summary>
	/// Clears existing data and loads the training set from the given path
	/// </summary>
	void load(std::string& path);

	/// <summary>
	/// Iterates over the data set images that are used in this training set
	/// </summary>
	void iterateDataSetImages(std::function<void(int imageNumber, cv::Mat&rgb, cv::Mat&depth, const std::vector<TrainingRegion>& regions)> func) const;

	/// <summary>
	/// Iterates over the regions in the training set. Optionally flips the region as well to have additional samples
	/// </summary>	
	void iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal)> func, bool addFlipped, int refWidth, int refHeight) const;	
};

