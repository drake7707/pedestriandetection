#pragma once
#include <string>
#include "IFeatureCreator.h"
#include "TrainingDataSet.h"

class VariableNumberFeatureCreator : public IFeatureCreator
{

private:
	int clusterSize;
	std::vector<FeatureVector> centroids;

public:

	void prepare(TrainingDataSet& trainingDataSet);

	virtual int getNumberOfFeatures() const;

	virtual std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const = 0;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;


	void saveCentroids(std::string& path);
	void loadCentroids(std::string& path);

	VariableNumberFeatureCreator(std::string& creatorName, int clusterSize);
	~VariableNumberFeatureCreator();
};

