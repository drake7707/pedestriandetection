#pragma once
#include <string>
#include "IFeatureCreator.h"

class VariableNumberFeatureCreator : public IFeatureCreator
{

private:
	int clusterSize;
	std::vector<FeatureVector> centroids;

public:

	void prepare(std::string& datasetPath);

	virtual int getNumberOfFeatures() const;

	virtual std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const = 0;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual std::string explainFeature(int featureIndex, double featureValue) const;


	void saveCentroids(std::string& path);
	void loadCentroids(std::string& path);

	VariableNumberFeatureCreator(std::string& creatorName, int clusterSize);
	~VariableNumberFeatureCreator();
};

