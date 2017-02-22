#pragma once
#include "FeatureVector.h"
#include "IFeatureCreator.h"
#include "opencv2/opencv.hpp"


class FeatureSet
{
private:
	std::vector<IFeatureCreator*> creators;

public:
	FeatureSet();
	~FeatureSet();

	void addCreator(IFeatureCreator* creator);

	int getNumberOfFeatures() const;
	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	std::string explainFeature(int featureIndex, double splitValue) const;
};

