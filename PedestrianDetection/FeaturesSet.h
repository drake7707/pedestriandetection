#pragma once
#include "FeatureVector.h"
#include "IFeatureCreator.h"
#include "opencv2/opencv.hpp"


class FeaturesSet
{
private:
	std::vector<IFeatureCreator*> creators;

public:
	FeaturesSet();
	~FeaturesSet();

	void addCreator(IFeatureCreator* creator);

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

