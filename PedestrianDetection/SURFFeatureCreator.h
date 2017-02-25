#pragma once
#include "VariableNumberFeatureCreator.h"
class SURFFeatureCreator :
	public VariableNumberFeatureCreator
{
public:
	SURFFeatureCreator();
	virtual ~SURFFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};


