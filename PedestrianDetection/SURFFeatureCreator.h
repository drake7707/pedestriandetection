#pragma once
#include "VariableNumberFeatureCreator.h"
class SURFFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	SURFFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~SURFFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};


