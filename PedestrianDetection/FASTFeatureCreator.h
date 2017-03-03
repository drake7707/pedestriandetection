#pragma once
#include "VariableNumberFeatureCreator.h"
class FASTFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	FASTFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~FASTFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

