#pragma once
#include "VariableNumberFeatureCreator.h"
class SIFTFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	SIFTFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~SIFTFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


