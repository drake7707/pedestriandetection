#pragma once
#include "VariableNumberFeatureCreator.h"
class SIFTFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	SIFTFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~SIFTFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


