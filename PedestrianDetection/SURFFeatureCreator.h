#pragma once
#include "VariableNumberFeatureCreator.h"
class SURFFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	SURFFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~SURFFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


