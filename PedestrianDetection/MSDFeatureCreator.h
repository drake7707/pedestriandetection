#pragma once
#include "VariableNumberFeatureCreator.h"
class MSDFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	MSDFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~MSDFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


