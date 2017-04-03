#pragma once
#include "VariableNumberFeatureCreator.h"
class FASTFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	FASTFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~FASTFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


