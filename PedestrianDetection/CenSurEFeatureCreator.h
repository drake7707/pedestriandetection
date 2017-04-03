#pragma once
#include "VariableNumberFeatureCreator.h"
class CenSurEFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	CenSurEFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~CenSurEFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


