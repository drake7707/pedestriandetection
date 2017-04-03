#pragma once
#include "VariableNumberFeatureCreator.h"
class BRISKFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	IFeatureCreator::Target target;
public:
	BRISKFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target);
	virtual ~BRISKFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


