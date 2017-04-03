#pragma once
#include "IFeatureCreator.h"
#include "VariableNumberFeatureCreator.h"

class CornerFeatureCreator : public VariableNumberFeatureCreator
{

private:
	int patchSize = 8;
	int refWidth = 64;
	int refHeight = 128;
	IFeatureCreator::Target target;
public:
	CornerFeatureCreator(std::string& name, IFeatureCreator::Target target, int clusterSize = 80);
	virtual ~CornerFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};

