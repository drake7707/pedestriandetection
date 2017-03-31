#pragma once
#include "VariableNumberFeatureCreator.h"
class ORBFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	ORBFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~ORBFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};


