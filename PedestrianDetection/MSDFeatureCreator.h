#pragma once
#include "VariableNumberFeatureCreator.h"
class MSDFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	MSDFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~MSDFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual std::vector<bool> getRequirements() const;
};


