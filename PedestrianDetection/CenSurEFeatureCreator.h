#pragma once
#include "VariableNumberFeatureCreator.h"
class CenSurEFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	CenSurEFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~CenSurEFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual std::vector<bool> getRequirements() const;
};


