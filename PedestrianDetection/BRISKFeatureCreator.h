#pragma once
#include "VariableNumberFeatureCreator.h"
class BRISKFeatureCreator :
	public VariableNumberFeatureCreator
{

private:
	bool onDepth;
public:
	BRISKFeatureCreator(std::string& name, int clusterSize, bool onDepth);
	virtual ~BRISKFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual std::vector<bool> getRequirements() const;
};


