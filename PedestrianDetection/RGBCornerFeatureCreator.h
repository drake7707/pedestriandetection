#pragma once
#include "IFeatureCreator.h"
#include "VariableNumberFeatureCreator.h"

class RGBCornerFeatureCreator : public VariableNumberFeatureCreator
{

private:
	int patchSize = 8;
	int refWidth = 64;
	int refHeight = 128;
public:
	RGBCornerFeatureCreator();
	virtual ~RGBCornerFeatureCreator();

	std::vector<FeatureVector> getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const;

};

