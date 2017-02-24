#pragma once
#include "IFeatureCreator.h"
class RGBCornerFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int refWidth = 64;
	int refHeight = 128;
public:
	RGBCornerFeatureCreator(int patchSize = 8, int refWidth = 64, int refHeight = 128);
	virtual ~RGBCornerFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

