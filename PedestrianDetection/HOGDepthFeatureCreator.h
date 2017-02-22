#pragma once
#include "IFeatureCreator.h"

class HOGDepthFeatureCreator : public IFeatureCreator
{
private:


	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;

public:
	HOGDepthFeatureCreator();
	virtual ~HOGDepthFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

