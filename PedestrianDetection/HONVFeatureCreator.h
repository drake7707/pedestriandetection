#pragma once
#include "IFeatureCreator.h"
class HONVFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;

public:
	HONVFeatureCreator(std::string& name, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HONVFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

