#pragma once
#include "IFeatureCreator.h"
class HOGFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;
	bool onDepth;

public:
	HOGFeatureCreator(std::string& name, bool onDepth, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HOGFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};
