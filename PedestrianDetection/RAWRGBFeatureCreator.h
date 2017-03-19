#pragma once
#include "IFeatureCreator.h"

class RAWRGBFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 16;

public:
	RAWRGBFeatureCreator(std::string& name);
	virtual ~RAWRGBFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

};

