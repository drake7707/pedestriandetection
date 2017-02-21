#pragma once
#include "IFeatureCreator.h"
class HOGRGBFeatureCreator : public IFeatureCreator
{

private:


	int patchSize = 8;
	int binSize = 9;

public:
	HOGRGBFeatureCreator();
	virtual ~HOGRGBFeatureCreator();

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

