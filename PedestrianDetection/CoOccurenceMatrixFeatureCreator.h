#pragma once
#include "IFeatureCreator.h"

class CoOccurenceMatrixFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 16;

public:
	CoOccurenceMatrixFeatureCreator(std::string& name, int patchSize = 8, int binSize = 16);
	virtual ~CoOccurenceMatrixFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

};

