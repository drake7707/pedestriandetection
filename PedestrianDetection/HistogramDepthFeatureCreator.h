#pragma once
#include "IFeatureCreator.h"


class HistogramDepthFeatureCreator : public IFeatureCreator
{

private:
	
public:
	HistogramDepthFeatureCreator();
	virtual ~HistogramDepthFeatureCreator();

	int getNumberOfFeatures() const;
	std::string explainFeature(int featureIndex, double featureValue) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

