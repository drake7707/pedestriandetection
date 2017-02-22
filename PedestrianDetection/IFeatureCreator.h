#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"

class IFeatureCreator
{
public:
	IFeatureCreator();
	virtual ~IFeatureCreator();

	virtual int getNumberOfFeatures() const = 0;

	virtual FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const = 0;

	virtual std::string explainFeature(int featureIndex, double featureValue) const = 0;
};

