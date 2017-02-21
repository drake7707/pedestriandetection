#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"

class IFeatureCreator
{
public:
	IFeatureCreator();
	virtual ~IFeatureCreator();

	virtual FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const = 0;
};

