#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"

class IFeatureCreator
{

private:
	std::string name;

public:
	IFeatureCreator(std::string& name);
	virtual ~IFeatureCreator();

	std::string getName() const;

	virtual int getNumberOfFeatures() const = 0;

	virtual FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const = 0;

	virtual std::string explainFeature(int featureIndex, double featureValue) const = 0;
};

