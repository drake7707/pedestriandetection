#pragma once
#include "IFeatureCreator.h"
#include "HoGFeatureCreator.h"

class HOGHistogramVarianceFeatureCreator : public HOGFeatureCreator
{

public:
	HOGHistogramVarianceFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HOGHistogramVarianceFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat HOGHistogramVarianceFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

};

