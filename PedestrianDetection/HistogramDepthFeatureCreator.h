#pragma once
#include "IFeatureCreator.h"


class HistogramDepthFeatureCreator : public IFeatureCreator
{

private:

public:
	HistogramDepthFeatureCreator(std::string& name);
	virtual ~HistogramDepthFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};


