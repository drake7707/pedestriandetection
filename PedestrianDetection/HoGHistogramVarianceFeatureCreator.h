#pragma once
#include "IFeatureCreator.h"
class HOGHistogramVarianceFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;
	bool onDepth;
public:
	HOGHistogramVarianceFeatureCreator(std::string& name, bool onDepth, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HOGHistogramVarianceFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat HOGHistogramVarianceFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;
};

