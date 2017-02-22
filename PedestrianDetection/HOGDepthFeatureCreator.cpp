#include "HOGDepthFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"


HOGDepthFeatureCreator::HOGDepthFeatureCreator(int patchSize, int binSize, int refWidth, int refHeight)
	: patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HOGDepthFeatureCreator::~HOGDepthFeatureCreator()
{
}

int HOGDepthFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize);
}

FeatureVector HOGDepthFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	auto result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);
	return result.getFeatureArray();
}

std::string HOGDepthFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return "Depth " + hog::explainHoGFeature(featureIndex, featureValue, refWidth, refHeight, patchSize, binSize);
}