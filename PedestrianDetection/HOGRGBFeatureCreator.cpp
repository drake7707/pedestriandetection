#include "HOGRGBFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGRGBFeatureCreator::HOGRGBFeatureCreator(int patchSize, int binSize, int refWidth, int refHeight)
	: patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HOGRGBFeatureCreator::~HOGRGBFeatureCreator()
{
}


int HOGRGBFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize);
}

FeatureVector HOGRGBFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	auto result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	return result.getFeatureArray();
}

std::string HOGRGBFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return "RGB " + hog::explainHoGFeature(featureIndex, featureValue, refWidth, refHeight, patchSize, binSize);
}