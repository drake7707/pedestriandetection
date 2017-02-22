#include "HOGRGBFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGRGBFeatureCreator::HOGRGBFeatureCreator()
{
}


HOGRGBFeatureCreator::~HOGRGBFeatureCreator()
{
}


int HOGRGBFeatureCreator::getNumberOfFeatures() const {
	return getNumberOfFeatures();
}

FeatureVector HOGRGBFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	auto result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	return result.getFeatureArray();
}

std::string HOGRGBFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return "RGB " + hog::explainHoGFeature(featureIndex, featureValue, refWidth, refHeight, patchSize, binSize);
}