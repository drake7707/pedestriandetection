#include "HOGRGBFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGRGBFeatureCreator::HOGRGBFeatureCreator()
{
}


HOGRGBFeatureCreator::~HOGRGBFeatureCreator()
{
}


FeatureVector HOGRGBFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	auto result = getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	return result.getFeatureArray(false);
}