#include "HOGRGBHistogramVarianceFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGRGBHistogramVarianceFeatureCreator::HOGRGBHistogramVarianceFeatureCreator(int patchSize, int binSize, int refWidth, int refHeight)
	: patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HOGRGBHistogramVarianceFeatureCreator::~HOGRGBHistogramVarianceFeatureCreator()
{
}


int HOGRGBHistogramVarianceFeatureCreator::getNumberOfFeatures() const {
	// each rectangle of 2x2 histograms, each containing binSize elements is now replaced by a single S2 value
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize) / (binSize * 4);
}

FeatureVector HOGRGBHistogramVarianceFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	auto result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	return result.getHistogramVarianceFeatures();
}

std::string HOGRGBHistogramVarianceFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;

	int x = featureIndex % (nrOfCellsWidth - 1);
	int y = featureIndex / (nrOfCellsWidth - 1);

	return "S2 HoG RGB Variance at (" + std::to_string(x) + "," + std::to_string(y) + ")";
}