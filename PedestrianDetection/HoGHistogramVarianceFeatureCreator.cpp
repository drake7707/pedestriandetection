#include "HoGHistogramVarianceFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGHistogramVarianceFeatureCreator::HOGHistogramVarianceFeatureCreator(std::string& name, bool onDepth, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), onDepth(onDepth) {
}


HOGHistogramVarianceFeatureCreator::~HOGHistogramVarianceFeatureCreator()
{
}


int HOGHistogramVarianceFeatureCreator::getNumberOfFeatures() const {
	// each rectangle of 2x2 histograms, each containing binSize elements is now replaced by a single S2 value
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize) / (binSize * 4);
}

FeatureVector HOGHistogramVarianceFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	hog::HoGResult result;
	if(onDepth)
		result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);
	else
		result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);

	return result.getHistogramVarianceFeatures();
}

std::string HOGHistogramVarianceFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;

	int x = featureIndex % (nrOfCellsWidth - 1);
	int y = featureIndex / (nrOfCellsWidth - 1);

	return getName() + " at (" + std::to_string(x) + "," + std::to_string(y) + ")";
}