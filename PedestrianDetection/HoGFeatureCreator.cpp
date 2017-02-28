#include "HoGFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGFeatureCreator::HOGFeatureCreator(std::string& name, bool onDepth, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), onDepth(onDepth)
{
}


HOGFeatureCreator::~HOGFeatureCreator()
{
}


int HOGFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}

FeatureVector HOGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	hog::HistogramResult result;
	if(onDepth)
		result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	else
		result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);

	return result.getFeatureArray();
}

std::string HOGFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return getName() + " " + hog::explainHOGFeature(featureIndex, featureValue, refWidth, refHeight, patchSize, binSize);
}