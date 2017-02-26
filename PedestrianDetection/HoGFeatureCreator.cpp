#include "HoGFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HoGFeatureCreator::HoGFeatureCreator(std::string& name, bool onDepth, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), onDepth(onDepth)
{
}


HoGFeatureCreator::~HoGFeatureCreator()
{
}


int HoGFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize);
}

FeatureVector HoGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	hog::HoGResult result;
	if(onDepth)
		result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);
	else
		result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);

	return result.getFeatureArray();
}

std::string HoGFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return getName() + " " + hog::explainHoGFeature(featureIndex, featureValue, refWidth, refHeight, patchSize, binSize);
}