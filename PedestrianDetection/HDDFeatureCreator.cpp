#include "HDDFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HDDFeatureCreator::HDDFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HDDFeatureCreator::~HDDFeatureCreator()
{
}


int HDDFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}

FeatureVector HDDFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	hog::HistogramResult result;
	result = hog::getHistogramsOfDepthDifferences(depth, patchSize, binSize, false, true);
	return result.getFeatureArray();
}

std::string HDDFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return getName() + " TODO";
}