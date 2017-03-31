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

cv::Mat HDDFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, occurrencePerFeature, refWidth, refHeight, patchSize, binSize, true, true);
}

std::vector<bool> HDDFeatureCreator::getRequirements() const {
	return{ false, true, false };
}