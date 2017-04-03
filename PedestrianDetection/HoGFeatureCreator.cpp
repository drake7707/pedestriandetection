#include "HOGFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGFeatureCreator::HOGFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), target(target)
{
}


HOGFeatureCreator::~HOGFeatureCreator()
{
}


int HOGFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}

FeatureVector HOGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {

	hog::HistogramResult result;
	if (target == IFeatureCreator::Target::Depth)
		result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);
	else if (target == IFeatureCreator::Target::Thermal)
		result = hog::getHistogramsOfOrientedGradient(thermal, patchSize, binSize, false, true);
	else
		result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);

	return result.getFeatureArray();
}

cv::Mat HOGFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, occurrencePerFeature, refWidth, refHeight, patchSize, binSize, false, true);
}


std::vector<bool> HOGFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
		target == IFeatureCreator::Target::Depth,
		target == IFeatureCreator::Target::Thermal
	};
}