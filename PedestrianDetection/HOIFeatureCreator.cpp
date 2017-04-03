#include "HOIFeatureCreator.h"


HOIFeatureCreator::HOIFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HOIFeatureCreator::~HOIFeatureCreator()
{
}


int HOIFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, false);
}

FeatureVector HOIFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {
	cv::normalize(thermal, thermal, 0, 1, cv::NormTypes::NORM_MINMAX);

	auto hogResult = hog::getHistogramsOfX(cv::Mat(thermal.rows, thermal.cols, CV_32FC1, cv::Scalar(1)), thermal, patchSize, binSize, false, true);

	return hogResult.getFeatureArray();
}

cv::Mat HOIFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, occurrencePerFeature, refWidth, refHeight, patchSize, binSize, false, false);
}

std::vector<bool> HOIFeatureCreator::getRequirements() const {
	return{ false, false, true };
}