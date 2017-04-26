#include "HOIFeatureCreator.h"


HOIFeatureCreator::HOIFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HOIFeatureCreator::~HOIFeatureCreator()
{
}


int HOIFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}


std::unique_ptr<IPreparedData> HOIFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {

	IntegralHistogram hist = hog::prepareDataForHistogramsOfX(cv::Mat(thermalScale.rows, thermalScale.cols, CV_32FC1, cv::Scalar(1)), thermalScale, binSize);
	HOG1DPreparedData* data = new HOG1DPreparedData();
	data->integralHistogram = hist;

	return std::unique_ptr<IPreparedData>(data);
}


FeatureVector HOIFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {


	const HOG1DPreparedData* hogData = static_cast<const HOG1DPreparedData*>(preparedData);

	auto hogResult = hog::getHistogramsOfX(cv::Mat(thermal.rows, thermal.cols, CV_32FC1, cv::Scalar(1)), thermal, patchSize, binSize, false, true, roi, hogData == nullptr ? nullptr : &(hogData->integralHistogram), refWidth, refHeight);

	return hogResult.getFeatureArray();
}

cv::Mat HOIFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, false, true);
}

std::vector<bool> HOIFeatureCreator::getRequirements() const {
	return{ false, false, true };
}