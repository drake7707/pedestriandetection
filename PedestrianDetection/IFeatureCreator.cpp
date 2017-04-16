#include "IFeatureCreator.h"



IFeatureCreator::IFeatureCreator(std::string& name) : name(name) {
}


IFeatureCreator::~IFeatureCreator() {
}


std::unique_ptr<IPreparedData> IFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {
	return nullptr;
}

std::string IFeatureCreator::getName() const {
	return name;
}

