#include "IFeatureCreator.h"



IFeatureCreator::IFeatureCreator(std::string& name) : name(name) {
}


IFeatureCreator::~IFeatureCreator() {
}


std::vector<IPreparedData*> IFeatureCreator::buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const {
	return std::vector<IPreparedData*>();
}

std::string IFeatureCreator::getName() const {
	return name;
}

