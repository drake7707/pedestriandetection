#include "RGBCornerFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



RGBCornerFeatureCreator::RGBCornerFeatureCreator(int patchSize, int refWidth, int refHeight)
	: patchSize(patchSize), refWidth(refWidth), refHeight(refHeight), VariableNumberFeatureCreator(std::string("RGBCorner"),10) {
}


RGBCornerFeatureCreator::~RGBCornerFeatureCreator()
{
}

std::vector<FeatureVector> RGBCornerFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	cv::Mat gray;
	cv::cvtColor(rgb, gray, CV_BGR2GRAY);


	std::vector<cv::Point2f> corners;
	int maxCorners = 1000;
	float qualityLevel = 0.01;
	float minDistance = 5;

	cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

	std::vector<FeatureVector> features;

	for (auto& p : corners) {
		FeatureVector v(2, 0);
		v[0] = p.x;
		v[1] = p.y;

		features.push_back(v);
	}
	return features;
}