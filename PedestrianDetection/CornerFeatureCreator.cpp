#include "CornerFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



CornerFeatureCreator::CornerFeatureCreator(std::string& name, bool onDepth, int clusterSize)
	: VariableNumberFeatureCreator(name, clusterSize), onDepth(onDepth) {
}


CornerFeatureCreator::~CornerFeatureCreator()
{
}

std::vector<FeatureVector> CornerFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	cv::Mat gray;
	if (onDepth) {

		cv::Mat d8U;
		depth.convertTo(d8U, CV_8UC1,255.0, 0);
		gray = d8U;
	}
	else
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