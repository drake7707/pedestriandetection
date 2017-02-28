#include "FASTFeatureCreator.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"



FASTFeatureCreator::FASTFeatureCreator(std::string& name, int clusterSize, bool onDepth)
	: VariableNumberFeatureCreator(name, clusterSize), onDepth(onDepth) {
}


FASTFeatureCreator::~FASTFeatureCreator() {
}



std::vector<FeatureVector> FASTFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const {


	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();

	std::vector<cv::KeyPoint> keypoints;
	if (onDepth) {
		cv::Mat d8U;
		depth.convertTo(d8U, CV_8UC1, 255.0, 0);
		detector->detect(d8U, keypoints);
	}
	else
		detector->detect(rgb, keypoints);

	std::vector<FeatureVector> features;

	for (auto& k : keypoints) {

		FeatureVector v;
		v.push_back(k.pt.x);
		v.push_back(k.pt.y);
		v.push_back(k.size);
		v.push_back(k.octave);
		v.push_back(k.angle);
		v.push_back(k.class_id);
		features.push_back(v);
	}

	return features;
}
