#include "CenSurEFeatureCreator.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"



CenSurEFeatureCreator::CenSurEFeatureCreator(std::string& name, int clusterSize, IFeatureCreator::Target target)
	: VariableNumberFeatureCreator(name, clusterSize), target(target) {
}


CenSurEFeatureCreator::~CenSurEFeatureCreator() {
}



std::vector<FeatureVector> CenSurEFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {


	cv::Ptr<cv::xfeatures2d::StarDetector> detector = cv::xfeatures2d::StarDetector::create(5, 5, 5, 10, 1);

	std::vector<cv::KeyPoint> keypoints;
	if (target == IFeatureCreator::Target::Depth) {
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


std::vector<bool> CenSurEFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
			target == IFeatureCreator::Target::Depth,
			target == IFeatureCreator::Target::Thermal
	};
}