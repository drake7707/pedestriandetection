#include "SURFFeatureCreator.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"



SURFFeatureCreator::SURFFeatureCreator()
	: VariableNumberFeatureCreator(std::string("RGBSURF"), 100) {
}


SURFFeatureCreator::~SURFFeatureCreator() {
}



std::vector<FeatureVector> SURFFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth) const {


	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
	std::vector<cv::KeyPoint> keypoints;

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
