#include "SIFTFeatureCreator.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"



SIFTFeatureCreator::SIFTFeatureCreator(std::string& name, int clusterSize, bool onDepth)
	: VariableNumberFeatureCreator(name, clusterSize), onDepth(onDepth) {
}


SIFTFeatureCreator::~SIFTFeatureCreator() {
}



std::vector<FeatureVector> SIFTFeatureCreator::getVariableNumberFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {


	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();

	std::vector<cv::KeyPoint> keypoints;

	cv::Mat descriptors;
	if (onDepth) {

		cv::Mat d8U;
		depth.convertTo(d8U, CV_8UC1, 255.0, 0);
		detector->detectAndCompute(d8U, cv::noArray(), keypoints, descriptors);

	}
	else
		detector->detectAndCompute(rgb, cv::noArray(), keypoints, descriptors);

	std::vector<FeatureVector> features;
	//for (int j = 0; j < descriptors.rows; j++)
	//{
	//	FeatureVector v;
	//	for (int i = 0; i < descriptors.cols; i++)
	//	{
	//		v.push_back(descriptors.at<float>(j, i));
	//	}
	//	
	//	auto& k = keypoints[j];

	//	v.push_back(k.pt.x);
	//	v.push_back(k.pt.y);
	//	v.push_back(k.size);
	//	v.push_back(k.octave);
	//	v.push_back(k.angle);
	//	v.push_back(k.class_id);
	//	features.push_back(v);
	//}
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


std::vector<bool> SIFTFeatureCreator::getRequirements() const {
	return{ !onDepth, onDepth, false };
}