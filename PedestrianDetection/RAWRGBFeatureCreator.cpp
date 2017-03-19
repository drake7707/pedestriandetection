#include "RAWRGBFeatureCreator.h"


RAWRGBFeatureCreator::RAWRGBFeatureCreator(std::string& name)
	: IFeatureCreator(name)
{
}


RAWRGBFeatureCreator::~RAWRGBFeatureCreator()
{
}


int RAWRGBFeatureCreator::getNumberOfFeatures() const {
	return binSize*binSize;
}

FeatureVector RAWRGBFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	
	FeatureVector v;
	v.reserve(rgb.cols * rgb.rows * 3);

	for (int j = 0; j < rgb.rows; j++) {
		for (int i = 0; i < rgb.cols; i++) {
			cv::Vec3b pixel = rgb.at<cv::Vec3b>(j, i);
			v.push_back(pixel[0]);
			v.push_back(pixel[1]);
			v.push_back(pixel[2]);
		}
	}

	return v;
}

std::string RAWRGBFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return getName() + " TODO";
}
