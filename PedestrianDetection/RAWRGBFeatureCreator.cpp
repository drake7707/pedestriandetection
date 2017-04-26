#include "RAWRGBFeatureCreator.h"


RAWRGBFeatureCreator::RAWRGBFeatureCreator(std::string& name, int refWidth, int refHeight)
	: IFeatureCreator(name), refWidth(refWidth), refHeight(refHeight)
{
}


RAWRGBFeatureCreator::~RAWRGBFeatureCreator()
{
}


int RAWRGBFeatureCreator::getNumberOfFeatures() const {
	return refWidth*refHeight * 3;
}

FeatureVector RAWRGBFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {
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

cv::Mat RAWRGBFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));

	int nrOfFeatures = getNumberOfFeatures();

	int idx = 0;
	for (int j = 0; j < refHeight; j++) {
		for (int i = 0; i < refWidth; i++) {

			double w1 = weightPerFeature[offset + idx];
			idx++;
			double w2 = weightPerFeature[offset + idx];
			idx++;
			double w3 = weightPerFeature[offset + idx];
			idx++;

			explanation.at<float>(j, i) = w1 + w2 + w3;
		}
	}


	return explanation;
}

std::vector<bool> RAWRGBFeatureCreator::getRequirements() const {
	return { true,false, false };
}