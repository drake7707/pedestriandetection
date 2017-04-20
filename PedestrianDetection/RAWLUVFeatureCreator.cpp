#include "RAWLUVFeatureCreator.h"


RAWLUVFeatureCreator::RAWLUVFeatureCreator(std::string& name, int refWidth, int refHeight)
	: IFeatureCreator(name), refWidth(refWidth), refHeight(refHeight)
{
}


RAWLUVFeatureCreator::~RAWLUVFeatureCreator()
{
}


int RAWLUVFeatureCreator::getNumberOfFeatures() const {
	return refWidth*refHeight * 3;
}

FeatureVector RAWLUVFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {
	FeatureVector v;
	v.reserve(rgb.cols * rgb.rows * 3);

	cv::Mat scaledRGB;
	rgb.convertTo(scaledRGB, CV_32FC3);
	scaledRGB *= 1. / 255;
	cv::Mat luv;
	cv::cvtColor(scaledRGB, luv, CV_BGR2Luv);

	for (int j = 0; j < luv.rows; j++) {
		for (int i = 0; i < luv.cols; i++) {
			cv::Vec3f pixel = luv.at<cv::Vec3f>(j, i);
			v.push_back(pixel[0]);
			v.push_back(pixel[1]);
			v.push_back(pixel[2]);
		}
	}

	return v;
}

cv::Mat RAWLUVFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));

	int nrOfFeatures = getNumberOfFeatures();

	int idx = 0;
	for (int j = 0; j < refHeight; j++) {
		for (int i = 0; i < refWidth; i++) {

			double w1 = occurrencePerFeature[offset + idx];
			idx++;
			double w2 = occurrencePerFeature[offset + idx];
			idx++;
			double w3 = occurrencePerFeature[offset + idx];
			idx++;

			explanation.at<float>(j, i) = w1 + w2 + w3;
		}
	}


	return explanation;
}

std::vector<bool> RAWLUVFeatureCreator::getRequirements() const {
	return{ true,false, false };
}