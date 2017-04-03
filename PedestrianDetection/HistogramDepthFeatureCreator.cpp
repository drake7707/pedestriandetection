#include "HistogramDepthFeatureCreator.h"


HistogramDepthFeatureCreator::HistogramDepthFeatureCreator(std::string& name)
	: IFeatureCreator(name) {
}


HistogramDepthFeatureCreator::~HistogramDepthFeatureCreator()
{
}



int HistogramDepthFeatureCreator::getNumberOfFeatures() const {
	return 26;
}

FeatureVector HistogramDepthFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {

	//	cv::normalize(depth, img, 0, 1, cv::NormTypes::NORM_MINMAX);

	// normalize but ignore 0 img values and set these 0 values to the min.
		// this ignores the holes

	cv::Mat img = depth;
	float max = std::numeric_limits<float>().min();
	float min = std::numeric_limits<float>().max();
	for (int j = 0; j < img.rows; j++)
	{
		for (int i = 0; i < img.cols; i++)
		{
			float val = img.at<float>(j, i);
			if (val > 0) {
				if (val > max) max = val;
				if (val < min) min = val;
			}
		}
	}
	for (int j = 0; j < img.rows; j++)
	{
		for (int i = 0; i < img.cols; i++)
		{
			float val = img.at<float>(j, i);
			if (val > 0) {
				img.at<float>(j, i) = (val - min) / (max - min);
			}
			else
				img.at<float>(j, i) = min;
		}
	}


	FeatureVector v(26, 0);
	for (int j = 0; j < img.rows; j++)
	{
		for (int i = 0; i < img.cols; i++)
		{
			int idx = floor(img.at<float>(j, i) * v.size());
			if (idx >= v.size())
				idx = v.size() - 1;
			v[idx]++;
		}
	}

	return v;
}


cv::Mat HistogramDepthFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	// todo
	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));
	return explanation;
}


std::vector<bool> HistogramDepthFeatureCreator::getRequirements() const {
	return{ false, true, false };
}