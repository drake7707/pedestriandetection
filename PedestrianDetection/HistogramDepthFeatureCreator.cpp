#include "HistogramDepthFeatureCreator.h"


HistogramDepthFeatureCreator::HistogramDepthFeatureCreator()
{
}


HistogramDepthFeatureCreator::~HistogramDepthFeatureCreator()
{
}



int HistogramDepthFeatureCreator::getNumberOfFeatures() const {
	return 26;
}

FeatureVector HistogramDepthFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	cv::Mat depthNorm;
	cv::normalize(depth, depthNorm, 0, 255, cv::NormTypes::NORM_MINMAX);

	FeatureVector v(26, 0);
	for (int j = 0; j < depth.rows; j++)
	{
		for (int i = 0; i < depth.cols; i++)
		{
			int idx = depthNorm.at<uchar>(j, i) / 10;
			v[idx]++;
		}
	}

	return v;
}

std::string HistogramDepthFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return "Histogram depth TODO";
}