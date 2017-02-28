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

FeatureVector HistogramDepthFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {


	FeatureVector v(26, 0);
	for (int j = 0; j < depth.rows; j++)
	{
		for (int i = 0; i < depth.cols; i++)
		{
			int idx = floor(depth.at<float>(j, i) * v.size());
			if (idx >= v.size())
				idx = v.size() - 1;
			v[idx]++;
		}
	}

	return v;
}

std::string HistogramDepthFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return "Histogram depth TODO";
}