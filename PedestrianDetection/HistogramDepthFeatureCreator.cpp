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


cv::Mat HistogramDepthFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	// todo
	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));
	return explanation;
}