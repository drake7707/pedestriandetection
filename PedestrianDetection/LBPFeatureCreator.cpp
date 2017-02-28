#include "LBPFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"



LBPFeatureCreator::LBPFeatureCreator(std::string& name, bool onDepth, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), onDepth(onDepth)
{
}


LBPFeatureCreator::~LBPFeatureCreator()
{
}


int LBPFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, false);
}

FeatureVector LBPFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	cv::Mat img;
	if(onDepth)
		cv::cvtColor(rgb, img, CV_BGR2GRAY);
	else
		cv::cvtColor(depth, img, CV_BGR2GRAY);

	cv::Mat lbp = Algorithms::OLBP(img);
	lbp.convertTo(lbp, CV_32FC1, 1 / 255.0, 0);

	cv::Mat padded;
	int padding = 1;
	padded.create(img.rows, img.cols, lbp.type());
	padded.setTo(cv::Scalar::all(0));
	lbp.copyTo(padded(Rect(padding, padding, lbp.cols, lbp.rows)));


	auto& result = hog::getHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), padded, patchSize, binSize, false, false);


	return result.getFeatureArray();
}

std::string LBPFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return getName() + " TODO";
}