#include "LBPFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"



LBPFeatureCreator::LBPFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), target(target)
{
}


LBPFeatureCreator::~LBPFeatureCreator()
{
}


int LBPFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, false);
}

FeatureVector LBPFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {

	cv::Mat img;
	if (target == IFeatureCreator::Target::Depth) {
		cv::cvtColor(depth, img, CV_BGR2GRAY);
	}
	else
		cv::cvtColor(rgb, img, CV_BGR2GRAY);


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

cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, occurrencePerFeature, refWidth, refHeight, patchSize, binSize, false, false);
}

std::vector<bool> LBPFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
		target == IFeatureCreator::Target::Depth,
		target == IFeatureCreator::Target::Thermal
	};
}