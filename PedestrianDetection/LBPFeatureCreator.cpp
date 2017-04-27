#include "LBPFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"



LBPFeatureCreator::LBPFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


LBPFeatureCreator::~LBPFeatureCreator()
{
}


int LBPFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, false);
}

std::unique_ptr<IPreparedData> LBPFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {

	cv::Mat weights;
	cv::Mat binningValues;
	buildWeightAndBinningValues(rgbScale, weights, binningValues);


	IntegralHistogram hist = hog::prepareDataForHistogramsOfX(weights, binningValues, binSize);
	HOG1DPreparedData* data = new HOG1DPreparedData();
	data->integralHistogram = hist;
	return std::unique_ptr<IPreparedData>(data);
}


void LBPFeatureCreator::buildWeightAndBinningValues(cv::Mat& img, cv::Mat& weights, cv::Mat& binningValues) const {
	cv::Mat mat = img.clone();

	cv::Mat lbp = Algorithms::OLBP(img);
	lbp.convertTo(lbp, CV_32FC1, 1 / 255.0, 0);

	cv::Mat padded;
	int padding = 1;
	padded.create(img.rows, img.cols, lbp.type());
	padded.setTo(cv::Scalar::all(0));
	lbp.copyTo(padded(Rect(padding, padding, lbp.cols, lbp.rows)));

	weights = cv::Mat(img.size(), CV_32FC1, cv::Scalar(1));
	binningValues = padded;
}


FeatureVector LBPFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {
	cv::Mat img;
	cv::cvtColor(rgb, img, CV_BGR2GRAY);


	cv::Mat weights;
	cv::Mat binningValues;

	const HOG1DPreparedData* hogData = static_cast<const HOG1DPreparedData*>(preparedData);
	if (hogData == nullptr) {
		buildWeightAndBinningValues(img, weights, binningValues);
	}
	auto& result = hog::getHistogramsOfX(weights, binningValues, patchSize, binSize, false, false, roi, hogData == nullptr ? nullptr : &(hogData->integralHistogram), refWidth, refHeight);


	return result.getFeatureArray();
}

cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, false, false);
}

std::vector<bool> LBPFeatureCreator::getRequirements() const {
	return{ true, false,false };
}