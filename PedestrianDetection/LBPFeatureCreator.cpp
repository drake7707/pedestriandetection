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

	cv::Mat img;
	cv::cvtColor(rgbScale, img, CV_BGR2GRAY);

	buildWeightAndBinningValues(img, weights, binningValues);

	IntegralHistogram hist = hog::prepareDataForHistogramsOfX(weights, binningValues, binSize);
	IntHistPreparedData* data = new IntHistPreparedData();
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

	cv::Mat& roi = padded(Rect(padding, padding, lbp.cols, lbp.rows));
	lbp.copyTo(roi);

	weights = cv::Mat(img.size(), CV_32FC1, cv::Scalar(1));
	binningValues = padded;
}


FeatureVector LBPFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {
	cv::Mat img;
	cv::cvtColor(rgb, img, CV_BGR2GRAY);


	cv::Mat weights;
	cv::Mat binningValues;

	hog::HistogramResult result;

	const IntHistPreparedData* hogData = static_cast<const IntHistPreparedData*>(preparedData);
	if (hogData == nullptr) {
		buildWeightAndBinningValues(img, weights, binningValues);
		result = hog::getHistogramsOfX(weights, binningValues, patchSize, binSize, false, false, roi, nullptr, refWidth, refHeight);
	}
	else {
		result = hog::getHistogramsOfX(weights, binningValues, patchSize, binSize, false, false, roi, &(hogData->integralHistogram), refWidth, refHeight);
	}

	return result.getFeatureArray();
}

cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, false, false);
}

std::vector<bool> LBPFeatureCreator::getRequirements() const {
	return{ true, false,false };
}