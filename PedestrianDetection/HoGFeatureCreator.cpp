#include "HOGFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGFeatureCreator::HOGFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), target(target)
{
}


HOGFeatureCreator::~HOGFeatureCreator()
{
}


int HOGFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}

void HOGFeatureCreator::buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const {
	cv::Mat mat = img.clone();

	if (mat.type() != CV_32FC1)
		cv::cvtColor(img, mat, cv::COLOR_RGB2GRAY);

	cv::Mat gx, gy;
	cv::Sobel(mat, gx, CV_32F, 1, 0, 1);
	cv::Sobel(mat, gy, CV_32F, 0, 1, 1);

	magnitude = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));
	angle = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));

	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			float sx = gx.at<float>(j, i);
			float sy = gy.at<float>(j, i);

			// calculate the correct unoriented angle: e.g. PI/4 and 3* PI / 4 are the same
			// this will map the angles on a [0-PI] range

			double anglePixel = atan2(sy, sx);
			anglePixel = anglePixel > 0 ? abs(anglePixel) : abs(CV_PI - abs(anglePixel)); // CV_PI is not that accurate, must abs!

			double magPixel = sqrt((sx*sx) + (sy*sy));

			magnitude.at<float>(j, i) = magPixel;
			angle.at<float>(j, i) = anglePixel / CV_PI; // 0-180° -> 0-1
		}
	}
}


hog::HistogramResult HOGFeatureCreator::getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, const IPreparedData* preparedData, bool createImage, bool l2normalize) const {

	cv::Mat magnitude;
	cv::Mat angle;

	const HOG1DPreparedData* hogData = dynamic_cast<const HOG1DPreparedData*>(preparedData);
	if (hogData == nullptr) {
		buildMagnitudeAndAngle(img, magnitude, angle);
	}

	return hog::getHistogramsOfX(magnitude, angle, patchSize, binSize, createImage, l2normalize, roi, hogData == nullptr ? nullptr : &(hogData->integralHistogram), refWidth, refHeight);
}

std::unique_ptr<IPreparedData> HOGFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {
	cv::Mat magnitude;
	cv::Mat angle;

	if (target == IFeatureCreator::Target::Depth) {
		buildMagnitudeAndAngle(depthScale, magnitude, angle);
	}
	else if (target == IFeatureCreator::Target::Thermal) {
		buildMagnitudeAndAngle(thermalScale, magnitude, angle);
	}
	else {
		buildMagnitudeAndAngle(rgbScale, magnitude, angle);
	}

	IntegralHistogram hist = hog::prepareDataForHistogramsOfX(magnitude, angle, binSize);
	HOG1DPreparedData* data = new HOG1DPreparedData();
	data->integralHistogram = hist;
	return std::unique_ptr<IPreparedData>(data);
}

FeatureVector HOGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {

	hog::HistogramResult result;
	if (target == IFeatureCreator::Target::Depth)
		result = getHistogramsOfOrientedGradient(depth, patchSize, binSize, roi, preparedData, false, true);
	else if (target == IFeatureCreator::Target::Thermal)
		result = getHistogramsOfOrientedGradient(thermal, patchSize, binSize, roi, preparedData, false, true);
	else
		result = getHistogramsOfOrientedGradient(rgb, patchSize, binSize, roi, preparedData, false, true);

	return result.getFeatureArray();
}

cv::Mat HOGFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, false, true);
}


std::vector<bool> HOGFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
		target == IFeatureCreator::Target::Depth,
		target == IFeatureCreator::Target::Thermal
	};
}