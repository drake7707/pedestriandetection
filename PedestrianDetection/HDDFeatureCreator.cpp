#include "HDDFeatureCreator.h"

HDDFeatureCreator::HDDFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HDDFeatureCreator::~HDDFeatureCreator()
{
}


int HDDFeatureCreator::getNumberOfFeatures() const {
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true);
}



void HDDFeatureCreator::buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const {
	cv::Mat depth = img; // depth is already in CV_32FC1
						 //cvtColor(img, depth, CV_BGR2GRAY);
						 //depth.convertTo(depth, CV_32FC1, 1 / 255.0, 0);

	magnitude = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));
	angle = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));


	for (int j = 0; j < depth.rows; j++)
	{
		for (int i = 0; i < depth.cols; i++)
		{

			float r = i + 1 >= depth.cols ? depth.at<float>(j, i) : depth.at<float>(j, i + 1);
			float l = i - 1 < 0 ? depth.at<float>(j, i) : depth.at<float>(j, i - 1);

			float b = j + 1 >= depth.rows ? depth.at<float>(j, i) : depth.at<float>(j + 1, i);
			float t = j - 1 < 0 ? depth.at<float>(j, i) : depth.at<float>(j - 1, i);

			float dx = (r - l) / 2;
			float dy = (b - t) / 2;

			double anglePixel = atan2(dy, dx);
			// don't limit to 0-pi, but instead use 0-2pi range
			anglePixel = (anglePixel < 0 ? anglePixel + 2 * CV_PI : anglePixel) + CV_PI / 2;
			if (anglePixel > 2 * CV_PI) anglePixel -= 2 * CV_PI;

			double magPixel = sqrt((dx*dx) + (dy*dy));
			magnitude.at<float>(j, i) = magPixel;
			angle.at<float>(j, i) = anglePixel / (2 * CV_PI);
		}
	}
}

hog::HistogramResult HDDFeatureCreator::getHistogramsOfDepthDifferences(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, const IPreparedData* preparedData, bool createImage, bool l2normalize) const {
	cv::Mat magnitude;
	cv::Mat angle;

	const IntHistPreparedData* hogData = static_cast<const IntHistPreparedData*>(preparedData);
	if (hogData == nullptr) {
		buildMagnitudeAndAngle(img, magnitude, angle);
	}

	return hog::getHistogramsOfX(magnitude, angle, patchSize, binSize, createImage, l2normalize, roi, hogData == nullptr ? nullptr : &(hogData->integralHistogram), refWidth, refHeight);
}

std::unique_ptr<IPreparedData> HDDFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {
	cv::Mat magnitude;
	cv::Mat angle;

	buildMagnitudeAndAngle(depthScale, magnitude, angle);
	IntegralHistogram hist = hog::prepareDataForHistogramsOfX(magnitude, angle, binSize);
	IntHistPreparedData* data = new IntHistPreparedData();
	data->integralHistogram = hist;
	return std::unique_ptr<IPreparedData>(data);
}

FeatureVector HDDFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {

	hog::HistogramResult result;
	result = getHistogramsOfDepthDifferences(depth, patchSize, binSize, roi, preparedData, false, true);

	return result.getFeatureArray();
}

cv::Mat HDDFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explainHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, true, true);
}

std::vector<bool> HDDFeatureCreator::getRequirements() const {
	return{ false, true, false };
}