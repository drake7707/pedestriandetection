#include "HONVFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HONVFeatureCreator::HONVFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


HONVFeatureCreator::~HONVFeatureCreator()
{
}


int HONVFeatureCreator::getNumberOfFeatures() const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;
	return nrOfCellsWidth * nrOfCellsHeight * (binSize * binSize);
}


cv::Mat HONVFeatureCreator::buildAngleMat(cv::Mat depth) const {
	cv::Mat angleMat(depth.rows, depth.cols, CV_32FC2, cv::Scalar(0));

	//depth = depth * 255;
	for (int y = 0; y < depth.rows; y++)
	{
		for (int x = 0; x < depth.cols; x++)
		{


			float r = x + 1 >= depth.cols ? depth.at<float>(y, x) : depth.at<float>(y, x + 1);
			float l = x - 1 < 0 ? depth.at<float>(y, x) : depth.at<float>(y, x - 1);

			float b = y + 1 >= depth.rows ? depth.at<float>(y, x) : depth.at<float>(y + 1, x);
			float t = y - 1 < 0 ? depth.at<float>(y, x) : depth.at<float>(y - 1, x);


			float dzdx = (r - l) / 2.0;
			float dzdy = (b - t) / 2.0;

			cv::Vec3f d(-dzdx, -dzdy, 1.0f);

			cv::Vec3f n = normalize(d);

			double azimuth = atan2(-d[1], -d[0]); // -pi -> pi
			if (azimuth < 0)
				azimuth += 2 * CV_PI;

			double zenith = atan(sqrt(d[1] * d[1] + d[0] * d[0]));

			cv::Vec2f angles(azimuth / (2 * CV_PI), (zenith + CV_PI / 2) / CV_PI);
			angleMat.at<cv::Vec2f>(y, x) = angles;

			//normals.at<Vec3f>(y, x) = n;
		}
	}
	return angleMat;
}

std::unique_ptr<IPreparedData> HONVFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {

	cv::Mat angle = buildAngleMat(depthScale);
	IntegralHistogram2D hist = hog::prepare2DDataForHistogramsOfX(cv::Mat(depthScale.rows, depthScale.cols, CV_32FC1, cv::Scalar(1)), angle, binSize);
	IntHist2DPreparedData* data = new IntHist2DPreparedData();
	data->integralHistogram = hist;
	return std::unique_ptr<IPreparedData>(data);
}

FeatureVector HONVFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {

	cv::Mat angleMat;


	const IntHist2DPreparedData* hogData = static_cast<const IntHist2DPreparedData*>(preparedData);

	hog::HistogramResult result;
	if (hogData == nullptr) {
		angleMat = buildAngleMat(depth);
		result = hog::get2DHistogramsOfX(cv::Mat(depth.rows, depth.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, binSize, false, roi, nullptr, refWidth, refHeight);
	}
	else {
		result = hog::get2DHistogramsOfX(cv::Mat(depth.rows, depth.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, binSize, false, roi, &(hogData->integralHistogram), refWidth, refHeight);
	}

	return result.getFeatureArray();
}


cv::Mat HONVFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	return hog::explain2DHOGFeature(offset, weightPerFeature, refWidth, refHeight, patchSize, binSize, false);
}

std::vector<bool> HONVFeatureCreator::getRequirements() const {
	return{ false, true, false };
}