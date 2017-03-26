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

FeatureVector HONVFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {

	cv::Mat angleMat(depth.rows, depth.cols, CV_32FC3, cv::Scalar(0));

	//depth = depth * 255;
	for (int y = 1; y < depth.rows; y++)
	{
		for (int x = 1; x < depth.cols; x++)
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

			cv::Vec3f angles(azimuth / (2 * CV_PI), (zenith + CV_PI / 2) / CV_PI, 1);
			angleMat.at<cv::Vec3f>(y, x) = angles;

			//normals.at<Vec3f>(y, x) = n;
		}
	}

	auto& result = hog::get2DHistogramsOfX(cv::Mat(depth.rows, depth.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, binSize, false, false);

	return result.getFeatureArray();
}


cv::Mat HONVFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	return hog::explain2DHOGFeature(offset, weightPerFeature, occurrencePerFeature, refWidth, refHeight, patchSize, binSize,false);
}