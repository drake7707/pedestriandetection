#include "RGBCornerFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



RGBCornerFeatureCreator::RGBCornerFeatureCreator(int patchSize, int refWidth, int refHeight)
	: patchSize(patchSize), refWidth(refWidth), refHeight(refHeight) {
}


RGBCornerFeatureCreator::~RGBCornerFeatureCreator()
{
}


int RGBCornerFeatureCreator::getNumberOfFeatures() const {
	int nrOfRows = refHeight / patchSize;
	int nrOfCols = refWidth / patchSize;

	return nrOfRows * nrOfCols;
}

FeatureVector RGBCornerFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	cv::Mat gray;
	cv::cvtColor(rgb, gray, CV_BGR2GRAY);


	std::vector<cv::Point2f> corners;
	int maxCorners = 1000;
	float qualityLevel = 0.01;
	float minDistance = 5;

	cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);


	int nrOfRows = refHeight / patchSize;
	int nrOfCols = refWidth / patchSize;
	std::vector<std::vector<float>> countCornerPoints(nrOfRows, std::vector<float>(nrOfCols, 0));


	
	for (auto& p : corners) {
		int x = floor(p.x / patchSize);
		int y = floor(p.y / patchSize);
		countCornerPoints[y][x]++;
	}


	FeatureVector v;
	for (int j = 0; j < nrOfRows; j++)
	{
		for (int i = 0; i < nrOfCols; i++)
		{
			v.push_back(countCornerPoints[j][i] / corners.size());
		}
	}
	return v;
}

std::string RGBCornerFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	int nrOfCols = refWidth / patchSize;

	int x = featureIndex % nrOfCols;
	int y = featureIndex / nrOfCols;
	return "#Corners on (" + std::to_string(x) + "," + std::to_string(y) + ")";
}