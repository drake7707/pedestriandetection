#include "CoOccurenceMatrixFeatureCreator.h"

CoOccurenceMatrixFeatureCreator::CoOccurenceMatrixFeatureCreator(std::string& name, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight)
{
}


CoOccurenceMatrixFeatureCreator::~CoOccurenceMatrixFeatureCreator()
{
}


int CoOccurenceMatrixFeatureCreator::getNumberOfFeatures() const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;

	return nrOfCellsWidth * nrOfCellsHeight * binSize*binSize;
}

// Not much of a performance gain and uses too much RAM
//std::unique_ptr<IPreparedData> CoOccurenceMatrixFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const {
//
//	cv::Mat img = rgbScale.clone();
//	cv::cvtColor(img, img, CV_BGR2HLS);
//	img.convertTo(img, CV_32FC1, 1.0);
//	cv::Mat hsl[3];
//	cv::split(img, hsl);
//
//	cv::Mat hue = hsl[0] / 180;
//
//	IntegralHistogram2D hist = coocc::prepareData(hue, binSize);
//
//	IntHist2DPreparedData* data = new IntHist2DPreparedData();
//	data->integralHistogram = hist;
//	return std::unique_ptr<IPreparedData>(data);
//}


FeatureVector CoOccurenceMatrixFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const {

	int nrOfCellsWidth = rgb.cols / patchSize;
	int nrOfCellsHeight = rgb.rows / patchSize;

	//	std::vector<std::vector<coocc::CoOccurrenceMatrix>> cells;
	cv::Mat coOccurrence;

	if (preparedData == nullptr) {

		cv::Mat img = rgb.clone();
		cv::cvtColor(img, img, CV_BGR2HLS);
		img.convertTo(img, CV_32FC1, 1.0);
		cv::Mat hsl[3];
		cv::split(img, hsl);

		cv::Mat hue = hsl[0] / 180;
		coOccurrence = coocc::getCoOccurenceMatrix(hue, hue.cols, hue.rows, patchSize, binSize, roi, nullptr);
	}
	else {
		cv::Mat hue;
		const IntHist2DPreparedData* intHistPreparedData = static_cast<const IntHist2DPreparedData*>(preparedData);
		coOccurrence = coocc::getCoOccurenceMatrix(hue, rgb.cols, rgb.rows, patchSize, binSize, roi, &(intHistPreparedData->integralHistogram));
	}
	FeatureVector v;
	v.reserve(nrOfCellsWidth * nrOfCellsHeight * binSize*binSize);

	for (int y = 0; y < nrOfCellsHeight; y++)
	{
		for (int x = 0; x < nrOfCellsWidth; x++)
		{
			cv::Mat coOccurrenceOfPatch = coOccurrence(cv::Rect(x * binSize, y * binSize, binSize, binSize));

			for (int l = 0; l < binSize; l++)
			{
				for (int k = 0; k < binSize; k++)
				{
					v.push_back(coOccurrenceOfPatch.at<float>(k, l));
					//v.push_back(cells[y][x][l][k]);
				}
			}
		}
	}

	return v;
}

cv::Mat CoOccurenceMatrixFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;

	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));


	auto histogram = std::vector<std::vector<float>>(nrOfCellsHeight, std::vector<float>(nrOfCellsWidth, 0));

	int idx = 0;
	for (int y = 0; y < nrOfCellsHeight; y++)
	{
		for (int x = 0; x < nrOfCellsWidth; x++)
		{
			for (int l = 0; l < binSize; l++)
			{
				for (int k = 0; k < binSize; k++)
				{
					histogram[y][x] += weightPerFeature[offset + idx];
					idx++;
				}
			}
		}
	}


	for (int y = 0; y < nrOfCellsHeight; y++)
	{
		for (int x = 0; x < nrOfCellsWidth; x++)
		{
			int offsetX = x * patchSize;
			int offsetY = y * patchSize;
			cv::rectangle(explanation, cv::Rect(offsetX, offsetY, patchSize, patchSize), cv::Scalar(histogram[y][x]), -1);
		}
	}

	return explanation;
}


std::vector<bool> CoOccurenceMatrixFeatureCreator::getRequirements() const {
	return{ true, false, false };
}