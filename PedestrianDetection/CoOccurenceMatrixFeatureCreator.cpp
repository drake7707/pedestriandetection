#include "CoOccurenceMatrixFeatureCreator.h"
#include "CoOccurenceMatrix.h"


CoOccurenceMatrixFeatureCreator::CoOccurenceMatrixFeatureCreator(std::string& name, int patchSize, int binSize)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize)
{
}


CoOccurenceMatrixFeatureCreator::~CoOccurenceMatrixFeatureCreator()
{
}


int CoOccurenceMatrixFeatureCreator::getNumberOfFeatures() const {
	return binSize*binSize;
}

FeatureVector CoOccurenceMatrixFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	int nrOfCellsWidth = rgb.cols / patchSize;
	int nrOfCellsHeight = rgb.rows / patchSize;

	cv::Mat img = rgb.clone();
	cv::cvtColor(img, img, CV_BGR2HLS);
	img.convertTo(img, CV_32FC1, 1.0);
	cv::Mat hsl[3];
	cv::split(img, hsl);

	cv::Mat hue = hsl[0] / 180;
	auto cells = getCoOccurenceMatrix(hue, patchSize, binSize);

	FeatureVector v;
	v.reserve(nrOfCellsWidth * nrOfCellsHeight * binSize*binSize);

	for (int y = 0; y < cells.size(); y++)
	{
		for (int x = 0; x < cells[y].size(); x++)
		{
			for (int l = 0; l < binSize; l++)
			{
				for (int k = 0; k < binSize; k++)
				{
					v.push_back(cells[y][x][l][k]);
				}
			}
		}
	}

	return v;
}

cv::Mat CoOccurenceMatrixFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
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
					histogram[y][x] += occurrencePerFeature[offset + idx];
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