#include "CoOccurenceMatrix.h"

namespace coocc {

	IntegralHistogram2D prepareData(cv::Mat& img, int binSize) {

		IntegralHistogram2D hist;

		hist.create(img.cols - 1, img.rows - 1, binSize, [&](int x, int y, std::vector<std::vector<cv::Mat>>& ihist) -> void {
			float val = floor(img.at<float>(y, x) * (binSize - 1));

			float valRight = floor(img.at<float>(y + 0, x + 1) * (binSize - 1));
			float valBottom = floor(img.at<float>(y + 1, x + 0) * (binSize - 1));

			// increment
			ihist[valRight][val].at<float>(y, x) = ihist[valRight][val].at<float>(y, x) + 1;
			ihist[valBottom][val].at<float>(y, x) = ihist[valBottom][val].at<float>(y, x) + 1;
		});

		return hist;
	}

	std::vector<std::vector<CoOccurrenceMatrix>> getCoOccurenceMatrix(cv::Mat& img, int imgWidth, int imgHeight, int patchSize, int binSize, cv::Rect& iHistROI, const IntegralHistogram2D* iHist) {
		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;
		
		std::vector<std::vector<CoOccurrenceMatrix>> cells(nrOfCellsHeight, std::vector<CoOccurrenceMatrix>(nrOfCellsWidth, CoOccurrenceMatrix(binSize, std::vector<float>(binSize, 0))));

		if (iHist == nullptr) {
			for (int y = 0; y < nrOfCellsHeight; y++) {

				for (int x = 0; x < nrOfCellsWidth; x++) {

					cv::Mat patch = img(cv::Rect(x * patchSize, y * patchSize, patchSize, patchSize));
					cells[y][x] = getCoOccurenceMatrixOfPatch(patch, binSize);
				}
			}
		}
		else {
			for (int y = 0; y < nrOfCellsHeight; y++) {
				for (int x = 0; x < nrOfCellsWidth; x++) {
					// width & height are -1 because that's the upper bounds (see getCoOccurenceMatrixOfPatch for loops)
					cells[y][x] = iHist->calculateHistogramIntegral(iHistROI.x  + x  * patchSize, iHistROI.y + y * patchSize, patchSize - 1, patchSize - 1);
				}
			}
		}
		return cells;
	}

	CoOccurrenceMatrix getCoOccurenceMatrixOfPatch(cv::Mat& img, int binSize) {
		// combined C0,1 and C1,0 2D histogram
		CoOccurrenceMatrix coOccurrenceMatrix(binSize, std::vector<float>(binSize, 0));

		/*for (int l = 0; l < binSize; l++)
		{
			for (int k = 0; k < binSize; k++)
			{*/
		for (int j = 0; j < img.rows - 1; j++)
		{
			for (int i = 0; i < img.cols - 1; i++)
			{
				float val = floor(img.at<float>(j, i) * (binSize - 1));

				float valRight = floor(img.at<float>(j + 0, i + 1) * (binSize - 1));
				float valBottom = floor(img.at<float>(j + 1, i + 0) * (binSize - 1));

				//if (val == k && valRight == l)
				coOccurrenceMatrix[valRight][val]++;

				//if (val == k && valBottom == l)
				coOccurrenceMatrix[valBottom][val]++;
			}
		}
		/*}
	}*/

		return coOccurrenceMatrix;
	}

	cv::Mat getCoOccurenceMatrixImage(int width, int height, CoOccurrenceMatrix& matrix) {

		int orgWidth = width;
		int orgHeight = height;

		int bins = matrix.size();

		width = width < bins ? bins : width;
		height = height < bins ? bins : height;


		float max = std::numeric_limits<float>().min();
		for (int j = 0; j < bins; j++)
		{
			for (int i = 0; i < bins; i++)
			{
				if (max < matrix[j][i])
					max = matrix[j][i];
			}
		}


		cv::Mat img(width, height, CV_8UC3, cv::Scalar(0, 0, 0));

		int cellSize = img.cols / bins;
		for (int j = 0; j < bins; j++)
		{
			for (int i = 0; i < bins; i++)
			{
				float alpha = 1.0f * matrix[j][i] / max;
				// white (255,255,255) -> blue (255,0,0)
				cv::rectangle(img, cv::Rect(i * cellSize, j * cellSize, cellSize, cellSize), cv::Scalar(255, 255 - alpha * 255, 255 - alpha * 255), -1);
			}
		}

		if (width != orgWidth || height != orgHeight)
			cv::resize(img, img, cv::Size(orgWidth, orgHeight));

		return img;
	}

	cv::Mat createFullCoOccurrenceMatrixImage(cv::Mat baseImage, std::vector<std::vector<CoOccurrenceMatrix>>& cells, int patchSize) {
		int nrOfCellsWidth = baseImage.cols / patchSize;
		int nrOfCellsHeight = baseImage.rows / patchSize;

		cv::Mat img = baseImage.clone();
		if (img.channels() == 1) {
			if (img.type() == CV_32FC1) {
				img.convertTo(img, CV_8UC1, 255, 0);
				cv::cvtColor(img, img, CV_GRAY2BGR);
			}
			else
				cv::cvtColor(img, img, CV_GRAY2BGR);
		}

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {


				cv::Mat& patch = img(cv::Rect(x * patchSize + 1, y * patchSize + 1, patchSize - 2, patchSize - 2));
				cv::Mat matrixImage = getCoOccurenceMatrixImage(patchSize - 2, patchSize - 2, cells[y][x]);
				matrixImage.copyTo(patch);
			}
		}

		return img;
	}

}