#include "HistogramOfOrientedGradients.h"
#include <vector>
#include <iostream>

namespace hog {
	std::vector<std::vector<Histogram>> getL2NormalizationOverLargerPatch(const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, bool l2normalize) {
		std::vector<std::vector<Histogram>> newcells(nrOfCellsHeight - 1, std::vector<Histogram>(nrOfCellsWidth - 1, Histogram(binSize * 4, 0))); // histogram of elements per cell
																																				  // now normalize all histograms
		for (int y = 0; y < nrOfCellsHeight - 1; y++) {
			for (int x = 0; x < nrOfCellsWidth - 1; x++) {

				auto& dstHistogram = newcells[y][x];
				int idx = 0;

				for (int i = 0; i < cells[y][x].size(); i++)
					dstHistogram[idx++] = cells[y][x][i];

				for (int i = 0; i < cells[y][x + 1].size(); i++)
					dstHistogram[idx++] = cells[y][x + 1][i];

				for (int i = 0; i < cells[y + 1][x].size(); i++)
					dstHistogram[idx++] = cells[y + 1][x][i];

				for (int i = 0; i < cells[y + 1][x + 1].size(); i++)
					dstHistogram[idx++] = cells[y + 1][x + 1][i];

				/*	double max = *std::max_element(dstHistogram.begin(), dstHistogram.end());
				if (max > 0) {
				for (int i = 0; i < dstHistogram.size(); i++)
				dstHistogram[i] /= max;
				}*/
				if (l2normalize) {
					double sum = 0;
					for (int i = 0; i < dstHistogram.size(); i++)
						sum += dstHistogram[i] * dstHistogram[i];

					double norm = sqrt(sum);
					if (norm > 0) {
						for (int i = 0; i < dstHistogram.size(); i++)
							dstHistogram[i] /= norm;
					}
				}
			}
		}
		return newcells;
	}


	int getNumberOfFeatures(int imgWidth, int imgHeight, int patchSize, int binSize) {
		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		// histograms are patchSize x patchSize in an image and contain binSize values
		// to normalize light and shadows histograms are normalized with their adjacent ones, forming an 2x2 rectangle or 4 histograms together
		return (nrOfCellsHeight - 1) * (nrOfCellsWidth - 1) * binSize * 4;
	}

	std::string explainHoGFeature(int featureIndex, double featureValue, int imgWidth, int imgHeight, int patchSize, int binSize) {
		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		int idx = 0;
		// now normalize all histograms
		for (int y = 0; y < nrOfCellsHeight - 1; y++) {
			for (int x = 0; x < nrOfCellsWidth - 1; x++) {

				for (int k = 0; k < binSize; k++)
				{

					if (featureIndex == idx)
						return "HoG @ (" + std::to_string(x) + "," + std::to_string(y) + ") with angle " + std::to_string((180.0 / binSize)*k);

					idx++;
				}
			}
		}
		return "";
	}


	HoGResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, bool createImage, bool l2normalize) {

		cv::Mat mat = img.clone();
		cv::cvtColor(img, mat, cv::COLOR_RGB2GRAY);

		cv::Mat gx, gy;
		cv::Sobel(mat, gx, CV_32F, 1, 0, 1);
		cv::Sobel(mat, gy, CV_32F, 0, 1, 1);


		int nrOfCellsWidth = mat.cols / patchSize;
		int nrOfCellsHeight = mat.rows / patchSize;

		std::vector<std::vector<Histogram>> cells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram(binSize, 0)));

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram& histogram = cells[y][x];

				for (int l = 0; l < patchSize; l++) {
					for (int k = 0; k < patchSize; k++) {
						float sx = gx.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));
						float sy = gy.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));

						// calculate the correct unoriented angle: e.g. PI/4 and 3* PI / 4 are the same
						// this will map the angles on a [0-PI] range
						double anglePixel = atan2(sy, sx);
						anglePixel = anglePixel > 0 ? abs(anglePixel) : abs(CV_PI - abs(anglePixel)); // CV_PI is not that accurate, must abs!

						double magPixel = sqrt((sx*sx) + (sy*sy));

						// distribute based on angle
						// 15 in [0-20] = 0.25 * 15 for bin 0 and 0.75 * 15 for bin 1
						double valBins = anglePixel / CV_PI * binSize;
						if (valBins >= binSize) valBins = binSize - 1;

						int bin1 = floor(valBins);
						int bin2 = (bin1 + 1) % binSize;

						// (t - t_begin) / (t_end - t_begin)
						// 15 - 0 / (20-0) = 0.75
						// (t_end - t) / (t_end - t_begin)
						// 20 - 15 / (20-0) = 0.25
						// yay for computergraphics triangular scheme

						float tBegin = bin1 == 0 ? 0 : bin1 * CV_PI / binSize;
						float tEnd = bin2 == 0 ? CV_PI : bin2 * CV_PI / binSize;
						/*	if (tBegin == tEnd) {
								tEnd += CV_PI / binSize;
							}*/

						histogram[bin1] += magPixel * (tEnd - anglePixel) / (tEnd - tBegin);
						histogram[bin2] += magPixel * (anglePixel - tBegin) / (tEnd - tBegin);


					}
				}

				//// rescaling to [0-1]
				//double max = *std::max_element(histogram.begin(), histogram.end());
				//if (max > 0) {
				//	for (int i = 0; i < histogram.size(); i++)
				//		histogram[i] /= max;
				//}


				// cell x,y -> pixel range [x * cellSize-x * cellSize + cellSize], ...
			}
		}



		std::vector<std::vector<Histogram>> newcells = getL2NormalizationOverLargerPatch(cells, nrOfCellsWidth, nrOfCellsHeight, binSize, l2normalize);

		if (l2normalize) {
			// L2 normalization
			for (int y = 0; y < nrOfCellsHeight; y++) {

				for (int x = 0; x < nrOfCellsWidth; x++) {
					Histogram& histogram = cells[y][x];

					double sum = 0;
					for (int i = 0; i < histogram.size(); i++)
						sum += histogram[i] * histogram[i];
					double norm = sqrt(sum);
					if (norm > 0) {
						for (int i = 0; i < histogram.size(); i++)
							histogram[i] /= norm;
					}
				}
			}
		}

		cv::Mat hog;
		if (createImage)
			hog = createHoGImage(mat, cells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);

		HoGResult result;
		result.width = nrOfCellsWidth - 1;
		result.height = nrOfCellsHeight - 1;
		result.data = newcells;
		result.hogImage = hog;
		return result;
	}



	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize) {
		cv::Mat hog;

		hog = mat.clone();
		cv::cvtColor(hog, hog, CV_GRAY2BGR);

		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {

				double cx = x * patchSize + patchSize / 2;
				double cy = y * patchSize + patchSize / 2;
				Histogram hist = cells[y][x];


				double maxVal = *std::max_element(hist.begin(), hist.end());
				if (maxVal > 0) {
					for (int i = 0; i < binSize; i++) {
						double angle = ((i + 0.5) / binSize) * CV_PI + CV_PI / 2; // + 90° so it aligns perpendicular to gradient
						double val = hist[i] / maxVal;

						double vx = cos(angle) * patchSize / 2 * val;
						double vy = sin(angle) * patchSize / 2 * val;

						cv::line(hog, cv::Point(floor(cx - vx), floor(cy - vy)), cv::Point(floor(cx + vx), floor(cy + vy)), cv::Scalar(0, 0, 255));
					}
				}

			}
		}
		return hog;
	}

	HoGResult getHistogramsOfX(cv::Mat& imgValues, cv::Mat& imgBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize) {


		double max = 1;

		int nrOfCellsWidth = imgValues.cols / patchSize;
		int nrOfCellsHeight = imgValues.rows / patchSize;

		std::vector<std::vector<Histogram>> cells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram(binSize, 0)));

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram& histogram = cells[y][x];

				for (int l = 0; l < patchSize; l++) {
					for (int k = 0; k < patchSize; k++) {

						float anglePixel = imgBinningValues.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));
						double magPixel = imgValues.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));

						// distribute based on angle
						// 15 in [0-20] = 0.25 * 15 for bin 0 and 0.75 * 15 for bin 1
						double valBins = anglePixel / max * binSize;
						if (valBins >= binSize) valBins = binSize - 1;

						int bin1 = floor(valBins);
						int bin2 = (bin1 + 1) % binSize;

						// (t - t_begin) / (t_end - t_begin)
						// 15 - 0 / (20-0) = 0.75
						// (t_end - t) / (t_end - t_begin)
						// 20 - 15 / (20-0) = 0.25
						// yay for computergraphics triangular scheme

						float tBegin = bin1 == 0 ? 0 : bin1 * max / binSize;
						float tEnd = bin2 == 0 ? max : bin2 * max / binSize;

						histogram[bin1] += magPixel * (tEnd - anglePixel) / (tEnd - tBegin);
						histogram[bin2] += magPixel * (anglePixel - tBegin) / (tEnd - tBegin);
					}
				}
			}
		}

		std::vector<std::vector<Histogram>> newcells = getL2NormalizationOverLargerPatch(cells, nrOfCellsWidth, nrOfCellsHeight, binSize, l2normalize);

		cv::Mat hog;
		if (createImage) {
			cv::Mat m(imgValues.rows, imgValues.cols, CV_32FC1);
			hog = createHoGImage(m, cells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);
		}
		HoGResult result;
		result.width = nrOfCellsWidth - 1;
		result.height = nrOfCellsHeight - 1;
		result.data = newcells;
		result.hogImage = hog;
		return result;

	}
}