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


	int getNumberOfFeatures(int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize) {
		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		// histograms are patchSize x patchSize in an image and contain binSize values
		// to normalize light and shadows histograms are normalized with their adjacent ones, forming an 2x2 rectangle or 4 histograms together
		if (l2normalize)
			return (nrOfCellsHeight - 1) * (nrOfCellsWidth - 1) * binSize * 4;
		else
			return (nrOfCellsHeight) * (nrOfCellsWidth)* binSize;
	}


	cv::Mat explainHOGFeature(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool full360, bool l2normalize) {

		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		int nrOfFeatures = getNumberOfFeatures(imgWidth, imgHeight, patchSize, binSize, l2normalize);
		int to = offset + nrOfFeatures;

		cv::Mat explanation(cv::Size(imgWidth, imgHeight), CV_32FC1, cv::Scalar(0));


		std::function<void(int, int, int, int)> func = [&](int featureIndex, int patchX, int patchY, int binIndex) -> void {
			int x = patchX * patchSize;
			int y = patchY * patchSize;
			double angle = 1.0 * binIndex / binSize * (full360 ? 2 * CV_PI : CV_PI) + CV_PI / 2;
			double weight = weightPerFeature[offset + featureIndex];
			if (weight > 0) {
				int cx = x + patchSize / 2;
				int cy = y + patchSize / 2;

				double radius = patchSize / 2;
				double vx = cos(angle) * radius;
				double vy = sin(angle) * radius;

				cv::line(explanation, cv::Point(floor(cx - vx), floor(cy - vy)), cv::Point(floor(cx + vx), floor(cy + vy)), cv::Scalar(weight), 1);
			}
		};


		int idx = 0;
		if (l2normalize) {
			for (int y = 0; y < nrOfCellsHeight - 1; y++) {
				for (int x = 0; x < nrOfCellsWidth - 1; x++) {

					std::vector<int> sorted;

					for (int k = 0; k < binSize; k++)
						sorted.push_back(k);
					std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool { return weightPerFeature[offset + idx + a] < weightPerFeature[offset + idx + b];  });
					for (int k = 0; k < binSize; k++) {
						func(idx, x, y, sorted[k]);
						idx++;
					}
					sorted.clear();


					for (int k = 0; k < binSize; k++)
						sorted.push_back(k);
					std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool { return weightPerFeature[offset + idx + a] < weightPerFeature[offset + idx + b];  });
					for (int k = 0; k < binSize; k++) {
						func(idx, x + 1, y, sorted[k]);
						idx++;
					}
					sorted.clear();


					for (int k = 0; k < binSize; k++)
						sorted.push_back(k);
					std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool { return weightPerFeature[offset + idx + a] < weightPerFeature[offset + idx + b];  });
					for (int k = 0; k < binSize; k++) {
						func(idx, x , y+1, sorted[k]);
						idx++;
					}
					sorted.clear();


					for (int k = 0; k < binSize; k++)
						sorted.push_back(k);
					std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool { return weightPerFeature[offset + idx + a] < weightPerFeature[offset + idx + b];  });
					for (int k = 0; k < binSize; k++) {
						func(idx, x + 1, y+1, sorted[k]);
						idx++;
					}
					sorted.clear();

				}
			}
		}
		else {
			for (int y = 0; y < nrOfCellsHeight; y++) {
				for (int x = 0; x < nrOfCellsWidth; x++) {

					std::vector<int> sorted;

					for (int k = 0; k < binSize; k++)
						sorted.push_back(k);
					std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool { return weightPerFeature[offset + idx + a] < weightPerFeature[offset + idx + b];  });
					for (int k = 0; k < binSize; k++) {
						func(idx, x + 1, y, sorted[k]);
						idx++;
					}
					sorted.clear();
				}
			}
		}
		return explanation;
	}


	HistogramResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, bool createImage, bool l2normalize) {

		cv::Mat mat = img.clone();

		if (mat.type() != CV_32FC1)
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

							// linear interpolation [ u ][ 1-u ]
						float u = (tEnd - anglePixel) / (tEnd - tBegin);
						histogram[bin1] += magPixel * u;
						histogram[bin2] += magPixel * (1 - u);


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

		HistogramResult result;
		result.width = nrOfCellsWidth - 1;
		result.height = nrOfCellsHeight - 1;
		result.data = newcells;
		result.hogImage = hog;
		return result;
	}



	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize) {
		cv::Mat hog;

		bool drawHistograms = true;
		hog = cv::Mat(mat.rows, mat.cols, CV_8UC3, cv::Scalar(0));

		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {

				double cx = x * patchSize + patchSize / 2;
				double cy = y * patchSize + patchSize / 2;
				Histogram hist = cells[y][x];


				double maxVal = *std::max_element(hist.begin(), hist.end());
				if (maxVal > 0) {

					float barSize = 1.0 * (patchSize - 2) / binSize;
					if (drawHistograms)
						cv::rectangle(hog, cv::Rect(x*patchSize + 1, y * patchSize + 1, patchSize - 2, patchSize - 2), cv::Scalar(10, 10, 10), -1);
					for (int i = 0; i < binSize; i++) {
						double angle = ((i + 0.5) / binSize) * CV_PI + CV_PI / 2; // + 90° so it aligns perpendicular to gradient
						double val = hist[i] / maxVal;
						double radius = patchSize / 2 * val;
						double vx = cos(angle) * radius;
						double vy = sin(angle) * radius;

						if (!drawHistograms)
							cv::line(hog, cv::Point(floor(cx - vx), floor(cy - vy)), cv::Point(floor(cx + vx), floor(cy + vy)), cv::Scalar(0, 0, 255));

						double height = val * (patchSize - 2);
						double yOffset = y*patchSize + 1;
						cv::Rect2f r = cv::Rect2f(x*patchSize + 1 + i*barSize, yOffset + (patchSize - 2) - height, barSize, height);
						if (drawHistograms)
							cv::rectangle(hog, r, cv::Scalar(0, 255, 255), -1);
					}
				}

			}
		}
		return hog;
	}


	HistogramResult getHistogramsOfDepthDifferences(cv::Mat& img, int patchSize, int binSize, bool createImage, bool l2normalize) {

		cv::Mat depth = img; // depth is already in CV_32FC1
		//cvtColor(img, depth, CV_BGR2GRAY);
		//depth.convertTo(depth, CV_32FC1, 1 / 255.0, 0);

		cv::Mat magnitude(img.size(), CV_32FC1, cv::Scalar(0));
		cv::Mat angle(img.size(), CV_32FC1, cv::Scalar(0));


		for (int j = 1; j < depth.rows - 1; j++)
		{
			for (int i = 1; i < depth.cols - 1; i++)
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
		return getHistogramsOfX(magnitude, angle, patchSize, binSize, createImage, l2normalize);
	}

	HistogramResult getHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize) {


		double max = 1;

		int nrOfCellsWidth = weights.cols / patchSize;
		int nrOfCellsHeight = weights.rows / patchSize;

		std::vector<std::vector<Histogram>> cells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram(binSize, 0)));

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram& histogram = cells[y][x];

				for (int l = 0; l < patchSize; l++) {
					for (int k = 0; k < patchSize; k++) {

						float anglePixel = normalizedBinningValues.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));
						double weight = weights.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));

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

						float u = (tEnd - anglePixel) / (tEnd - tBegin);
						histogram[bin1] += weight * u;
						histogram[bin2] += weight * (1 - u);
					}
				}
			}
		}

		std::vector<std::vector<Histogram>> newcells = getL2NormalizationOverLargerPatch(cells, nrOfCellsWidth, nrOfCellsHeight, binSize, l2normalize);

		cv::Mat hog;
		if (createImage) {
			cv::Mat m(weights.rows, weights.cols, CV_32FC1);
			hog = createHoGImage(m, cells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);
		}
		HistogramResult result;
		result.width = nrOfCellsWidth - 1;
		result.height = nrOfCellsHeight - 1;
		result.data = newcells;
		result.hogImage = hog;
		return result;

	}


	cv::Mat explain2DHOGFeature(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize) {

		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		int idx = 0;

		int nrOfFeatures = getNumberOfFeatures(imgWidth, imgHeight, patchSize, binSize, l2normalize);
		int to = offset + nrOfFeatures;

		cv::Mat explanation(cv::Size(imgWidth, imgHeight), CV_32FC1, cv::Scalar(0));

		for (int featureIndex = 0; featureIndex < nrOfFeatures; featureIndex++) {
			// now normalize all histograms

			int patchX = -1;
			int patchY = -1;
			int binIndex = -1;
			int binIndex2 = -1;

			bool found = false;

			for (int y = 0; y < nrOfCellsHeight && !found; y++) {
				for (int x = 0; x < nrOfCellsWidth && !found; x++) {
					for (int k = 0; k < binSize && !found; k++) {
						for (int l = 0; l < binSize && !found; l++) {
							if (featureIndex == idx) {
								patchX = x;
								patchY = y;
								binIndex = k;
								binIndex2 = l;
								found = true;
							}
							idx++;
						}
					}
				}
			}

			if (found) {
				int x = patchX * patchSize;
				int y = patchY * patchSize;

				double weight = weightPerFeature[offset + featureIndex];
				cv::rectangle(explanation, cv::Rect(x + 1 + binSize * 2, y + 1 + binSize * 2, 2, 2), cv::Scalar(weight), -1);
			}
		}
		return explanation;
	}


	HistogramResult get2DHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize) {


		double max = 1;

		int nrOfCellsWidth = weights.cols / patchSize;
		int nrOfCellsHeight = weights.rows / patchSize;

		std::vector<std::vector<Histogram2D>> cells(nrOfCellsHeight, std::vector<Histogram2D>(nrOfCellsWidth, Histogram2D(binSize, 0)));

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram2D& histogram = cells[y][x];

				for (int l = 0; l < patchSize; l++) {
					for (int k = 0; k < patchSize; k++) {

						cv::Vec2f anglePixel = normalizedBinningValues.at<cv::Vec2f>(cv::Point(x * patchSize + k, y * patchSize + l));
						double weight = weights.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));


						// distribute based on angle
						// 15 in [0-20] = 0.25 * 15 for bin 0 and 0.75 * 15 for bin 1
						cv::Vec2f valBins = anglePixel / max * binSize;
						if (valBins[0] >= binSize) valBins[0] = binSize - 1;
						if (valBins[1] >= binSize) valBins[1] = binSize - 1;

						int bin1x = floor(valBins[0]);
						int bin2x = (bin1x + 1) % binSize;

						int bin1y = floor(valBins[1]);
						int bin2y = (bin1y + 1) % binSize;

						// (t - t_begin) / (t_end - t_begin)
						// 15 - 0 / (20-0) = 0.75
						// (t_end - t) / (t_end - t_begin)
						// 20 - 15 / (20-0) = 0.25
						// yay for computergraphics triangular scheme

						float tBeginX = bin1x == 0 ? 0 : bin1x * max / binSize;
						float tEndX = bin2x == 0 ? max : bin2x * max / binSize;

						float tBeginY = bin1y == 0 ? 0 : bin1y * max / binSize;
						float tEndY = bin2y == 0 ? max : bin2y * max / binSize;

						double u = (tEndX - anglePixel[0]) / (tEndX - tBeginX);
						double v = (tEndY - anglePixel[1]) / (tEndY - tBeginY);


						histogram[bin1x][bin1y] += weight * u * (v);
						histogram[bin2x][bin1y] += weight * (1 - u) * (v);
						histogram[bin1x][bin2y] += weight * u * (1 - v);
						histogram[bin2x][bin2y] += weight * (1 - u) * (1 - v);
						//histogram[bin2] += weight * (anglePixel[0] - tBeginX) / (tEndX - tBeginX);
					}
				}
			}
		}
		std::vector<std::vector<Histogram>> flattenedCells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram()));
		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {
				flattenedCells[y][x] = cells[y][x].flatten();
			}
		}

		//	std::vector<std::vector<Histogram>> newcells = getL2NormalizationOverLargerPatch(flattenedCells, nrOfCellsWidth, nrOfCellsHeight, binSize, l2normalize);

		cv::Mat hog;
		if (createImage) {
			cv::Mat m(weights.rows, weights.cols, CV_32FC1);
			hog = createHoGImage(m, flattenedCells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);
		}
		HistogramResult result;
		result.width = nrOfCellsWidth - 1;
		result.height = nrOfCellsHeight - 1;
		result.data = flattenedCells;
		result.hogImage = hog;
		return result;
	}

}