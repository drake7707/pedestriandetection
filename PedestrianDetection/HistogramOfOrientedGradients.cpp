#include "HistogramOfOrientedGradients.h"


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

	cv::Mat explainHOGFeature(int offset, std::vector<float>& weightPerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool full360, bool l2normalize) {

		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		int nrOfFeatures = getNumberOfFeatures(imgWidth, imgHeight, patchSize, binSize, l2normalize);
		int to = offset + nrOfFeatures;

		cv::Mat explanation(cv::Size(imgWidth, imgHeight), CV_32FC1, cv::Scalar(0));

		auto histogram = std::vector<std::vector<float>>(nrOfCellsHeight, std::vector<float>(nrOfCellsWidth, 0));
		int idx = 0;
		if (l2normalize) {
			for (int y = 0; y < nrOfCellsHeight - 1; y++) {
				for (int x = 0; x < nrOfCellsWidth - 1; x++) {

					//std::vector<int> sorted;

					for (int k = 0; k < binSize; k++) {

						histogram[y][x] += weightPerFeature[offset + idx];
						idx++;
					}

					for (int k = 0; k < binSize; k++) {

						histogram[y][x + 1] += weightPerFeature[offset + idx];
						idx++;
					}

					for (int k = 0; k < binSize; k++) {

						histogram[y + 1][x] += weightPerFeature[offset + idx];
						idx++;
					}

					for (int k = 0; k < binSize; k++) {

						histogram[y + 1][x + 1] += weightPerFeature[offset + idx];
						idx++;
					}
				}
			}
		}
		else {
			for (int y = 0; y < nrOfCellsHeight; y++) {
				for (int x = 0; x < nrOfCellsWidth; x++) {

					for (int k = 0; k < binSize; k++) {
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

	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize) {
		cv::Mat hog;

		bool drawHistograms = false;
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



	IntegralHistogram prepareDataForHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int binSize) {

		IntegralHistogram hist;
		double max = 1.0;


		hist.create(weights.cols, weights.rows, binSize, [&](int x, int y, std::vector<cv::Mat>& ihist) -> void {

			double anglePixel = normalizedBinningValues.at<float>(y, x);
			if (anglePixel < 0) anglePixel = 0; // this can only happen when either no normalization was done, or a rounding error in float

			double weight = weights.at<float>(y, x);

			// distribute based on angle
			// 15 in [0-20] = 0.25 * 15 for bin 0 and 0.75 * 15 for bin 1
			double valBins = anglePixel / max * binSize;
			if (valBins >= binSize) valBins = binSize - 1;

			int bin1 = floor(valBins);
			int bin2 = (bin1 + 1) % binSize;

			double tBegin = bin1 == 0 ? 0 : bin1 * max / binSize;
			double tEnd = bin2 == 0 ? max : bin2 * max / binSize;

			double u = (tEnd - anglePixel) / (tEnd - tBegin);

			ihist[bin1].at<double>(y, x) += (weight * u);
			ihist[bin2].at<double>(y, x) += (weight * (1-u));
			/*histogram[bin1] += weight * u;
			histogram[bin2] += weight * (1 - u);*/
		});

		return hist;
	}

	HistogramResult getHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize, 
		cv::Rect& iHistRoi, const IntegralHistogram* preparedData, int refWidth, int refHeight) {
		double max = 1.0;

		int nrOfCellsWidth = refWidth / patchSize;
		int nrOfCellsHeight = refHeight / patchSize;
		
		std::vector<std::vector<Histogram>> cells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram(binSize, 0)));

		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram& histogram = cells[y][x];
				if (preparedData == nullptr) {

					for (int l = 0; l < patchSize; l++) {
						for (int k = 0; k < patchSize; k++) {

							double anglePixel = normalizedBinningValues.at<float>(cv::Point(x * patchSize + k, y * patchSize + l));
							if (anglePixel < 0) anglePixel = 0; // this can only happen when either no normalization was done, or a rounding error in float

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

							double tBegin = bin1 == 0 ? 0 : bin1 * max / binSize;
							double tEnd = bin2 == 0 ? max : bin2 * max / binSize;

							double u = (tEnd - anglePixel) / (tEnd - tBegin);
							histogram[bin1] += (weight * u);
							histogram[bin2] += (weight * (1 - u));
						}
					}
				}
				else
					preparedData->calculateHistogramIntegral(iHistRoi.x + x * patchSize, iHistRoi.y + y * patchSize, patchSize, patchSize, histogram);

			}
		}

		std::vector<std::vector<Histogram>> newcells;
		if (l2normalize)
			newcells = getL2NormalizationOverLargerPatch(cells, nrOfCellsWidth, nrOfCellsHeight, binSize, l2normalize);

		cv::Mat hog;
		if (createImage) {
			cv::Mat m(refHeight, refWidth, CV_32FC1);
			hog = createHoGImage(m, cells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);
		}
		HistogramResult result;
		result.width = l2normalize ? nrOfCellsWidth - 1 : nrOfCellsWidth;
		result.height = l2normalize ? nrOfCellsHeight - 1 : nrOfCellsHeight;
		result.data = l2normalize ? newcells : cells;
		result.hogImage = hog;
		return result;

	}


	cv::Mat explain2DHOGFeature(int offset, std::vector<float>& weightPerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize) {

		int nrOfCellsWidth = imgWidth / patchSize;
		int nrOfCellsHeight = imgHeight / patchSize;

		int idx = 0;

		int nrOfFeatures = getNumberOfFeatures(imgWidth, imgHeight, patchSize, binSize, l2normalize);
		int to = offset + nrOfFeatures;

		cv::Mat explanation(cv::Size(imgWidth, imgHeight), CV_32FC1, cv::Scalar(0));


		auto histogram = std::vector<std::vector<float>>(nrOfCellsHeight, std::vector<float>(nrOfCellsWidth, 0));

		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {
				for (int k = 0; k < binSize; k++) {
					for (int l = 0; l < binSize; l++) {

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

	IntegralHistogram2D prepare2DDataForHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int binSize) {

		IntegralHistogram2D hist;
		double max = 1.0;


		hist.create(weights.cols, weights.rows, binSize, [&](int x, int y, std::vector<std::vector<cv::Mat>>& ihist) -> void {

			cv::Vec2f anglePixel = normalizedBinningValues.at<cv::Vec2f>(y,x);

			double weight = weights.at<float>(y,x);

			// distribute based on angle
			// 15 in [0-20] = 0.25 * 15 for bin 0 and 0.75 * 15 for bin 1
			cv::Vec2f valBins = anglePixel / max * binSize;
			if (valBins[0] >= binSize) valBins[0] = binSize - 1;
			if (valBins[1] >= binSize) valBins[1] = binSize - 1;
			if (valBins[0] < 0) valBins[0] = 0; // watch out for rounding errors
			if (valBins[1] < 0) valBins[1] = 0;

			int bin1x = floor(valBins[0]);
			int bin2x = (bin1x + 1) % binSize;

			int bin1y = floor(valBins[1]);
			int bin2y = (bin1y + 1) % binSize;

			// linearly interpolate
			float tBeginX = bin1x == 0 ? 0 : bin1x * max / binSize;
			float tEndX = bin2x == 0 ? max : bin2x * max / binSize;

			float tBeginY = bin1y == 0 ? 0 : bin1y * max / binSize;
			float tEndY = bin2y == 0 ? max : bin2y * max / binSize;

			double u = (tEndX - anglePixel[0]) / (tEndX - tBeginX);
			double v = (tEndY - anglePixel[1]) / (tEndY - tBeginY);


			ihist[bin1x][bin1y].at<double>(y, x) += weight * u * (v);
			ihist[bin2x][bin1y].at<double>(y, x) += weight * (1 - u) * (v);
			ihist[bin1x][bin2y].at<double>(y, x) += weight * u * (1 - v);
			ihist[bin2x][bin2y].at<double>(y, x) += weight * (1 - u) * (1 - v);
		});

		return hist;
	}


	HistogramResult get2DHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, cv::Rect& iHistRoi, const IntegralHistogram2D* preparedData, int refWidth, int refHeight) {


		double max = 1;

		int nrOfCellsWidth = refWidth / patchSize;
		int nrOfCellsHeight = refHeight / patchSize;

		std::vector<std::vector<Histogram2D>> cells(nrOfCellsHeight, std::vector<Histogram2D>(nrOfCellsWidth, Histogram2D(binSize, 0)));

		
		for (int y = 0; y < nrOfCellsHeight; y++) {

			for (int x = 0; x < nrOfCellsWidth; x++) {

				Histogram2D& histogram = cells[y][x];

				
				if (preparedData == nullptr) {
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

							// linearly interpolate
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
						}
					}
				}
				else {
					histogram = preparedData->calculateHistogramIntegral(iHistRoi.x + x * patchSize, iHistRoi.y + y * patchSize, patchSize, patchSize);
				}
			}
		}



		std::vector<std::vector<Histogram>> flattenedCells(nrOfCellsHeight, std::vector<Histogram>(nrOfCellsWidth, Histogram()));
		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {
				flattenedCells[y][x] = cells[y][x].flatten();
			}
		}

		cv::Mat hog;
		if (createImage) {
			cv::Mat m(weights.rows, weights.cols, CV_32FC1);
			hog = createHoGImage(m, flattenedCells, nrOfCellsWidth, nrOfCellsHeight, binSize, patchSize);
		}
		HistogramResult result;
		result.width = nrOfCellsWidth;
		result.height = nrOfCellsHeight;
		result.data = flattenedCells;
		result.hogImage = hog;
		return result;
	}

}