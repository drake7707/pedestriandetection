#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "Histogram.h"
#include "Histogram2D.h"
#include "FeatureVector.h"
#include "HistogramResult.h"
#include "Histogram2DResult.h"
#include <memory>
#include "IPreparedData.h"
#include "HOG1DPreparedData.h"

namespace hog {

	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize);

	int getNumberOfFeatures(int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize);

	cv::Mat explain2DHOGFeature(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize);
	cv::Mat explainHOGFeature(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool full360, bool l2normalize);

	HistogramResult get2DHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize);

	HistogramResult getHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize,
		cv::Rect& iHistRoi, const IntegralHistogram* preparedData, int refWidth, int refHeight);

	IntegralHistogram prepareDataForHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int binSize);

}
