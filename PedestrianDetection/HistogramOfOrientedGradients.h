#pragma once
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "Histogram.h"
#include "Histogram2D.h"
#include "FeatureVector.h"
#include "HistogramResult.h"
#include <memory>
#include "IPreparedData.h"
#include "HOG1DPreparedData.h"

namespace hog {

	/// <summary>
	/// Creates a HOG image, either drawing histograms or oriented strokes
	/// </summary>
	cv::Mat createHoGImage(cv::Mat& mat, const std::vector<std::vector<Histogram>>& cells, int nrOfCellsWidth, int nrOfCellsHeight, int binSize, int patchSize);

	/// <summary>
	/// Returns the number of features the feature vector for HOG-like descriptors will contain
	/// </summary>
	int getNumberOfFeatures(int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize);

	/// <summary>
	/// Explains the 2D HOG feature by building a heat map of their occurrence
	/// </summary>
	cv::Mat explain2DHOGFeature(int offset, std::vector<float>& weightPerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool l2normalize);

	/// <summary>
	/// Explains the HOG features by building a heat map of their occurrence
	/// </summary>
	cv::Mat explainHOGFeature(int offset, std::vector<float>& weightPerFeature, int imgWidth, int imgHeight, int patchSize, int binSize, bool full360, bool l2normalize);

	/// <summary>
	/// Returns the 2D HOG-like features for given weights and binning values
	/// </summary>
	HistogramResult get2DHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage);

	/// <summary>
	/// Returns the HOG-like result for given weights and binning values. For example, for HOG the weights will be the magnitudes, whereas the binning values will be
	/// the orientation of the gradients, but normalized to 0-1
	/// </summary>
	HistogramResult getHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int patchSize, int binSize, bool createImage, bool l2normalize,
		cv::Rect& iHistRoi, const IntegralHistogram* preparedData, int refWidth, int refHeight);

	/// <summary>
	/// Prepares the HOG-like feature evaluation by calculating an integral histogram
	/// </summary>
	IntegralHistogram prepareDataForHistogramsOfX(cv::Mat& weights, cv::Mat& normalizedBinningValues, int binSize);

}
