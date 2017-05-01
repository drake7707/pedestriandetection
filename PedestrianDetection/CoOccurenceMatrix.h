#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include "IntegralHistogram2D.h"


namespace coocc {

	typedef std::vector<std::vector<float>> CoOccurrenceMatrix;


	IntegralHistogram2D prepareData(cv::Mat& img, int binSize);

	/// <summary>
	/// Creates a 2D array of co-occurrence matrices, each corresponding to a patch
	/// </summary>
	std::vector<std::vector<CoOccurrenceMatrix>> getCoOccurenceMatrix(cv::Mat& img, int imgWidth, int imgHeight, int patchSize, int binSize, cv::Rect& iHistROI, const IntegralHistogram2D* iHist);

	/// <summary>
	/// Creates a cooccurrence of the given image (which will be the truncated to the patch)
	/// </summary>
	CoOccurrenceMatrix getCoOccurenceMatrixOfPatch(cv::Mat& img, int binSize);

	/// <summary>
	/// Creates a visual representation (heat map-like) of a co-occurrence matrix
	/// </summary>
	cv::Mat getCoOccurenceMatrixImage(int width, int height, CoOccurrenceMatrix& matrix);

	/// <summary>
	/// Creates a visual representation of all the co-occurrence matrices of the patches in the image
	/// </summary>
	cv::Mat createFullCoOccurrenceMatrixImage(cv::Mat baseImage, std::vector<std::vector<CoOccurrenceMatrix>>& cells, int patchSize);

}