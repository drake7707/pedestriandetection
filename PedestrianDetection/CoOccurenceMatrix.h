#pragma once
#include <vector>
#include "opencv2/opencv.hpp"

typedef std::vector<std::vector<float>> CoOccurrenceMatrix;

std::vector<std::vector<CoOccurrenceMatrix>> getCoOccurenceMatrix(cv::Mat& img, int patchSize, int binSize);

CoOccurrenceMatrix getCoOccurenceMatrixOfPatch(cv::Mat& img, int binSize);

cv::Mat getCoOccurenceMatrixImage(int width, int height, CoOccurrenceMatrix& matrix);

cv::Mat createFullCoOccurrenceMatrixImage(cv::Mat baseImage, std::vector<std::vector<CoOccurrenceMatrix>>& cells, int patchSize);