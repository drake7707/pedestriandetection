#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include <functional>

namespace Algorithms {

	std::vector<cv::Point> getUV(int nr, float threshold, float vLT, float vLB, float vRT, float vRB);

	void marchingSquares(std::function<float(int, int)> func, std::function<void(int x1, int y1, int x2, int y2)> lineFunc, float targetValue, int imgWidth, int imgHeight, int nrCellsX = 300, int nrCellsY = 100);

};