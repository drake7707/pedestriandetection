#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include <functional>

namespace Algorithms {

	/// <summary>
	/// Returns the UV coordinates to linearly interpolate between 2 marching square cells
	/// </summary>
	std::vector<cv::Point> getUV(int nr, float threshold, float vLT, float vLB, float vRT, float vRB);

	/// <summary>
	/// Applies the matching squares algorithm to draw iso-lines
	/// </summary>
	void marchingSquares(std::function<float(int, int)> func, std::function<void(int x1, int y1, int x2, int y2)> lineFunc, float targetValue, int imgWidth, int imgHeight, int nrCellsX = 300, int nrCellsY = 100);

};