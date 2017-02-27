
#include "opencv2/opencv.hpp"

using namespace cv;

namespace Algorithms {

	// From https://github.com/bytefish/opencv/blob/master/lbp/lbp.cpp

	// now the wrapper functions
	void OLBP(const Mat& src, Mat& dst, uchar minDiff = 0);

	void ELBP(const Mat& src, Mat& dst, int radius, int neighbors);

	void VARLBP(const Mat& src, Mat& dst, int radius, int neighbors);
	// now the Mat return functions
	Mat OLBP(const Mat& src);

	Mat ELBP(const Mat& src, int radius, int neighbors);
	Mat VARLBP(const Mat& src, int radius, int neighbors);

}