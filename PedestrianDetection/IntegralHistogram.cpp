#include "IntegralHistogram.h"

void IntegralHistogram::create(int width, int height, int binSize, std::function<void(int x, int y, std::vector<cv::Mat>& ihist)> setBinValues) {
	ihist = std::vector<cv::Mat>();
	for (int bin = 0; bin < binSize; bin++)
		ihist.push_back(cv::Mat(height, width, CV_32FC1, cv::Scalar(0)));

	this->binSize = binSize;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{

			// copy over
			for (int bin = 0; bin < binSize; bin++) {
				ihist[bin].at<float>(j, i) =
					(i - 1 < 0 ? 0 : ihist[bin].at<float>(j, i - 1))
					+ (j - 1 < 0 ? 0 : ihist[bin].at<float>(j - 1, i))
					- ((i - 1 < 0 || j - 1 < 0) ? 0 : ihist[bin].at<float>(j - 1, i - 1));
			}

			// set specific bin values for pixel i,j
			setBinValues(i, j, ihist);
		}
	}
}

void IntegralHistogram::calculateHistogramIntegral(int x, int y, int w, int h, Histogram& hist) const {
	int minx = x - 1;
	int miny = y - 1;
	int maxx = x + w - 1;
	int maxy = y + h - 1;
	for (int bin = 0; bin < binSize; bin++) {
		// A - B
		// |   |
		// C - D
		//   A + D - B - C
		// determine integral histogram values of bin at [i][j]

		float A = (minx > 0 && miny > 0) ? ihist[bin].at<float>(miny, minx) : 0;
		float B = (miny > 0) ? ihist[bin].at<float>(miny, maxx) : 0;
		float C = (minx > 0) ? ihist[bin].at<float>(maxy, minx) : 0;
		float D = ihist[bin].at<float>(maxy, maxx);

		float value = A + D - C - B;
		hist[bin] = value;
	}
}

Histogram IntegralHistogram::calculateHistogramIntegral(int x, int y, int w, int h) const {
	Histogram hist(binSize, 0);
	calculateHistogramIntegral(x, y, w, h, hist);
	return hist;
}


