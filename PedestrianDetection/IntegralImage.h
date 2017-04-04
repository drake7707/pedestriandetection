#pragma once
#include <opencv2/opencv.hpp>
#include "Histogram.h"
#include <functional>


class IntegralHistogram {

private:
	std::vector<std::vector<Histogram>> ihist;
	int binSize;
public:
	void create(int width, int height, int binSize, std::function<void(int x, int y, Histogram& hist)> setBinValues) {
		ihist = std::vector<std::vector<Histogram>>(height, std::vector<Histogram>(width, Histogram(binSize, 0)));
		this->binSize = binSize;

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{

				// copy over
				for (int bin = 0; bin < binSize; bin++) {
					ihist[j][i][bin] =
						(i - 1 < 0 ? 0 : ihist[j][i - 1][bin])
						+ (j - 1 < 0 ? 0 : ihist[j - 1][i][bin])
						- ((i - 1 < 0 || j - 1 < 0) ? 0 : ihist[j - 1][i - 1][bin]);
				}

				// set specific bin values for pixel i,j
				setBinValues(i, j, ihist[j][i]);
			}
		}
	}

	Histogram calculateHistogramIntegral(int x, int y, int w, int h) const {
		Histogram hist(binSize, 0);
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

			float A = (minx > 0 && miny > 0) ? ihist[miny][minx][bin] : 0;
			float B = (miny > 0) ? ihist[miny][maxx][bin] : 0;
			float C = (minx > 0) ? ihist[maxy][minx][bin] : 0;
			float D = ihist[maxy][maxx][bin];

			float value = A + D - C - B;
			hist[bin] = value;
		}
		return hist;
	}

};