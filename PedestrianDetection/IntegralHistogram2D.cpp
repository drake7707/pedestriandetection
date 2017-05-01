#include "IntegralHistogram2D.h"

void IntegralHistogram2D::create(int width, int height, int binSize, std::function<void(int x, int y, std::vector<std::vector<cv::Mat>>& ihist)> setBinValues) {
	ihist = std::vector<std::vector<cv::Mat>>(binSize, std::vector<cv::Mat>());
	for (int binX = 0; binX < binSize; binX++) {		
		for (int binY = 0; binY < binSize; binY++)
		{
			ihist[binX].push_back(cv::Mat(height, width, CV_32FC1, cv::Scalar(0)));
		}
	}

	this->binSize = binSize;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{

			// copy over
			for (int binX = 0; binX < binSize; binX++) {
				for (int binY = 0; binY < binSize; binY++)
				{
					auto&  cell = ihist[binX][binY];
					cell.at<float>(j, i) =
						(i - 1 < 0 ? 0 : cell.at<float>(j, i - 1))
						+ (j - 1 < 0 ? 0 : cell.at<float>(j - 1, i))
						- ((i - 1 < 0 || j - 1 < 0) ? 0 : cell.at<float>(j - 1, i - 1));
				}
			}

			// set specific bin values for pixel i,j
			setBinValues(i, j, ihist);
		}
	}
}

void IntegralHistogram2D::calculateHistogramIntegral(int x, int y, int w, int h, Histogram2D& hist) const {
	int minx = x - 1;
	int miny = y - 1;
	int maxx = x + w - 1;
	int maxy = y + h - 1;
	for (int binX = 0; binX < binSize; binX++) {
		for (int binY = 0; binY < binSize; binY++)
		{
			// A - B
			// |   |
			// C - D
			//   A + D - B - C
			// determine integral histogram values of bin at [i][j]

			auto& cell = ihist[binX][binY];
			float A = (minx > 0 && miny > 0) ? cell.at<float>(miny, minx) : 0;
			float B = (miny > 0) ? cell.at<float>(miny, maxx) : 0;
			float C = (minx > 0) ? cell.at<float>(maxy, minx) : 0;
			float D = cell.at<float>(maxy, maxx);

			float value = A + D - C - B;
			hist[binX][binY] = value;
		}
	}
}

Histogram2D IntegralHistogram2D::calculateHistogramIntegral(int x, int y, int w, int h) const {
	Histogram2D hist(binSize, 0);
	calculateHistogramIntegral(x, y, w, h, hist);
	return hist;
}


