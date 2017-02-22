#include "Helper.h"



void showHistogram(Histogram& histogram) {


	cv::Mat hist(cv::Size(300, 300), CV_8U, cv::Scalar(255));

	int histWidth = 256;

	int padding = (hist.cols - histWidth) / 2;
	rectangle(hist, cv::Rect(0, 0, padding, hist.rows), cv::Scalar(200), -1);
	rectangle(hist, cv::Rect(hist.cols - (hist.cols - histWidth) / 2, 0, padding, hist.rows), cv::Scalar(200), -1);

	float maxVal = *std::max_element(histogram.begin(), histogram.end());

	for (int i = 0; i < histogram.size(); i++)
	{
		float val = histogram[i] / maxVal;

		int rectWidth = histWidth / histogram.size();

		int x = padding + rectWidth * i;
		int y = hist.rows - val * hist.rows;
		rectangle(hist, cv::Rect(x, y, rectWidth, hist.rows), cv::Scalar(0), -1);
	}

	cv::namedWindow("Histogram", CV_WINDOW_KEEPRATIO);
	cv::imshow("Histogram", hist);
}

int randBetween(int min, int max) {
	return min + 1.0 * rand()/RAND_MAX * (max - min);
}

int ceilTo(double val, double target) {
	return ceil(val / target) * target;
}

