#include "Helper.h"



void showHistogram(Histogram& histogram, std::string title) {


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

	cv::namedWindow(title, CV_WINDOW_KEEPRATIO);
	cv::imshow(title, hist);
}

int randBetween(int min, int max) {
	return min + 1.0 * rand() / RAND_MAX * (max - min);
}

int ceilTo(double val, double target) {
	return ceil(val / target) * target;
}

void slideWindow(int imgWidth, int imgHeight, std::function<void(cv::Rect2d bbox)> func, double minScaleReduction , double maxScaleReduction, int slidingWindowStep, int refWidth, int refHeight) {
	int slidingWindowWidth = 64;
	int slidingWindowHeight = 128;
//	int slidingWindowStep = 8;


	double invscale = minScaleReduction;
	while (invscale < maxScaleReduction) {
		double  iWidth = 1.0 *imgWidth / invscale;
		double iHeight = 1.0* imgHeight / invscale;
		double rectWidth = 1.0 * slidingWindowWidth / invscale;
		double rectHeight = 1.0 * slidingWindowHeight / invscale;
		for (double j  = 0; j < imgHeight - rectHeight; j += slidingWindowStep / invscale) {
			for (double i  = 0; i < imgWidth - rectWidth; i += slidingWindowStep / invscale) {
				
				cv::Rect windowRect(i,j,rectWidth, rectHeight);
				func(windowRect);
			}
		}
		invscale *= 2;
	}
}



void iterateDataSet(const std::string& baseDatasetPath, std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth)> func) {
	int i = 0;
	bool stop = false;
	while (!stop) {
		if (canSelectFunc(i)) {
			std::string rgbTP = baseDatasetPath + PATH_SEPARATOR + "tp" + PATH_SEPARATOR + "rgb" + std::to_string(i) + ".png";
			std::string rgbTN = baseDatasetPath + PATH_SEPARATOR + "tn" + PATH_SEPARATOR + "rgb" + std::to_string(i) + ".png";
			std::string depthTP = baseDatasetPath + PATH_SEPARATOR + "tp" + PATH_SEPARATOR + "depth" + std::to_string(i) + ".png";
			std::string depthTN = baseDatasetPath + PATH_SEPARATOR + "tn" + PATH_SEPARATOR + "depth" + std::to_string(i) + ".png";

			cv::Mat rgb;
			cv::Mat depth;
			rgb = cv::imread(rgbTP);
			depth = cv::imread(depthTP, CV_LOAD_IMAGE_ANYDEPTH);
			
			if (rgb.cols == 0 || rgb.rows == 0 || depth.cols == 0 || depth.rows == 0) {
				stop = true;
				break;
			}
			depth.convertTo(depth, CV_32FC1, 1.0 / 0xFFFF, 0);

			func(i, 1, rgb, depth);

			rgb = cv::imread(rgbTN);
			depth = cv::imread(depthTN, CV_LOAD_IMAGE_ANYDEPTH);
			if (rgb.cols == 0 || rgb.rows == 0 || depth.cols == 0 || depth.rows == 0) {
				stop = true;
				break;
			}
			depth.convertTo(depth, CV_32FC1, 1.0 / 0xFFFF, 0);


			func(i, -1, rgb, depth);
		}
		i++;
	}
}