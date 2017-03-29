#include "JetHeatMap.h"


namespace heatmap {

	double interpolate(double val, double y0, double x0, double y1, double x1) {
		return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
	}

	double base(double val) {
		if (val <= -0.75) return 0;
		else if (val <= -0.25) return interpolate(val, 0.0, -0.75, 1.0, -0.25);
		else if (val <= 0.25) return 1.0;
		else if (val <= 0.75) return interpolate(val, 1.0, 0.25, 0.0, 0.75);
		else return 0.0;
	}

	double red(double gray) {
		return base(gray - 0.5);
	}

	double green(double gray) {
		return base(gray);
	}

	double blue(double gray) {
		return base(gray + 0.5);
	}

	cv::Mat toHeatMap(cv::Mat& input) {
		cv::Mat heatmap(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		for (int j = 0; j < input.rows; j++)
		{
			for (int i = 0; i < input.cols; i++)
			{
				// 0 - 1 range to -1 - 1
				float val = input.at<float>(j, i) * 2 - 1;
				heatmap.at<cv::Vec3b>(j, i) = cv::Vec3b(heatmap::blue(val) * 255, heatmap::green(val) * 255, heatmap::red(val) * 255);
			}
		}
		return heatmap;
	}

};