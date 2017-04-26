#include "JetHeatMap.h"


// Code from http://stackoverflow.com/a/7706668/694640
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

	// This is a subfunction of HSLtoRGB
	void HSLtoRGB_Subfunction(int* c, const double& temp1, const double& temp2, const double& temp3)
	{
		if ((temp3 * 6) < 1)
			*c = (uint)((temp2 + (temp1 - temp2) * 6 * temp3) * 100);
		else
			if ((temp3 * 2) < 1)
				*c = (uint)(temp1 * 100);
			else
				if ((temp3 * 3) < 2)
					*c = (uint)((temp2 + (temp1 - temp2)*(.66666 - temp3) * 6) * 100);
				else
					*c = (uint)(temp2 * 100);
		return;
	}
	void HSLtoRGB(const int h, const int s, const int l, int* r, int* g, int* b)
	{
		*r = 0;
		*g = 0;
		*b = 0;

		double L = ((double)l) / 100;
		double S = ((double)s) / 100;
		double H = ((double)h) / 360;

		if (s == 0)
		{
			*r = l;
			*g = l;
			*b = l;
		}
		else
		{
			double temp1 = 0;
			if (L < .50)
			{
				temp1 = L*(1 + S);
			}
			else
			{
				temp1 = L + S - (L*S);
			}

			double temp2 = 2 * L - temp1;

			double temp3 = 0;
			for (int i = 0; i < 3; i++)
			{
				switch (i)
				{
				case 0: // red
				{
					temp3 = H + .33333;
					if (temp3 > 1)
						temp3 -= 1;
					HSLtoRGB_Subfunction(r, temp1, temp2, temp3);
					break;
				}
				case 1: // green
				{
					temp3 = H;
					HSLtoRGB_Subfunction(g, temp1, temp2, temp3);
					break;
				}
				case 2: // blue
				{
					temp3 = H - .33333;
					if (temp3 < 0)
						temp3 += 1;
					HSLtoRGB_Subfunction(b, temp1, temp2, temp3);
					break;
				}
				default:
				{

				}
				}
			}
		}
		*r = (int)((((double)*r) / 100) * 255);
		*g = (int)((((double)*g) / 100) * 255);
		*b = (int)((((double)*b) / 100) * 255);
	}



	cv::Mat toHeatMap(cv::Mat& input) {
		cv::Mat heatmap(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		for (int j = 0; j < input.rows; j++)
		{
			for (int i = 0; i < input.cols; i++)
			{
				// 0 - 1 range to -1 - 1
				//float val = input.at<float>(j, i) * 2 - 1;
				//heatmap.at<cv::Vec3b>(j, i) = cv::Vec3b(heatmap::blue(val) * 255, heatmap::green(val) * 255, heatmap::red(val) * 255);

				int r, g, b;

				// h: 210, 360 
				// l: 20 -> 100 -> 20
				float val = input.at<float>(j, i);
				HSLtoRGB(val < 0.5 ? 210 : 360, 100, val < 0.5 ? (20 + (val * 2) * (100 - 20)) : (20 + (1 - (val - 0.5) * 2) * (100 - 20)), &r,&g,&b);

				//heatmap.at<cv::Vec3b>(j, i) = cv::Vec3b(heatmap::blue(val) * 255, heatmap::green(val) * 255, heatmap::red(val) * 255);
				heatmap.at<cv::Vec3b>(j, i) = cv::Vec3b(b, g, r);
			}
		}
		return heatmap;
	}



};