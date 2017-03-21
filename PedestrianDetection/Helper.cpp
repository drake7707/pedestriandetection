#include "Helper.h"
#include <map>



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


void parallel_for(int from, int to, int nrOfThreads, std::function<void(int)> func) {
	std::vector<std::thread> threads;
	threads.reserve(nrOfThreads);


	int blocks = ceil(1.0 * (to - from) / nrOfThreads);

	for (int t = 0; t < nrOfThreads; t++)
	{
		threads.push_back(std::thread([&](int offset) -> void {
			int length = offset + blocks >= to ? to : offset + blocks;
			for (int i = offset; i < length; i++)
			{
				func(i);
			}
		}, from + t * blocks));
	}

	for (auto& t : threads)
		t.join();
}


void slideWindow(int imgWidth, int imgHeight, std::function<void(cv::Rect bbox)> func, std::vector<cv::Size>& windowSizes, int slidingWindowStep, int refWidth, int refHeight) {
	int slidingWindowWidth = 64;
	int slidingWindowHeight = 128;
	//	int slidingWindowStep = 8;

	double topOffset = 0.3 * imgHeight;

	for (auto& s : windowSizes) {

		double scale = 1.0 *  s.width / slidingWindowWidth;
		double rectWidth = s.width;// 1.0 * slidingWindowWidth / invscale;
		double rectHeight = s.height;// 1.0 * slidingWindowHeight / invscale;

		for (double j = topOffset; j < imgHeight - rectHeight; j += slidingWindowStep * scale) {
			for (double i = 0; i < imgWidth - rectWidth; i += slidingWindowStep * scale) {

				cv::Rect windowRect(i, j, rectWidth, rectHeight);
				func(windowRect);
			}
		}
	}
}


double getIntersectionOverUnion(const cv::Rect& r1, const cv::Rect& r2) {
	double intersectionRect = (r1 & r2).area();
	double unionRect = (r1 | r2).area();


	return unionRect == 0 ? 0 : intersectionRect / unionRect;
}


std::vector < SlidingWindowRegion> applyNonMaximumSuppression(std::vector< SlidingWindowRegion>& windows, float iouTreshold) {
	std::vector<SlidingWindowRegion> wnds = windows;
	std::vector< SlidingWindowRegion> newwindows;

	bool hasMerged = true;
	while (hasMerged) {
		hasMerged = false;
		std::vector<bool> flaggedToRemove(wnds.size(), false);
		for (int j = 0; j < wnds.size(); j++)
		{
			for (int i = j + 1; i < wnds.size(); i++)
			{
				if (!flaggedToRemove[i] && !flaggedToRemove[j]) {
					if (getIntersectionOverUnion(wnds[i].bbox, wnds[j].bbox) > iouTreshold) {
						hasMerged = true;
						// overlap, only keep 1 of the 2
						if (wnds[i].score > wnds[j].score) {
							// remove j
							flaggedToRemove[j] = true;
						}
						else {
							//remove i
							flaggedToRemove[i] = true;
						}
					}
				}
			}
		}

		for (int i = 0; i < wnds.size(); i++)
		{
			if (!flaggedToRemove[i])
				newwindows.push_back(wnds[i]);
		}
		wnds = newwindows;
		newwindows.clear();
	}

	return wnds;
}



bool overlaps(cv::Rect2d r, std::vector<cv::Rect2d>& selectedRegions) {
	for (auto& region : selectedRegions) {
		if (getIntersectionOverUnion(r, region) > 0.5)
			return true;
	}

	return false;
}

bool intersectsWith(cv::Rect2d r, std::vector<cv::Rect2d>& selectedRegions) {
	for (auto& region : selectedRegions) {
		double intersectionRect = (r & region).area();
		if(intersectionRect > 0)
			return true;
	}

	return false;
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

bool FileExists(const std::string &Filename)
{
	return access(Filename.c_str(), 0) == 0;
}