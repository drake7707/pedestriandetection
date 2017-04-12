#include "ROIManager.h"



ROIManager::ROIManager()
{
}


ROIManager::~ROIManager()
{
}

void ROIManager::prepare(cv::Mat& mRGB, cv::Mat& mDepth, cv::Mat& mThermal)
{

	if (mThermal.cols > 0 && mThermal.rows > 0) {
		double w = 12;
		double alpha = 2;
		//double beta = 8;


		cv::Mat mThermal8U;
		mThermal.convertTo(mThermal8U, CV_8UC1, 255);


		std::vector<float> scales = { 0.5, 0.75, 1 };

		std::vector<cv::Rect> candidates;

		cv::Mat thermalMask(mThermal.rows, mThermal.cols, CV_8UC1, cv::Scalar(0));


		for (auto& scale : scales) {

			cv::Mat thermal;
			cv::resize(mThermal8U, thermal, cv::Size(mThermal8U.cols * scale, mThermal8U.rows * scale));

			cv::Mat dest = thermal.clone();
			for (int j = 0; j < thermal.rows; j++)
			{
				for (int i = 0; i < thermal.cols; i++)
				{
					char value = thermal.at<char>(j, i);

					double tl = 0;

					int count = 0;
					for (int x = i - w; x <= i + w; x++) {
						if (x >= 0 && x < thermal.cols) {
							tl += thermal.at<char>(j, x);
							count++;
						}
					}
					tl = tl / count + alpha;

					double t3 = cv::max(1.06 * (tl - alpha), tl + 2);
					double t2 = cv::min(t3, tl + 8);
					double t1 = cv::min(t2, 230.0);
					double th = cv::max(t1, tl);
					//  let th = tl + beta;


					if (value > th)
						dest.at<char>(j, i) = 255;
					else if (value < tl)
						dest.at<char>(j, i) = 0;

					else if (i - 1 >= 0 && dest.at<char>(j, i - 1) > 0)
						dest.at<char>(j, i) = 255;
					else
						dest.at<char>(j, i) = 0;
				}
			}


			cv::dilate(dest, dest, cv::Mat());
			cv::erode(dest, dest, cv::Mat());

			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Rect> boundRect(contours.size());
			cv::findContours(dest, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			for (int i = 0; i < contours.size(); i++)
			{
				int minX = std::numeric_limits<int>().max();
				int maxX = std::numeric_limits<int>().min();

				int minY = std::numeric_limits<int>().max();
				int maxY = std::numeric_limits<int>().min();
				for (auto& p : contours[i]) {
					if (p.x > maxX) maxX = p.x;
					if (p.x < minX) minX = p.x;
					if (p.y > maxY) maxY = p.y;
					if (p.y < minY) minY = p.y;
				}

				cv::Rect r = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);

				if (1.0 * r.height / r.width >= 1 && 1.0 * r.height / r.width < 6 && r.width >= 8 && r.height >= 16) {
					candidates.push_back(cv::Rect(r.x / scale, r.y / scale, r.width / scale, r.height / scale));
				}
			}
		}

		for (auto& r : candidates) {
			cv::rectangle(thermalMask, r, cv::Scalar(1), -1);
		}

		thermalRegions.create(mThermal.cols, mThermal.rows, 1, [&](int x, int y, std::vector<cv::Mat>& ihist) -> void { ihist[0].at<float>(y, x) += thermalMask.at<char>(y, x); });

	}



	
}

bool ROIManager::needToEvaluate(const cv::Rect2d& bbox, const cv::Mat& mRGB, const cv::Mat & mDepth, const cv::Mat& mThermal, std::function<bool(double, double)> isValidDepthRangeFunc) const
{
	bool needToEvaluate = true;

	// calculate the average depth IF depth is available, //TODO maybe also use integral image for the depth so the sum can be instantly calculated ?
	bool hasDepth = mDepth.rows > 0 && mDepth.cols > 0;
	if (needToEvaluate && hasDepth) {
		double depthAvg = 0;
		double depthSum = 0;
		int depthCount = 0;
		int xOffset = bbox.x + bbox.width / 2;
		for (int y = bbox.y; y < bbox.y + bbox.height; y++)
		{
			for (int i = xOffset - 1; i <= xOffset + 1; i++)
			{
				depthSum += mDepth.at<float>(y, i);
				depthCount++;
			}
		}
		depthAvg = (depthSum / depthCount);
		needToEvaluate = isValidDepthRangeFunc(bbox.height, depthAvg);
	}

	bool hasThermal = mThermal.rows > 0 && mThermal.cols > 0;
	if (needToEvaluate && hasThermal) {
		float sum = thermalRegions.calculateHistogramIntegral(bbox.x, bbox.y, bbox.width, bbox.height)[0];
		if (sum > 0) {
			// bbox has intersected with a candidate region
		}
		else {
			// bbox did not intersect with candidate region, no need to evaluate
			needToEvaluate = false;
		}
	}

	return needToEvaluate;
}
