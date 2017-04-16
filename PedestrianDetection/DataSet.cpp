#include "DataSet.h"
#include "Helper.h"


DataSet::DataSet()
{
}


DataSet::~DataSet()
{
}

std::vector<std::vector<DataSetLabel>> DataSet::getLabelsPerNumber() const {

	auto labels = getLabels();
	std::vector<std::vector<DataSetLabel>> labelsPerNumber(getNrOfImages(), std::vector<DataSetLabel>());
	for (auto& l : labels)
		labelsPerNumber[atoi(l.getNumber().c_str())].push_back(l);
	return labelsPerNumber;
}


void DataSet::iterateDataSetWithSlidingWindow(const std::vector<cv::Size>& windowSizes, int baseWindowStride,
	int refWidth, int refHeight,
	std::function<bool(int number)> canSelectFunc,
	std::function<void(int imgNr)> onImageStarted,
	std::function<void(int imgNr, double scale, cv::Mat& fullRGBScale, cv::Mat& fullDepthScale, cv::Mat& fullThermalScale)> onScaleStarted,
	std::function<void(int idx, int resultClass, int imageNumber, int scale, cv::Rect2d& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTruePositive)> func,
	std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
	int parallization) const {

	std::vector<std::vector<DataSetLabel>> labelsPerNumber = getLabelsPerNumber();

	parallel_for(0, labelsPerNumber.size(), parallization, [&](int imgNumber) -> void {
		//for (auto& pair : images) {

		// check if the image number has to be processed
		if (canSelectFunc(imgNumber)) {

			auto imgs = getImagesForNumber(imgNumber);
			cv::Mat mRGB = imgs[0];
			cv::Mat mDepth = imgs[1];
			cv::Mat mThermal = imgs[2];

			// notify start of image
			onImageStarted(imgNumber);

			// determine the true positive and their categories and the don't care regions
			std::vector<cv::Rect2d> truePositiveRegions;
			std::vector<std::string> truePositiveCategories;
			std::vector<cv::Rect2d> dontCareRegions;
			for (auto& r : labelsPerNumber[imgNumber]) {
				if (!r.isDontCareArea()) {
					truePositiveRegions.push_back(r.getBbox());
					truePositiveCategories.push_back(getCategory(&r));
				}
				else
					dontCareRegions.push_back(r.getBbox());
			}

			ROIManager roiManager;
			roiManager.prepare(mRGB, mDepth, mThermal);

			for (int s = 0; s < windowSizes.size(); s++) {
				double scale = 1.0  *  windowSizes[s].width / refWidth;

				cv::Mat rgbScale;
				if (mRGB.cols > 0 && mRGB.rows > 0)
					cv::resize(mRGB, rgbScale, cv::Size2d(mRGB.cols * scale, mRGB.rows * scale));

				cv::Mat depthScale;
				if (mDepth.cols > 0 && mDepth.rows > 0)
					cv::resize(mDepth, depthScale, cv::Size2d(mDepth.cols * scale, mDepth.rows * scale));

				cv::Mat thermalScale;
				if (mThermal.cols > 0 && mThermal.rows > 0)
					cv::resize(mThermal, thermalScale, cv::Size2d(mThermal.cols * scale, mThermal.rows * scale));

				onScaleStarted(imgNumber, scale, rgbScale, depthScale, thermalScale);

				// slide window over the image
				int idx = 0;
				slideWindow(rgbScale.cols, rgbScale.rows, [&](cv::Rect bbox) -> void {

					// Calculate the bounding box on the original image size
					cv::Rect2d scaledBBox = cv::Rect2d(bbox.x / scale, bbox.y / scale,bbox.width / scale, bbox.height / scale);

					// skip all windows that intersect with the don't care regions
					if (!intersectsWith(scaledBBox, dontCareRegions)) {

						bool needToEvaluate = roiManager.needToEvaluate(scaledBBox, mRGB, mDepth, mThermal,
							[&](double height, double depthAvg) -> bool { return this->isWithinValidDepthRange(height, depthAvg); });

						if (needToEvaluate) {

							cv::Mat regionRGB;
							if (rgbScale.cols > 0 && rgbScale.rows > 0)
								regionRGB = rgbScale(bbox);

							cv::Mat regionDepth;
							if (depthScale.cols > 0 && depthScale.rows > 0)
								regionDepth = depthScale(bbox);

							cv::Mat regionThermal;
							if (thermalScale.cols > 0 && thermalScale.rows > 0)
								regionThermal = thermalScale(bbox);

							bool overlapsWithTruePositive = false;
							int resultClass;
							if (overlaps(scaledBBox, truePositiveRegions)) {
								resultClass = 1;
								overlapsWithTruePositive = true;
							}
							else
								resultClass = -1;

							func(idx, resultClass, imgNumber, s, scaledBBox, bbox, regionRGB, regionDepth, regionThermal, overlapsWithTruePositive);
							idx++;
						}
					}
				}, baseWindowStride, refWidth, refHeight);
			}


			onImageProcessed(imgNumber, truePositiveCategories, truePositiveRegions);
			//cv::imshow("Temp", tmp);
			//cv::waitKey(0);


		}
		//}
	});
}


bool DataSet::isWithinValidDepthRange(int height, float depthAverage) const {
	return true;
}