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

	std::function<void(int imgNr, std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales)> onImageStarted,
	std::function<void(int idx, int resultClass, int imageNumber, int scale, cv::Rect& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTruePositive)> func,
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
			std::vector<cv::Mat> rgbScales;
			std::vector<cv::Mat> depthScales;
			std::vector<cv::Mat> thermalScales;
			for (auto& s : windowSizes) {
				double scale = 1.0  * refWidth / s.width;
				if (mRGB.cols > 0 && mRGB.rows > 0) {
					cv::Mat rgbScale;
					cv::resize(mRGB, rgbScale, cv::Size2d(mRGB.cols * scale, mRGB.rows * scale));
					rgbScales.push_back(rgbScale);
				}
				if (mDepth.cols > 0 && mDepth.rows > 0) {
					cv::Mat depthScale;
					cv::resize(mDepth, depthScale, cv::Size2d(mDepth.cols * scale, mDepth.rows * scale));
					depthScales.push_back(depthScale);
				}
				if (mThermal.cols > 0 && mThermal.rows > 0) {
					cv::Mat thermalScale;
					cv::resize(mThermal, thermalScale, cv::Size2d(mThermal.cols * scale, mThermal.rows * scale));
					thermalScales.push_back(thermalScale);
				}
			}


			onImageStarted(imgNumber, rgbScales, depthScales, thermalScales);

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

			for (int s = 0; s < windowSizes.size(); s++) {

				// slide window over the image
				int idx = 0;
				slideWindow(rgbScales[s].cols, rgbScales[s].rows, [&](cv::Rect bbox) -> void {
					double scale = 1.0  *  windowSizes[s].width / refWidth;
					cv::Rect scaledBBox = cv::Rect(bbox.x * scale, bbox.y * scale, windowSizes[s].width, windowSizes[s].height);

					// skip all windows that intersect with the don't care regions
					if (!intersectsWith(scaledBBox, dontCareRegions)) {

						cv::Mat regionRGB;
						if (rgbScales.size() > 0)
							regionRGB = rgbScales[s](bbox);

						cv::Mat regionDepth;
						if (depthScales.size() > 0)
							regionDepth = depthScales[s](bbox);

						cv::Mat regionThermal;
						if (thermalScales.size() > 0)
							regionThermal = thermalScales[s](bbox);


						// calculate the average depth IF depth is available
						bool hasDepth = depthScales.size() > 0 && depthScales[s].rows > 0 && depthScales[s].cols > 0;
						double depthAvg = 0;
						if (hasDepth) {
							double depthSum = 0;
							int depthCount = 0;
							int xOffset = bbox.x + bbox.width / 2;
							for (int y = bbox.y; y < bbox.y + bbox.height; y++)
							{
								for (int i = xOffset - 1; i <= xOffset + 1; i++)
								{
									depthSum += depthScales[s].at<float>(y, i);
									depthCount++;
								}
							}
							depthAvg = (depthSum / depthCount);
						}

						//	 only evaluate windows that fall within the depth range to speed up the evaluation
						if (!hasDepth || isWithinValidDepthRange(scaledBBox.height, depthAvg)) {

							bool overlapsWithTruePositive = false;
							int resultClass;
							if (overlaps(scaledBBox, truePositiveRegions)) {
								resultClass = 1;
								overlapsWithTruePositive = true;
							}
							else
								resultClass = -1;

							if (resultClass != 0) { // don't evaluate don't care regions
								func(idx, resultClass, imgNumber, s, scaledBBox, bbox, regionRGB, regionDepth, regionThermal, overlapsWithTruePositive);
								idx++;
							}
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