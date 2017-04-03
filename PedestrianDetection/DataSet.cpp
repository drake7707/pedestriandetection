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
	std::function<void(int imageNumber)> onImageStarted,
	std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTP)> func,
	std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
	int parallization) const {

	std::vector<std::vector<DataSetLabel>> labelsPerNumber = getLabelsPerNumber();

	parallel_for(0, labelsPerNumber.size(), parallization, [&](int imgNumber) -> void {
		//for (auto& pair : images) {

		// check if the image number has to be processed
		if (canSelectFunc(imgNumber)) {

			// notify start of image
			onImageStarted(imgNumber);

			auto imgs = getImagesForNumber(imgNumber);
			cv::Mat mRGB = imgs[0];
			cv::Mat mDepth = imgs[1];
			cv::Mat mThermal = imgs[2];


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

			// slide window over the image
			int idx = 0;
			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {

				// skip all windows that intersect with the don't care regions
				if (!intersectsWith(bbox, dontCareRegions)) { 

					// resize the window to the reference size
					cv::Mat regionRGB;
					if (mRGB.cols > 0 && mRGB.rows > 0)
						cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));

					cv::Mat regionDepth;
					if (mDepth.cols > 0 && mDepth.rows > 0)
						cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));

					cv::Mat regionThermal;
					if (mThermal.cols > 0 && mThermal.rows > 0)
						cv::resize(mThermal(bbox), regionThermal, cv::Size2d(refWidth, refHeight));



					// calculate the average depth IF depth is available
					bool hasDepth = mDepth.rows > 0 && mDepth.cols > 0;
					double depthAvg = 0;
					if (hasDepth) {
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
					}

					//	 only evaluate windows that fall within the depth range to speed up the evaluation
					if (!hasDepth || isWithinValidDepthRange(bbox.height, depthAvg)) {

						bool overlapsWithTruePositive = false;
						int resultClass;
						if (overlaps(bbox, truePositiveRegions)) {
							resultClass = 1;
							overlapsWithTruePositive = true;
						}
						else
							resultClass = -1;

						if (resultClass != 0) { // don't evaluate don't care regions
							func(idx, resultClass, imgNumber, bbox, regionRGB, regionDepth, regionThermal, overlapsWithTruePositive);
							idx++;
						}
					}
				}
			}, windowSizes, baseWindowStride, refWidth, refHeight);

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