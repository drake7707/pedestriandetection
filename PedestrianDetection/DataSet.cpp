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


void DataSet::iterateDataSetWithSlidingWindow(std::vector<cv::Size>& windowSizes, int baseWindowStride,
	int refWidth, int refHeight,
	std::function<bool(int number)> canSelectFunc,
	std::function<void(int imageNumber)> onImageStarted,
	std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTP)> func,
	std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
	int parallization) const {



	auto labels = getLabels();
	std::vector<std::vector<DataSetLabel>> labelsPerNumber(getNrOfImages(), std::vector<DataSetLabel>());
	for (auto& l : labels)
		labelsPerNumber[atoi(l.getNumber().c_str())].push_back(l);

	parallel_for(0, labelsPerNumber.size(), parallization, [&](int imgNumber) -> void {
		//for (auto& pair : images) {


		if (canSelectFunc(imgNumber)) {

			onImageStarted(imgNumber);

			auto imgs = getImagesForNumber(imgNumber);
			cv::Mat mRGB = imgs[0];
			cv::Mat mDepth = imgs[1];



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

			cv::Mat tmp = mRGB.clone();
			int idx = 0;
			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {

				if (!intersectsWith(bbox, dontCareRegions)) { // skip all windows that intersect with the don't care regions
					cv::Mat regionRGB;
					cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));

					cv::Mat regionDepth;
					cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));

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
					double depthAvg = (depthSum / depthCount);
					//	 only evaluate windows that fall within the depth range to speed up the evaluation
					if (isWithinValidDepthRange(bbox.height, depthAvg)) {

						bool overlapsWithTruePositive = false;
						int resultClass;
						if (overlaps(bbox, truePositiveRegions)) {
							resultClass = 1;
							overlapsWithTruePositive = true;
						}
						else
							resultClass = -1;

						if (resultClass != 0) { // don't evaluate don't care regions
							func(idx, resultClass, imgNumber, bbox, regionRGB, regionDepth, tmp, overlapsWithTruePositive);
							idx++;
						}
					}
				}
			}, windowSizes, baseWindowStride);

			onImageProcessed(imgNumber, truePositiveCategories, truePositiveRegions);
			//cv::imshow("Temp", tmp);
			//cv::waitKey(0);


		}
		//}
	});
}
