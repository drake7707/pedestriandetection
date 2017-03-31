#include "TrainingDataSet.h"
#include <fstream>
#include "Helper.h"
#include "KITTIDataSet.h"


TrainingDataSet::TrainingDataSet(DataSet* dataSet)
	: baseDataSetPath(baseDataSetPath), dataSet(dataSet)
{
}

TrainingDataSet::TrainingDataSet(const TrainingDataSet& trainingDataSet) {
	this->baseDataSetPath = trainingDataSet.baseDataSetPath;
	this->images = trainingDataSet.images;
	this->dataSet = trainingDataSet.dataSet;
}

TrainingDataSet::~TrainingDataSet()
{
}

void TrainingDataSet::addTrainingRegion(int imageNumber, TrainingRegion& region)
{
	auto& img = images.find(imageNumber);
	if (img != images.end())
		img->second.regions.push_back(region);
}


void TrainingDataSet::addTrainingImage(TrainingImage & img)
{
	images.emplace(img.number, img);
}

void TrainingDataSet::save(std::string & path)
{
	std::ofstream str(path);
	if (!str.is_open())
		throw std::exception("Unable to open file");

	str << std::fixed;

	for (auto& pair : images) {
		str << pair.second.number << " " << pair.second.regions.size() << std::endl;

		for (auto& r : pair.second.regions) {
			str << r.region.x << " " << r.region.y << " " << r.region.width << " " << r.region.height << " " << r.regionClass << std::endl;
		}
	}
}

void TrainingDataSet::load(std::string & path)
{
	std::ifstream str(path);
	if (!str.is_open())
		throw std::exception("Unable to open file");

	images.clear();
	nrOfSamples = 0;
	while (!str.eof()) {
		int number;
		int count;
		str >> number;
		str >> count;

		TrainingImage img;
		img.number = number;
		img.regions.reserve(count);

		for (int i = 0; i < count; i++)
		{
			int x, y, w, h, regionClass;
			str >> x;
			str >> y;
			str >> w;
			str >> h;
			str >> regionClass;
			TrainingRegion region;
			region.region = cv::Rect(x, y, w, h);
			region.regionClass = regionClass;
			img.regions.push_back(region);
			nrOfSamples++;
		}

		images.emplace(img.number, img);
	}
}

int TrainingDataSet::getNumberOfImages() const {
	return dataSet->getNrOfImages();
}


void TrainingDataSet::iterateDataSetImages(std::function<void(int imageNumber, cv::Mat&rgb, cv::Mat&depth, const std::vector<TrainingRegion>& regions)> func) const {
	for (auto& pair : images) {
		auto imgs = dataSet->getImagesForNumber(pair.first);
		cv::Mat rgb = imgs[0];
		cv::Mat depth = imgs[1];
		func(pair.first, rgb, depth, pair.second.regions);
	}
}

void TrainingDataSet::iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal)> func) const {


	int idx = 0;
	for (auto& pair : images) {

		if (canSelectFunc(pair.first)) {
			auto imgs = dataSet->getImagesForNumber(pair.first);
			cv::Mat rgb = imgs[0];
			cv::Mat depth = imgs[1];
			cv::Mat thermal = imgs[2];

			for (auto& r : pair.second.regions) {
				if (r.regionClass != 0) { // exclude don't care regions
					cv::Mat regionRGB;
					if (rgb.cols > 0 && rgb.rows > 0)
						cv::resize(rgb(r.region), regionRGB, cv::Size2d(refWidth, refHeight));

					cv::Mat regionDepth;
					if (depth.cols > 0 && depth.rows > 0)
						cv::resize(depth(r.region), regionDepth, cv::Size2d(refWidth, refHeight));

					cv::Mat regionThermal;
					if (thermal.cols > 0 && thermal.rows > 0)
						cv::resize(thermal(r.region), regionThermal, cv::Size2d(refWidth, refHeight));

					func(idx, r.regionClass, pair.first, r.region, regionRGB, regionDepth, regionThermal);
					idx++;

					cv::Mat rgbFlipped;
					cv::Mat depthFlipped;
					cv::Mat thermalFlipped;
					if (rgb.cols > 0 && rgb.rows > 0)
						cv::flip(regionRGB, rgbFlipped, 1);

					if (depth.cols > 0 && depth.rows > 0)
						cv::flip(regionDepth, depthFlipped, 1);

					if (thermal.cols > 0 && thermal.rows > 0)
						func(idx, r.regionClass, pair.first, r.region, rgbFlipped, depthFlipped, thermalFlipped);
					idx++;
				}
			}
		}
	}
}

//
//void TrainingDataSet::iterateDataSetWithSlidingWindow(std::vector<cv::Size>& windowSizes, int baseWindowStride,
//	std::function<bool(int number)> canSelectFunc,
//	std::function<void(int imageNumber)> onImageStarted,
//	std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTP)> func,
//	std::function<void(int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions)> onImageProcessed,
//	int parallization) const {
//
//	KITTIDataSet dataSet(baseDataSetPath);
//	int idx = 0;
//
//	auto labels = dataSet.getLabels();
//	parallel_foreach<int, TrainingImage>(images, parallization, [&](std::pair<int, TrainingImage> pair) -> void {
//		//for (auto& pair : images) {
//
//
//		if (canSelectFunc(pair.first)) {
//
//			onImageStarted(pair.first);
//
//			auto imgs = dataSet.getImagesForNumber(pair.first);
//			cv::Mat mRGB = imgs[0];
//			cv::Mat mDepth = imgs[1];
//
//
//			std::vector<cv::Rect2d> truePositiveRegions;
//			std::vector<std::string> truePositiveCategories;
//			std::vector<cv::Rect2d> dontCareRegions;
//			for (auto& r : pair.second.regions) {
//				if (r.regionClass == 1) {
//					truePositiveRegions.push_back(r.region);
//					truePositiveCategories.push_back(r.category);
//				}
//				else if (r.regionClass == 0)
//					dontCareRegions.push_back(r.region);
//			}
//
//			cv::Mat tmp = mRGB.clone();
//			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {
//
//				if (!intersectsWith(bbox, dontCareRegions)) { // skip all windows that intersect with the don't care regions
//					cv::Mat regionRGB;
//					cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));
//
//					cv::Mat regionDepth;
//					cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));
//
//					double depthSum = 0;
//					int depthCount = 0;
//					int xOffset = bbox.x + bbox.width / 2;
//					for (int y = bbox.y; y < bbox.y + bbox.height; y++)
//					{
//						for (int i = xOffset - 1; i <= xOffset + 1; i++)
//						{
//							depthSum += mDepth.at<float>(y, i);
//							depthCount++;
//						}
//					}
//					double depthAvg = (depthSum / depthCount);
//					//	 only evaluate windows that fall within the depth range to speed up the evaluation
//					if (isWithinValidDepthRange(bbox.height, depthAvg)) {
//
//						bool overlapsWithTruePositive;
//						int resultClass;
//						if (overlaps(bbox, truePositiveRegions)) {
//							resultClass = 1;
//							overlapsWithTruePositive = true;
//						}
//						else
//							resultClass = -1;
//
//						if (resultClass != 0) { // don't evaluate don't care regions
//							func(idx, resultClass, pair.first, bbox, regionRGB, regionDepth, tmp, overlapsWithTruePositive);
//							idx++;
//						}
//					}
//				}
//			}, windowSizes, baseWindowStride);
//
//			onImageProcessed(pair.first, truePositiveCategories, truePositiveRegions);
//			//cv::imshow("Temp", tmp);
//			//cv::waitKey(0);
//
//
//		}
//		//}
//	});
//}

std::string TrainingDataSet::getBaseDataSetPath() const
{
	return baseDataSetPath;
}

bool TrainingDataSet::isWithinValidDepthRange(int height, float depthAverage) const {
	return dataSet->isWithinValidDepthRange(height, depthAverage);
}

DataSet* TrainingDataSet::getDataSet() const {
	return (DataSet*)dataSet;
}