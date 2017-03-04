#include "TrainingDataSet.h"
#include <fstream>
#include "Helper.h"
#include "KITTIDataSet.h"


TrainingDataSet::TrainingDataSet(std::string& baseDataSetPath)
	: baseDataSetPath(baseDataSetPath) , dataSet(baseDataSetPath)
{
}

TrainingDataSet::TrainingDataSet(const TrainingDataSet& trainingDataSet) : dataSet(trainingDataSet.baseDataSetPath) {
	this->baseDataSetPath = trainingDataSet.baseDataSetPath;
	this->images = trainingDataSet.images;
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
	return dataSet.getNrOfImages();
}

void TrainingDataSet::iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth)> func) const {

	
	int idx = 0;
	for (auto& pair : images) {

		if (canSelectFunc(pair.first)) {
			auto imgs = dataSet.getImagesForNumber(pair.first);
			cv::Mat rgb = imgs[0];
			cv::Mat depth = imgs[1];

			for (auto& r : pair.second.regions) {

				cv::Mat regionRGB;
				cv::resize(rgb(r.region), regionRGB, cv::Size2d(refWidth, refHeight));

				cv::Mat regionDepth;
				cv::resize(depth(r.region), regionDepth, cv::Size2d(refWidth, refHeight));


				func(idx, r.regionClass, pair.first, r.region, regionRGB, regionDepth);
				idx++;

				cv::Mat rgbFlipped;
				cv::Mat depthFlipped;
				cv::flip(regionRGB, rgbFlipped, 1);
				cv::flip(regionDepth, depthFlipped, 1);
				func(idx, r.regionClass, pair.first, r.region, rgbFlipped, depthFlipped);
				idx++;
			}
		}
	}
}


void TrainingDataSet::iterateDataSetWithSlidingWindow(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb)> func) const {

	KITTIDataSet dataSet(baseDataSetPath);
	int idx = 0;

	int parallization = 7;
	parallel_foreach<int,TrainingImage>(images, parallization, [&](std::pair<int, TrainingImage> pair) -> void {
		//for (auto& pair : images) {

		if (canSelectFunc(pair.first)) {
			auto imgs = dataSet.getImagesForNumber(pair.first);
			cv::Mat mRGB = imgs[0];
			cv::Mat mDepth = imgs[1];


			std::vector<cv::Rect> truePositiveRegions;
			for (auto& r : pair.second.regions) {
				if (r.regionClass == 1)
					truePositiveRegions.push_back(r.region);
			}

			cv::Mat tmp = mRGB.clone();
			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {
				cv::Mat regionRGB;
				cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));

				cv::Mat regionDepth;
				cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));


				int resultClass = -1;
				bool overlapsWithTruePositive = false;
				for (int i = 0; i < truePositiveRegions.size() && !overlapsWithTruePositive; i++)
				{
					cv::Rect tp = truePositiveRegions[i];
					double intersectionRect = (tp & bbox).area();
					double unionRect = (tp | bbox).area();
					if (unionRect > 0 && intersectionRect / unionRect > 0.5) {
						resultClass = 1;
						overlapsWithTruePositive = true;
						break;
					}
				}
				func(idx, resultClass, pair.first, bbox, regionRGB, regionDepth, tmp);
				idx++;
			}, 0.5, 2, 16);
			//cv::imshow("Temp", tmp);
			//cv::waitKey(0);


		}
		//}
	});
}

std::string TrainingDataSet::getBaseDataSetPath() const
{
	return baseDataSetPath;
}
