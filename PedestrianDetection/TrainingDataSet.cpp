#include "TrainingDataSet.h"
#include <fstream>
#include "Helper.h"


TrainingDataSet::TrainingDataSet(std::string& baseDataSetPath)
	: baseDataSetPath(baseDataSetPath)
{
}


TrainingDataSet::~TrainingDataSet()
{
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
		}

		images.emplace(img.number, img);
	}
}

void TrainingDataSet::iterateDataSet(std::function<bool(int number)> canSelectFunc, std::function<void(int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth)> func) const {
	int idx = 0;
	for (auto& pair : images) {

		if (canSelectFunc(pair.first)) {
			char nrStr[7];
			sprintf(nrStr, "%06d", pair.second.number);

			std::string rgbPath = baseDataSetPath + PATH_SEPARATOR + "rgb" + PATH_SEPARATOR + nrStr + ".png";
			std::string depthPath = baseDataSetPath + PATH_SEPARATOR + "depth" + PATH_SEPARATOR + nrStr + ".png";

			cv::Mat rgb = cv::imread(rgbPath);
			cv::Mat depth = cv::imread(depthPath, CV_LOAD_IMAGE_UNCHANGED);
			depth.convertTo(depth, CV_32FC1, 1.0 / 0xFFFF, 0);


			if (rgb.rows == 0 || rgb.cols == 0) {
				throw std::exception(std::string("RGB image " + rgbPath + " is corrupt").c_str());
			}
			if (depth.rows == 0 || depth.cols == 0) {
				throw std::exception(std::string("Depth image " + depthPath + " is corrupt").c_str());
			}

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

std::string TrainingDataSet::getBaseDataSetPath() const
{
	return baseDataSetPath;
}
