#include "KITTIDataSet.h"
#include <fstream>

std::vector<DataSetLabel> KITTIDataSet::getLabels() const {
	std::string baseLabelPath = folderPath + PATH_SEPARATOR + "labels";

	std::vector<DataSetLabel> labels;

	int nr = 0;
	bool stop = false;
	while (!stop) {
		char nrStr[7];
		sprintf(nrStr, "%06d", nr);

		std::string fullPath = baseLabelPath + PATH_SEPARATOR + std::string(nrStr) + ".txt";

		std::ifstream istr(fullPath);
		if (!istr.is_open()) {
			stop = true;
		}
		else {
			std::string line;
			while (std::getline(istr, line)) {
				std::vector<std::string> parts;
				std::stringstream ss(line);
				std::string token;
				while (ss >> token)
					parts.push_back(token);

				if (parts.size() >= 15) {
					std::string type = parts[0];

					double truncation = atof(parts[1].c_str());
					int occlusion = atoi(parts[2].c_str());
					double left = atof(parts[4].c_str());
					double top = atof(parts[5].c_str());
					double right = atof(parts[6].c_str());
					double bottom = atof(parts[7].c_str());
					cv::Rect2d r(left, top, right - left + 1, bottom - top + 1);


					if (type == "Pedestrian") {
						// 0 1 2 3 | 4 5 6 7 |
						DataSetLabel lbl(nrStr, r, (OcclusionEnum)occlusion, truncation, false);
						labels.push_back(lbl);
					}
					else if (type == "Person_sitting" || type == "Cyclist" || type == "DontCare") {
						DataSetLabel lbl(nrStr, r, (OcclusionEnum)occlusion, truncation, true);
						labels.push_back(lbl);
					}
				}
			}
		}
		nr++;
	}
	return labels;
}

std::vector<cv::Mat> KITTIDataSet::getImagesForNumber(int number) const {
	char nrStr[7];
	sprintf(nrStr, "%06d", number);

	std::string rgbPath = folderPath + PATH_SEPARATOR + "rgb" + PATH_SEPARATOR + nrStr + ".png";
	std::string depthPath = folderPath + PATH_SEPARATOR + "depth" + PATH_SEPARATOR + nrStr + ".png";

	cv::Mat rgb = cv::imread(rgbPath);
	cv::Mat depth = cv::imread(depthPath, CV_LOAD_IMAGE_UNCHANGED);
	depth.convertTo(depth, CV_32FC1, 1.0 / 0xFFFF, 0);

	if (rgb.rows == 0 || rgb.cols == 0) {
		throw std::exception(std::string("RGB image " + rgbPath + " is corrupt").c_str());
	}
	if (depth.rows == 0 || depth.cols == 0) {
		throw std::exception(std::string("Depth image " + depthPath + " is corrupt").c_str());
	}

	return{ rgb, depth };
}

int KITTIDataSet::getNrOfImages() const
{
	return 7481;
}

bool KITTIDataSet::isWithinValidDepthRange(int height, float depthAverage) const {
	// scale to 0-80
	int idx = height;
	if (idx >= 0 && idx < validdepthrange.size()) {
		double depthAvg = 80 - 80 * (depthAverage);
		if (depthAvg >= validdepthrange[idx].first && depthAvg < validdepthrange[idx].second) {
			// falls within range where TP can lie, continue
			return true;
		}
		else {
			// reject outright, will most likely never be TP
			return false;
		}
	}
	return true;
}