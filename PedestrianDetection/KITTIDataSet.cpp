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
					if (type == "Pedestrian") {
						// 0 1 2 3 | 4 5 6 7 |
						double left = atof(parts[4].c_str());
						double top = atof(parts[5].c_str());
						double right = atof(parts[6].c_str());
						double bottom = atof(parts[7].c_str());
						cv::Rect2d r(left, top, right - left + 1, bottom - top + 1);
						DataSetLabel lbl(nrStr, r);
						labels.push_back(lbl);
					}
				}
			}
		}
		nr++;
	}
	return labels;
}

std::vector<cv::Mat> KITTIDataSet::getImagesForNumber(const std::string& number) const {
	std::string rgbPath = folderPath + PATH_SEPARATOR + "rgb" + PATH_SEPARATOR + number + ".png";
	std::string depthPath = folderPath + PATH_SEPARATOR + "depth" + PATH_SEPARATOR + number + ".png";

	cv::Mat rgb = cv::imread(rgbPath);
	cv::Mat depth = cv::imread(depthPath);

	return{ rgb, depth };
}