#include "KAISTDataSet.h"
#include "Helper.h"
#include <fstream>


KAISTDataSet::KAISTDataSet(std::string& folderPath) :folderPath(folderPath) {

}
std::string KAISTDataSet::getName() const {
	return "KAIST";
}


std::vector<DataSetLabel> KAISTDataSet::getLabels() const {
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
				if (line[0] != '%') {
					std::vector<std::string> parts = splitString(line, ' ');

					//for example person 40 226 25 52 0 0 0 0 0 0 0
					if (parts.size() >= 5) {
						std::string type = parts[0];


						double left = atof(parts[1].c_str());
						double top = atof(parts[2].c_str());
						double width = atof(parts[3].c_str());
						double height = atof(parts[4].c_str());

						cv::Rect2d r(left, top, width, height);

						if (type == "person") {
							DataSetLabel lbl(nrStr, r, OcclusionEnum::Unknown, 0, false);
							lbl.width = width;
							lbl.height = height;

							// estimated from paper 11~28m corresponds to 45~115 pixels and depth/height relation is linear

							// d = a / h + b
							
							// solving to a and b:
							// 11 = a * 1/115 + b
							// 28 = a * 1/45 +  b

							double a = (28.0 - 11.0) / (1 / 45.0 - 1 / 115.0);
							double b = 11.0 - a*(1 / 115.0);
							lbl.z_3d = a / height + b;

							labels.push_back(lbl);
						}
						else if (type == "people" || type == "cyclist") {
							DataSetLabel lbl(nrStr, r, OcclusionEnum::Unknown, 0, true);
							labels.push_back(lbl);
						}
					}
				}
			}
		}
		nr++;
	}
	return labels;
}

std::vector<cv::Mat> KAISTDataSet::getImagesForNumber(int number) const {
	char nrStr[7];
	sprintf(nrStr, "%06d", number);

	std::string rgbPath = folderPath + PATH_SEPARATOR + "rgb" + PATH_SEPARATOR + nrStr + ".jpg";
	std::string thermalPath = folderPath + PATH_SEPARATOR + "thermal" + PATH_SEPARATOR + nrStr + ".jpg";

	cv::Mat rgb = cv::imread(rgbPath);
	cv::Mat depth;

	cv::Mat thermal = cv::imread(thermalPath);
	cv::cvtColor(thermal, thermal, CV_BGR2GRAY);
	thermal.convertTo(thermal, CV_32FC1, 1 / 255.0);

	if (rgb.rows == 0 || rgb.cols == 0) {
		throw std::exception(std::string("RGB image " + rgbPath + " is corrupt").c_str());
	}
	if (thermal.rows == 0 || thermal.cols == 0) {
		throw std::exception(std::string("Thermal image " + thermalPath + " is corrupt").c_str());
	}

	return{ rgb, depth, thermal };
}

int KAISTDataSet::getNrOfImages() const
{
	return 4767;
}


std::vector<std::string> KAISTDataSet::getCategories() const {
	return{};
}

std::string KAISTDataSet::getCategory(DataSetLabel* label) const {
	return "";
}

bool KAISTDataSet::isWithinValidDepthRange(int height, float depthAverage) const {
	// no depth available
	return true;
}

std::vector<bool> KAISTDataSet::getFullfillsRequirements() const {
	return{ true,false, true }; // rgb and thermal only
}

bool KAISTDataSet::canDoRiskAnalysis() const {
	// label depth are approximated
	return true;
}
