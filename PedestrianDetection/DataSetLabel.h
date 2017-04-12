#pragma once

#include <string>
#include "opencv2/opencv.hpp"

struct Bbox {

};

enum OcclusionEnum {
	FullyVisible = 0,
	PartlyOccluded = 1,
	DifficultToSee = 2,
	Unknown = 3
};

class DataSetLabel
{

private:
	std::string number;
	cv::Rect2d bbox;


	bool isDontCare;

public:

	OcclusionEnum occlusion = OcclusionEnum::Unknown;
	double truncation = 0;
	float height = 0;
	float width = 0;
	float length = 0;

	float x_3d = 0;
	float y_3d = 0;
	float z_3d = 0;


	DataSetLabel(const std::string& number, const cv::Rect2d& bbox, OcclusionEnum occlusion, double truncation, bool isDontCare) : number(number), bbox(bbox), occlusion(occlusion), truncation(truncation), isDontCare(isDontCare) {

	}

	~DataSetLabel() {

	}

	std::string getNumber()  const {
		return this->number;
	}
	cv::Rect2d getBbox()  const {
		return bbox;
	}

	bool isDontCareArea() {
		return this->isDontCare;
	}
};

