#pragma once

#include <string>
#include "opencv2/opencv.hpp"

struct Bbox {

};
class DataSetLabel
{

private:
	std::string number;
	cv::Rect2d bbox;
	bool isDontCare;

public:
	DataSetLabel(const std::string& number, const cv::Rect2d& bbox, bool isDontCare) : number(number), bbox(bbox), isDontCare(isDontCare) {

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

