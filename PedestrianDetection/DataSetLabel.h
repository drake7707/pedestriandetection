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

public:
	DataSetLabel(const std::string& number, const cv::Rect2d& bbox) : number(number), bbox(bbox) {

	}

	~DataSetLabel() {

	}

	std::string getNumber()  const {
		return this->number;
	}
	cv::Rect2d getBbox()  const {
		return bbox;
	}
};

