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
	OcclusionEnum occlusion;
	double truncation;
public:


	float height;
	float width;
	float length;

	float x_3d;
	float y_3d;
	float z_3d;


	DataSetLabel(const std::string& number, const cv::Rect2d& bbox, OcclusionEnum occlusion, double truncation, bool isDontCare) : number(number), bbox(bbox), occlusion(occlusion), truncation(truncation) , isDontCare(isDontCare) {

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

	std::string getCategory() {
		//Easy: Min.bounding box height : 40 Px, Max.occlusion level : Fully visible, Max.truncation : 15 %
		//Moderate : Min.bounding box height : 25 Px, Max.occlusion level : Partly occluded, Max.truncation : 30 %
		//Hard : Min.bounding box height : 25 Px, Max.occlusion level : Difficult to see, Max.truncation : 50 %

		//double intersectionArea = (bbox & cv::Rect2d(0, 0, img.cols, img.rows)).area();
		//double truncationPercentage = (bbox.area() - intersectionArea) / bbox.area();

		if (bbox.height >= 40 && (occlusion <= OcclusionEnum::FullyVisible) && truncation <= 0.15)
			return "easy";
		else if (bbox.height >= 25 && (occlusion <= OcclusionEnum::PartlyOccluded) && truncation <= 0.30)
			return "moderate";
		else if (bbox.height >= 25 && (occlusion <= OcclusionEnum::DifficultToSee) && truncation <= 0.50)
			return "hard";

		return "";
	}
};

