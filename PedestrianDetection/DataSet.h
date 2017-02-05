#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DataSetLabel.h"

class DataSet
{
public:
	DataSet();
	~DataSet();

	virtual std::vector<DataSetLabel> getLabels() const = 0;

	virtual std::vector<cv::Mat> getImagesForNumber(const std::string& number) const = 0;

};

