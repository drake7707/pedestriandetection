#pragma once
#include "DataSet.h"
#include <string>

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 

class KITTIDataSet :
	public DataSet
{

private:
	std::string folderPath;

public:
	KITTIDataSet(const std::string& folderPath) : DataSet(), folderPath(folderPath) { }
	~KITTIDataSet() {}

	virtual std::vector<DataSetLabel> getLabels() const;
	virtual std::vector<cv::Mat> getImagesForNumber(const std::string& number) const;

};

