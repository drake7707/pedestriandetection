#pragma once
#include "DataSet.h"

class KAISTDataSet :
	public DataSet
{

private:
	std::string folderPath;
public:
	KAISTDataSet(std::string& folderPath);

	virtual std::string getName() const;

	virtual std::vector<DataSetLabel> getLabels() const;
	virtual std::vector<cv::Mat> getImagesForNumber(int number) const;

	virtual int getNrOfImages() const;

	virtual bool isWithinValidDepthRange(int height, float depthAverage) const;

	virtual std::vector<std::string> getCategories() const;

	virtual std::string getCategory(DataSetLabel* label) const;

	virtual std::vector<bool> getFullfillsRequirements() const;

};

