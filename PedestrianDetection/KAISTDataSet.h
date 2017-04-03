#pragma once
#include "DataSet.h"

class KAISTDataSet :
	public DataSet
{

private:
	std::string folderPath;
public:
	KAISTDataSet(std::string& folderPath);

	/// <summary>
	/// Returns the name of the data set
	/// </summary>
	virtual std::string getName() const;
	
	/// <summary>
	/// Returns all the labels of the data set
	/// </summary>
	virtual std::vector<DataSetLabel> getLabels() const;

	/// <summary>
	/// Returns the images (RGB/Depth/Thermal) for the given image number. Depth is not available in KAIST
	/// </summary>
	virtual std::vector<cv::Mat> getImagesForNumber(int number) const;
	
	/// <summary>
	/// Returns the number of images in the data set
	/// </summary>
	virtual int getNrOfImages() const;

	/// <summary>
	/// Checks whether the height of a window and the depth average in the middle of the window can be a true positive
	/// </summary>
	virtual bool isWithinValidDepthRange(int height, float depthAverage) const;

	/// <summary>
	/// Returns the various categories possible, in KAIST there are no specific categories
	/// </summary>
	virtual std::vector<std::string> getCategories() const;

	/// <summary>
	/// Returns the category of the data set label
	/// </summary>
	virtual std::string getCategory(DataSetLabel* label) const;

	/// <summary>
	/// Returns which requirements the data set fullfills: RGB,Depth,Thermal. KAIST only has RGB and Thermal
	/// </summary>
	virtual std::vector<bool> getFullfillsRequirements() const;

};

