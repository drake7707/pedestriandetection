#pragma once
#include "IFeatureCreator.h"
#include <functional>

class SDDGFeatureCreator :
	public IFeatureCreator
{

	int refWidth = 64;
	int refHeight = 128;


	int cellSize = 9;
	int SDDGLength = 56; // with cellSize = 9 the length is 56

	IFeatureCreator::Target target;

	void bresenhamLine(int srcX, int srcY, int dstX, int dstY, std::function<void(int x, int y)> setPixelFunc) const;
	std::vector<float> calculateSDDG(int nrOfCellsX, int nrOfCellsY, std::function<float(int x, int y)> gradientAt) const;

public:



	SDDGFeatureCreator(std::string& name, IFeatureCreator::Target target, int refWidth, int refHeight);

	virtual ~SDDGFeatureCreator();


	int getNumberOfFeatures() const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;
	
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	std::vector<bool> getRequirements() const;

};

