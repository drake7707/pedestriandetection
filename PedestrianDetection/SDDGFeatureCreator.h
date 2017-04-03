#pragma once
#include "IFeatureCreator.h"
#include <functional>

class SDDGFeatureCreator :
	public IFeatureCreator
{

	int refWidth = 64;
	int refHeight = 128;


	int cellSize = 9;

	IFeatureCreator::Target target;

	void bresenhamLine(int srcX, int srcY, int dstX, int dstY, std::function<void(int x, int y)> setPixelFunc) const;
	std::vector<float> calculateSDDG(int nrOfCellsX, int nrOfCellsY, std::function<float(int x, int y)> gradientAt) const;

public:



	SDDGFeatureCreator::SDDGFeatureCreator(std::string& name, IFeatureCreator::Target target, int refWidth, int refHeight);

	virtual SDDGFeatureCreator::~SDDGFeatureCreator();


	int SDDGFeatureCreator::getNumberOfFeatures() const;

	FeatureVector SDDGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	cv::Mat SDDGFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	std::vector<bool> SDDGFeatureCreator::getRequirements() const;

};

