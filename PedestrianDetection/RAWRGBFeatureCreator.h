#pragma once
#include "IFeatureCreator.h"

class RAWRGBFeatureCreator : public IFeatureCreator
{

private:
	int refWidth;
	int refHeight;

public:
	RAWRGBFeatureCreator(std::string& name, int refWidth, int refHeight);
	virtual ~RAWRGBFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;

};

