#pragma once
#include "IFeatureCreator.h"
class HONVFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;

public:
	HONVFeatureCreator(std::string& name, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HONVFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};

