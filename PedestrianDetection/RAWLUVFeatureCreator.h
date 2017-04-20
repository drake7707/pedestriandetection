#pragma once
#include "IFeatureCreator.h"

class RAWLUVFeatureCreator : public IFeatureCreator
{

private:
	int refWidth;
	int refHeight;

public:
	RAWLUVFeatureCreator(std::string& name, int refWidth, int refHeight);
	virtual ~RAWLUVFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& LUV, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;

};

