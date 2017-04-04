#pragma once
#include "IFeatureCreator.h"

class CoOccurenceMatrixFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 16;
	int refWidth = 64;
	int refHeight = 128;

public:
	CoOccurenceMatrixFeatureCreator(std::string& name, int patchSize = 8, int binSize = 16, int refWidth = 64, int refHeight = 128);
	virtual ~CoOccurenceMatrixFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat CoOccurenceMatrixFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;

};

