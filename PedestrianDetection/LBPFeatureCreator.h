#pragma once
#include "IFeatureCreator.h"
class LBPFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;
	bool onDepth;

public:
	LBPFeatureCreator(std::string& name, bool onDepth, int patchSize = 8, int binSize = 20, int refWidth = 64, int refHeight = 128);
	virtual ~LBPFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	virtual std::vector<bool> getRequirements() const;
};

