#pragma once
#include "IFeatureCreator.h"
class HDDFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;

public:
	HDDFeatureCreator(std::string& name, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HDDFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat HDDFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	virtual std::vector<bool> getRequirements() const;
};

