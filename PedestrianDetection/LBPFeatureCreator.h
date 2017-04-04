#pragma once
#include "IFeatureCreator.h"
class LBPFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;
	IFeatureCreator::Target target;

public:
	LBPFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize = 8, int binSize = 20, int refWidth = 64, int refHeight = 128);
	virtual ~LBPFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	std::vector<IPreparedData*> buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const;

	void buildWeightAndBinningValues(cv::Mat& img, cv::Mat& weights, cv::Mat& binningValues) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};

