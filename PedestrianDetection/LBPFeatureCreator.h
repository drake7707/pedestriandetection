#pragma once
#include "IFeatureCreator.h"
class LBPFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;

public:
	LBPFeatureCreator(std::string& name, int patchSize = 8, int binSize = 20, int refWidth = 64, int refHeight = 128);
	virtual ~LBPFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat LBPFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	virtual std::unique_ptr<IPreparedData> buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const;

	void buildWeightAndBinningValues(cv::Mat& img, cv::Mat& weights, cv::Mat& binningValues) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};

