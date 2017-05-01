#pragma once
#include "IFeatureCreator.h"
#include "IntHist2DPreparedData.h"
#include "HistogramResult.h"

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

	std::unique_ptr<IPreparedData> HONVFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	cv::Mat buildAngleMat(cv::Mat depth) const;

	virtual std::vector<bool> getRequirements() const;
};

