#pragma once
#include "IFeatureCreator.h"
#include "IPreparedData.h"
#include "HistogramOfOrientedGradients.h"

class HOGFeatureCreator : public IFeatureCreator
{

protected:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;
	IFeatureCreator::Target target;

	void buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const;


public:
	HOGFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HOGFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	std::vector<IPreparedData*> HOGFeatureCreator::buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const;

	hog::HistogramResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, const IPreparedData* preparedData, bool createImage, bool l2normalize) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};

