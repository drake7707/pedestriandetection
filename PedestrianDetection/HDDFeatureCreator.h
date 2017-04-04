#pragma once
#include "IFeatureCreator.h"
#include "IPreparedData.h"
#include "HistogramOfOrientedGradients.h"

class HDDFeatureCreator : public IFeatureCreator
{

private:
	int patchSize = 8;
	int binSize = 9;
	int refWidth = 64;
	int refHeight = 128;


	hog::HistogramResult HDDFeatureCreator::getHistogramsOfDepthDifferences(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, 
		const IPreparedData* preparedData, bool createImage, bool l2normalize) const;
	void buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const;


public:
	HDDFeatureCreator(std::string& name, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HDDFeatureCreator();

	int getNumberOfFeatures() const;
	cv::Mat HDDFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	std::vector<IPreparedData*> buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const;

	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	virtual std::vector<bool> getRequirements() const;
};

