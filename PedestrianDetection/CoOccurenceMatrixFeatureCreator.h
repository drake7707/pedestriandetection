#pragma once
#include "IFeatureCreator.h"
#include "IntegralHistogram2D.h"
#include "IntHist2DPreparedData.h"
#include "CoOccurenceMatrix.h"

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

	/// <summary>
	/// Returns the number of features a feature vector will contain
	/// </summary>
	int getNumberOfFeatures() const;

	/// <summary>
	/// Creates a heat map of the occurrences of the features
	/// </summary>
	cv::Mat CoOccurenceMatrixFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const;

	std::unique_ptr<IPreparedData> CoOccurenceMatrixFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const;

	/// <summary>
	/// Creates a feature vector of the given input data
	/// </summary>
	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	/// <summary>
	/// Returns the requirements to run this feature descriptor (RGB)
	/// </summary>
	virtual std::vector<bool> getRequirements() const;

};

