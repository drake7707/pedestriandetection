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

	/// <summary>
	/// Calculates the magnitude and angle images from the given img
	/// </summary>
	void buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const;


public:
	HOGFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HOGFeatureCreator();

	/// <summary>
	/// Returns the number of features a feature vector will contain
	/// </summary>
	int getNumberOfFeatures() const;

	/// <summary>
	/// Creates a heat map of the occurrences of the features
	/// </summary>
	cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	virtual std::unique_ptr<IPreparedData> buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const;

	/// <summary>
	/// Determines the HOG result from gradient magnitude and orientation
	/// </summary>
	hog::HistogramResult getHistogramsOfOrientedGradient(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, const IPreparedData* preparedData, bool createImage, bool l2normalize) const;

	/// <summary>
	/// Creates a feature vector of the given input data. Prepared data will be used if given
	/// </summary>
	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	/// <summary>
	/// Returns the requirements to run this feature descriptor (Depth)
	/// </summary>
	virtual std::vector<bool> getRequirements() const;
};

