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

	/// <summary>
	/// Calculates the histogram of depth differences from the given image
	/// </summary>
	hog::HistogramResult HDDFeatureCreator::getHistogramsOfDepthDifferences(cv::Mat& img, int patchSize, int binSize, cv::Rect& roi, 
		const IPreparedData* preparedData, bool createImage, bool l2normalize) const;

	/// <summary>
	/// Calculates the magnitude and angle images from the given img
	/// </summary>
	void buildMagnitudeAndAngle(cv::Mat& img, cv::Mat& magnitude, cv::Mat& angle) const;


public:
	HDDFeatureCreator(std::string& name, int patchSize = 8, int binSize = 9, int refWidth = 64, int refHeight = 128);
	virtual ~HDDFeatureCreator();

	/// <summary>
	/// Returns the number of features a feature vector will contain
	/// </summary>
	int getNumberOfFeatures() const;

	/// <summary>
	/// Creates a heat map of the occurrences of the features
	/// </summary>
	cv::Mat HDDFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	/// <summary>
	/// Prepares integral histograms for each scale to quickly evaluate windows at each scale
	/// </summary>
	std::vector<IPreparedData*> buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const;

	/// <summary>
	/// Creates a feature vector of the given input data. Prepared data will be used if given
	/// </summary>
	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const;

	/// <summary>
	/// Returns the requirements to run this feature descriptor (Depth)
	/// </summary>
	virtual std::vector<bool> getRequirements() const;
};

