#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"
#include "IPreparedData.h"
#include <memory>

class IFeatureCreator
{

private:
	std::string name;

public:
	IFeatureCreator(std::string& name);
	virtual ~IFeatureCreator();

	/// <summary>
	/// Returns the (unique) name of this feature descriptor
	/// </summary>
	std::string getName() const;

	/// <summary>
	/// Returns the number of features that will be in a feature vector
	/// </summary>
	virtual int getNumberOfFeatures() const = 0;

	/// <summary>
	/// Obtains prepared data of the used feature descriptors. This prepared data will be passed along during evaluation.
	/// Feature descriptors that prepare data can often evaluate much quicker (e.g. integral histograms)
	/// </summary>
	virtual std::unique_ptr<IPreparedData> IFeatureCreator::buildPreparedDataForFeatures(cv::Mat& rgbScale, cv::Mat& depthScale, cv::Mat& thermalScale) const;

	/// <summary>
	/// Builds a feature vector from the given window input data, which are all at reference size if available
	/// </summary>
	virtual FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const IPreparedData* preparedData) const = 0;

	/// <summary>
	/// Explains the various features from the trained models, usually in the form of a heat map
	/// </summary>
	virtual cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, int refWidth, int refHeight) const = 0;

	/// <summary>
	/// Returns the requirements for the feature descriptor to match against the data set RGB/Depth/Thermal
	/// </summary>
	virtual std::vector<bool> getRequirements() const = 0;


	enum Target {
		RGB = 0,
		Depth = 1,
		Thermal = 2
	};
};

