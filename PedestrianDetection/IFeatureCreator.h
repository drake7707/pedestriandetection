#pragma once
#include "FeatureVector.h"
#include "opencv2/opencv.hpp"

class IFeatureCreator
{

private:
	std::string name;

protected:
	cv::Mat explainFeaturesHeatMap(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight, int patchSize) const {

	}

public:
	IFeatureCreator(std::string& name);
	virtual ~IFeatureCreator();

	std::string getName() const;

	virtual int getNumberOfFeatures() const = 0;

	virtual FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const = 0;

	virtual cv::Mat explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const = 0;

	virtual std::vector<bool> getRequirements() const = 0;


	enum Target {
		RGB = 0,
		Depth = 1,
		Thermal = 2
	};
};

