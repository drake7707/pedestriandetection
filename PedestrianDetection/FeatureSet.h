#pragma once
#include "FeatureVector.h"
#include "IFeatureCreator.h"
#include "opencv2/opencv.hpp"
#include "TrainingDataSet.h"
#include <memory>

class FeatureSet
{
private:
	std::vector<std::unique_ptr<IFeatureCreator>> creators;

public:
	FeatureSet();

	FeatureSet(FeatureSet&& set) {
		this->creators = std::move(creators);
	}

	FeatureSet(const FeatureSet& set) = delete;

	~FeatureSet();

	void addCreator(std::unique_ptr<IFeatureCreator> creator);

	int size() const;

	void prepare(TrainingDataSet& trainingDataSet, int trainingRound);

	int getNumberOfFeatures() const;
	FeatureVector getFeatures(cv::Mat& rgb, cv::Mat& depth) const;

	std::vector<cv::Mat> FeatureSet::explainFeatures(std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;
};

