#pragma once
#include "FeatureVector.h"
#include "IFeatureCreator.h"
#include "opencv2/opencv.hpp"
#include "TrainingDataSet.h"
#include <memory>
#include "EvaluationSettings.h"

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

	/// <summary>
	/// Adds a feature descriptor to the set
	/// </summary>

	void addCreator(std::unique_ptr<IFeatureCreator> creator);

	/// <summary>
	/// Returns the number of feature descriptors in the set
	/// </summary>
	int size() const;

	/// <summary>
	/// Prepares all feature descriptors that require preparing in order to evaluate the feature set
	/// </summary>
	void FeatureSet::prepare(TrainingDataSet& trainingDataSet, const EvaluationSettings& settings);

	/// <summary>
	/// Returns the total number of features that will be in the combined feature vector
	/// </summary>
	int getNumberOfFeatures() const;

	/// <summary>
	/// Builds feature vectors from the given window input data from all the separarate feature descriptor
	/// </summary>
	FeatureVector FeatureSet::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const;

	/// <summary>
	/// Explains the various features from the trained models, usually in the form of a heat map
	/// </summary>
	std::vector<cv::Mat> FeatureSet::explainFeatures(std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const;

	/// <summary>
	/// Returns the combination of requirements for all the feature descriptors to match against the data set RGB/Depth/Thermal
	/// </summary>
	std::vector<bool> FeatureSet::getRequirements() const;
};

