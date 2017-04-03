#pragma once
#include <string>
#include <functional>

#include "FeatureSet.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "ClassifierEvaluation.h"
#include "TrainingDataSet.h"
#include "ProgressWindow.h"
#include "IEvaluator.h"
#include "EvaluationSettings.h"


struct Model {
	std::vector<float> meanVector;
	std::vector<float> sigmaVector;
	cv::Ptr<cv::ml::Boost> boost;
};


class ModelEvaluator : public IEvaluator
{
	Model model;

public:
	ModelEvaluator(std::string& name);
	virtual ~ModelEvaluator();

	/// <summary>
	/// Returns the name of the model
	/// </summary>
	std::string getName() const;

	/// <summary>
	/// Returns the model on the given training set, feature set and settings. Only the images numbers that can be selected will be used as samples for training
	/// </summary>
	void train(const TrainingDataSet& trainingDataSet, const FeatureSet& set, const EvaluationSettings& settings, std::function<bool(int number)> canSelectFunc);

	/// <summary>
	/// Evaluates the model based on the feature vector and returns the score and predicted class (sign)
	/// </summary>
	double evaluateFeatures(FeatureVector& v);

	/// <summary>
	/// Explains the features used as weak classifier decision tree stumps
	/// </summary>
	std::vector<cv::Mat> ModelEvaluator::explainModel(const std::unique_ptr<FeatureSet>& set, int refWidth, int refHeight) const;

	/// <summary>
	/// Saves the boost model and necessary data to the given file
	/// </summary>
	void saveModel(std::string& path);

	/// <summary>
	/// Loads a boost model and necessary data from the given path
	/// </summary>
	void loadModel(std::string& path);



};

