#pragma once
#include "ClassifierEvaluation.h"
#include "opencv2/opencv.hpp"
#include "FeatureSet.h"
#include "TrainingDataSet.h"
#include "FeatureSet.h"
#include "FeatureVector.h"
#include <functional>
#include "Helper.h"
#include "EvaluationSettings.h"


struct EvaluationResult {
	int resultClass;
	double rawResponse;
	EvaluationResult(int resultClass, double rawResponse) : resultClass(resultClass), rawResponse(rawResponse) { }
};

struct EvaluationSlidingWindowResult {
	std::vector<ClassifierEvaluation> evaluations;
	std::vector<SlidingWindowRegion> worstFalsePositives;
	//std::vector<SlidingWindowRegion> worstFalseNegatives;

	int evaluationIndexWhenTPRThresholdIsReached;
};


struct FinalEvaluationSlidingWindowResult {

	/// <summary>
	/// Evaluation per category and per decision boundary shift
	/// </summary>
	std::map<std::string, std::vector<ClassifierEvaluation>> evaluations;

	/// <summary>
	/// Evaluations per decision boundary shift
	/// </summary>
	std::vector<ClassifierEvaluation> combinedEvaluations;

	/// <summary>
	/// Missed positive regions per image and per decision boundary shift
	/// </summary>
	std::map<int, std::vector<std::vector<cv::Rect>>> missedPositivesPerImage;
};


class IEvaluator
{
protected:


	std::string name;

public:
	IEvaluator(std::string& name);
	virtual ~IEvaluator();

	/// <summary>
	/// Calculates the decisioun boundary shift for the given index i
	/// </summary>
	float getValueShift(int i, int nrOfEvaluations, float evaluationRange) const;

	/// <summary>
	/// Evaluates the given training set with the given settings
	/// </summary>
	std::vector<ClassifierEvaluation> evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set, 
		const EvaluationSettings& settings, std::function<bool(int imageNumber)> canSelectFunc);

	/// <summary>
	/// Evaluates the data set with a sliding window with the given settings
	/// </summary>
	EvaluationSlidingWindowResult evaluateWithSlidingWindow(const EvaluationSettings& settings,
		const DataSet* dataSet, const FeatureSet& set, std::function<bool(int number)> canSelectFunc);

	/// <summary>
	/// Evaluate the data set with a sliding window and apply non maximum suppression with the given settings and feature set
	/// </summary>
	FinalEvaluationSlidingWindowResult evaluateWithSlidingWindowAndNMS(const EvaluationSettings& settings,
		const DataSet* dataSet, const FeatureSet& set, std::function<bool(int number)> canSelectFunc);

	/// <summary>
	/// Evaluates the given feature vector and returns the score and class (sign)
	/// </summary>
	virtual double evaluateFeatures(FeatureVector& v) = 0;
};

