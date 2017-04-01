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
	std::map<std::string, std::vector<ClassifierEvaluation>> evaluations;
	std::vector<ClassifierEvaluation> combinedEvaluations;


	std::map<int, std::vector<std::vector<cv::Rect>>> missedPositivesPerImage;
};


class IEvaluator
{
protected:


	std::string name;

	//int refWidth = 64;
	//int refHeight = 128;
	//int parallelization = 8;
	//int baseWindowStride = 16;
	//int slidingWindowEveryXImage = 1;


public:
	IEvaluator(std::string& name);
	virtual ~IEvaluator();


	float getValueShift(int i, int nrOfEvaluations, float evaluationRange) const;

	std::vector<ClassifierEvaluation> evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set, const EvaluationSettings& settings, bool includeRawResponses, std::function<bool(int imageNumber)> canSelectFunc);

	EvaluationSlidingWindowResult evaluateWithSlidingWindow(const EvaluationSettings& settings,
		const DataSet* dataSet, const FeatureSet& set, int trainingRound,
		std::function<bool(int number)> canSelectFunc);

	FinalEvaluationSlidingWindowResult evaluateWithSlidingWindowAndNMS(const EvaluationSettings& settings,
		const DataSet* dataSet, const FeatureSet& set, std::function<bool(int number)> canSelectFunc);

	//double evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const;

	virtual double evaluateFeatures(FeatureVector& v) = 0;
};

