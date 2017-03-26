#pragma once
#include "ClassifierEvaluation.h"
#include "opencv2/opencv.hpp"
#include "FeatureSet.h"
#include "TrainingDataSet.h"
#include "FeatureSet.h"
#include "FeatureVector.h"
#include <functional>
#include "Helper.h"

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
};


class IEvaluator
{
protected:


	std::string name;

	int refWidth = 64;
	int refHeight = 128;
	int parallelization = 8;
	int baseWindowStride = 16;
	int slidingWindowEveryXImage = 1;
	int evaluationRange = 60;

public:
	IEvaluator(std::string& name);
	virtual ~IEvaluator();


	std::vector<ClassifierEvaluation> evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set, int nrOfEvaluations, bool includeRawResponses, std::function<bool(int imageNumber)> canSelectFunc) const;

	EvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindow(std::vector<cv::Size>& windowSizes,
		const DataSet* dataSet, const FeatureSet& set, int nrOfEvaluations, int trainingRound,
		float tprToObtainWorstFalsePositives, int maxNrOfFalsePosOrNeg, std::function<bool(int number)> canSelectFunc) const;

	FinalEvaluationSlidingWindowResult evaluateWithSlidingWindowAndNMS(std::vector<cv::Size>& windowSizes, 
		const DataSet* dataSet, const FeatureSet& set, int nrOfEvaluations, std::function<bool(int number)> canSelectFunc,
		int refWidth = 64, int refHeight = 128, int paralellization = 8) const;

	//double evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const;

	virtual double evaluateFeatures(FeatureVector& v) const = 0;
};

