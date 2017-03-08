#pragma once
#include "ClassifierEvaluation.h"
#include "opencv2/opencv.hpp"
#include "FeatureSet.h"
#include "TrainingDataSet.h"
#include "FeatureSet.h"
#include "FeatureVector.h"
#include <functional>

struct EvaluationResult {
	int resultClass;
	double rawResponse;
	EvaluationResult(int resultClass, double rawResponse) : resultClass(resultClass), rawResponse(rawResponse) { }
};



struct SlidingWindowRegion {
	int imageNumber;
	cv::Rect bbox;
	SlidingWindowRegion(int imageNumber, cv::Rect bbox) : imageNumber(imageNumber), bbox(bbox) { }
};

struct EvaluationSlidingWindowResult {
	std::vector<ClassifierEvaluation> evaluations;
	std::vector<SlidingWindowRegion> worstFalsePositives;
	//std::vector<SlidingWindowRegion> worstFalseNegatives;

	int evaluationIndexWhenTPRThresholdIsReached;
};

class IEvaluator
{
protected:
	std::string name;
	
	int slidingWindowEveryXImage = 1;
	int evaluationRange = 30;

public:
	IEvaluator(std::string& name);
	virtual ~IEvaluator();

	
	std::vector<ClassifierEvaluation> evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set,int nrOfEvaluations, bool includeRawResponses, std::function<bool(int imageNumber)> canSelectFunc) const;

	EvaluationSlidingWindowResult evaluateWithSlidingWindow(const TrainingDataSet& trainingDataSet, const FeatureSet& set, int nrOfEvaluations, int trainingRound, float tprToObtainWorstFalsePositives, int maxNrOfFalsePosOrNeg) const;

	//double evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const;

	virtual double evaluateFeatures(FeatureVector& v) const = 0;
};

