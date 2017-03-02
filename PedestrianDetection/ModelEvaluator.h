#pragma once
#include <string>
#include <functional>

#include "FeatureSet.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "ClassifierEvaluation.h"
#include "TrainingDataSet.h"



struct Model {
	std::vector<float> meanVector;
	std::vector<float> sigmaVector;
	cv::Ptr<cv::ml::Boost> boost;
};


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
	std::vector<SlidingWindowRegion> worstFalseNegatives;
};

class ModelEvaluator
{
private:
	const TrainingDataSet& trainingDataSet;
	const FeatureSet& set;

	Model model;

	EvaluationResult evaluateFeatures(FeatureVector& v, double valueShift = 0) const;

	int trainEveryXImage = 2;
	int slidingWindowEveryXImage = 10;

public:
	ModelEvaluator(const TrainingDataSet& trainingDataSet, const FeatureSet& set);
	~ModelEvaluator();

	
	

	void train();

	std::vector<ClassifierEvaluation> evaluateDataSet(int nrOfEvaluations, bool includeRawResponses);
	EvaluationResult evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift = 0) const;

	EvaluationSlidingWindowResult ModelEvaluator::evaluateWithSlidingWindow(int nrOfEvaluations, int trainingRound, float valueShiftForFalsePosOrNegCollection, int maxNrOfFalsePosOrNeg);

	void saveModel(std::string& path);
	void loadModel(std::string& path);
};

