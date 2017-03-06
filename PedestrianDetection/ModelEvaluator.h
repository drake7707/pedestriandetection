#pragma once
#include <string>
#include <functional>

#include "FeatureSet.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "ClassifierEvaluation.h"
#include "TrainingDataSet.h"
#include "ProgressWindow.h"


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
	std::string name;
	const TrainingDataSet& trainingDataSet;
	const FeatureSet& set;

	Model model;

	double evaluateFeatures(FeatureVector& v) const;

	int trainEveryXImage = 2;
	int slidingWindowEveryXImage = 1;

public:
	ModelEvaluator(std::string& name, const TrainingDataSet& trainingDataSet, const FeatureSet& set);
	~ModelEvaluator();

	
	

	void train();

	std::vector<ClassifierEvaluation> evaluateDataSet(int nrOfEvaluations, bool includeRawResponses);
	double evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const;

	EvaluationSlidingWindowResult ModelEvaluator::evaluateWithSlidingWindow(int nrOfEvaluations, int trainingRound, float valueShiftForFalsePosOrNegCollection, int maxNrOfFalsePosOrNeg);

	void saveModel(std::string& path);
	void loadModel(std::string& path);
};

