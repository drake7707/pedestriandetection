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


class ModelEvaluator
{
private:
	const TrainingDataSet& trainingDataSet;
	const FeatureSet& set;

	Model model;

	EvaluationResult evaluateFeatures(FeatureVector& v, double valueShift = 0) const;


public:
	ModelEvaluator(const TrainingDataSet& trainingDataSet, const FeatureSet& set);
	~ModelEvaluator();


	double trainingTimeMS = 0;

	void train();

	std::vector<ClassifierEvaluation> evaluateDataSet(int nrOfEvaluations, bool includeRawResponses);
	EvaluationResult evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift = 0) const;


	void saveModel(std::string& path);
	void loadModel(std::string& path);
};

