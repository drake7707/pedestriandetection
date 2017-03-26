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

	
	
	double evaluateFeatures(FeatureVector& v) const;



	void train(const TrainingDataSet& trainingDataSet, const FeatureSet& set, int maxWeakClassifiers, std::function<bool(int number)> canSelectFunc);

	// std::vector<ClassifierEvaluation> evaluateDataSet(int nrOfEvaluations, bool includeRawResponses) const;
	// double evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const;

   //EvaluationSlidingWindowResult evaluateWithSlidingWindow(int nrOfEvaluations, int trainingRound, float tprToObtainWorstFalsePositives, int maxNrOfFalsePosOrNeg) const;

	void saveModel(std::string& path);
	void loadModel(std::string& path);


	std::string getName() const;

};

