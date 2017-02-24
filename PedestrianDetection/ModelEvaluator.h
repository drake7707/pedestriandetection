#pragma once
#include <string>
#include <functional>

#include "FeatureSet.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "ClassifierEvaluation.h"


struct Model {
	std::vector<float> meanVector;
	std::vector<float> sigmaVector;
	cv::Ptr<cv::ml::Boost> boost;
};




class ModelEvaluator
{
private:
	const std::string baseDatasetPath;
	const FeatureSet& set;

	Model model;

	void iterateDataSet(std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat& rgb, cv::Mat& depth)> func) const;

public:
	ModelEvaluator(const std::string& baseDatasetPath, const FeatureSet& set);
	~ModelEvaluator();


	double trainingTimeMS = 0;

	void train();

	std::vector<ClassifierEvaluation> evaluate(int nrOfEvaluations);
	int ModelEvaluator::evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift = 0) const;

	void saveModel(std::string& path);
	void loadModel(std::string& path);
};

