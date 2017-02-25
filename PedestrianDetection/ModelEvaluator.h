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

	int evaluateFeatures(FeatureVector& v, double valueShift = 0) const;


public:
	ModelEvaluator(const std::string& baseDatasetPath, const FeatureSet& set);
	~ModelEvaluator();


	double trainingTimeMS = 0;

	void train();

	std::vector<ClassifierEvaluation> evaluate(int nrOfEvaluations);
	int evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift = 0) const;
	

	void saveModel(std::string& path);
	void loadModel(std::string& path);
};

