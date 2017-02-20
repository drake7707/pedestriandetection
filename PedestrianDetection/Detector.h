#pragma once
#include "opencv2\opencv.hpp"
#include <functional>
#include "HistogramOfOrientedGradients.h"
#include "FeatureVector.h"


class Detector
{

private:



public:
	double biasShift = 0.003;

private:
	bool modelReady = false;

	struct DetectorModel {
		cv::Ptr<cv::ml::SVM> svm;
		cv::Mat weightVector; // w . x - b = 0
		double bias;
		std::vector<float> meanVector;
		std::vector<float> sigmaVector;

		cv::Ptr<cv::ml::Boost> boost;
	};
	DetectorModel model;


	void buildWeakHoGSVMClassifier(std::vector<FeatureVector>& truePositiveFeatures, std::vector<FeatureVector>& trueNegativeFeatures);

	bool hasModel() const {
		return this->modelReady;
	}

	void loadSVMEvaluationParameters();
public:


	void buildModel(std::vector<FeatureVector>& truePositiveFeatures, std::vector<FeatureVector>& trueNegativeFeatures);


	double evaluate(FeatureVector& vec) const;


	void saveModel(std::string& path);


	void loadModel(std::string& path);


	Detector()
	{
	}


	~Detector()
	{
	}


};

