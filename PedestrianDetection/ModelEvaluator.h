#pragma once
#include <string>
#include <functional>

#include "FeaturesSet.h"
#include "opencv2/opencv.hpp"
#include "Helper.h"


struct Model {
	std::vector<float> meanVector;
	std::vector<float> sigmaVector;
	cv::Ptr<cv::ml::Boost> boost;
};


struct ClassifierEvaluation {

	int nrOfTruePositives = 0;
	int nrOfTrueNegatives = 0;
	int nrOfFalsePositives = 0;
	int nrOfFalseNegatives = 0;

	void print(std::ostream& out) {

		int totalNegatives = nrOfTrueNegatives + nrOfFalseNegatives;
		int totalPositives = nrOfTruePositives + nrOfFalsePositives;

		out << std::setprecision(2);
		out << "Evaluation result " << std::endl;
		out << "# True Positives " << nrOfTruePositives << " (" << floor(nrOfTruePositives * 1.0 / totalPositives * 100) << "%)" << std::endl;
		out << "# True Negatives " << nrOfTrueNegatives << " (" << floor(nrOfTrueNegatives * 1.0 / totalNegatives * 100) << "%)" << std::endl;
		out << "# False Positives " << nrOfFalsePositives << " (" << floor(nrOfFalsePositives * 1.0 / totalPositives * 100) << "%)" << std::endl;
		out << "# False Negatives " << nrOfFalseNegatives << " (" << floor(nrOfFalseNegatives * 1.0 / totalNegatives * 100) << "%)" << std::endl;
	}
};


class ModelEvaluator
{
private:
	std::string baseDatasetPath;
	FeaturesSet& set;

	Model model;

	void iterateDataSet(std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat& rgb, cv::Mat& depth)> func) const;

public:
	ModelEvaluator(std::string& baseDatasetPath, FeaturesSet& set);
	~ModelEvaluator();


	void train();

	ClassifierEvaluation evaluate() const ;
};

