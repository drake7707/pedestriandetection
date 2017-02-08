#pragma once
#include "opencv2\opencv.hpp"
#include <functional>
#include "HistogramOfOrientedGradients.h"


struct ClassifierEvaluation {

	int nrOfTruePositives = 0;
	int nrOfTrueNegatives = 0;
	int nrOfFalsePositives = 0;
	int nrOfFalseNegatives = 0;

	void print(std::ostream& out) {
		out << "Evaluation result " << std::endl;
		out << "# True Positives " << nrOfTruePositives << std::endl;
		out << "# True Negatives " << nrOfTrueNegatives << std::endl;
		out << "# False Positives " << nrOfFalsePositives << std::endl;
		out << "# False Negatives " << nrOfFalseNegatives << std::endl;
	}
};

class Detector
{

private:

	
	std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";
	//std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
	std::string weakClassifierSVMFile = "kittitraining.xml";

	int refWidth = 64;
	int refHeight = 128;

	int patchSize = 8;
	int binSize = 9;

	void iterateDataset(std::function<void(cv::Mat&, HoGResult&)> tpFunc, std::function<void(cv::Mat&, HoGResult&)> tnFunc);

public:
	cv::Ptr<cv::ml::SVM> buildWeakHoGSVMClassifier();


	ClassifierEvaluation evaluateWeakHoGSVMClassifier();



	Detector()
	{
	}


	~Detector()
	{
	}


};

