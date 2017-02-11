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

class Detector
{

private:

	
	//std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";
	std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
	std::string weakClassifierSVMFile = "kittitraining.xml";

	int refWidth = 64;
	int refHeight = 128;

	int patchSize = 8;
	int binSize = 9;

	void iterateDataset(std::function<void(cv::Mat&, HoGResult&)> tpFunc, std::function<void(cv::Mat&, HoGResult&)> tnFunc, std::function<bool(int)> includeSample);

public:
	cv::Ptr<cv::ml::SVM> buildWeakHoGSVMClassifier();

	void saveSVMLightFiles();

	ClassifierEvaluation evaluateWeakHoGSVMClassifier(bool onTrainingSet);



	Detector()
	{
	}


	~Detector()
	{
	}


};

