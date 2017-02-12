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

	int nrOfTN = 2;
	int testSampleEvery = 5;
	bool addS2 = true;

	int patchSize = 8;
	int binSize = 9;

	cv::Mat getFeatures(cv::Mat& mat);

	void Detector::iterateDataset(std::function<void(cv::Mat&)> tpFunc, std::function<void(cv::Mat&)> tnFunc, std::function<bool(int)> includeSample);

public:
	void toString(std::ostream& o) {
		std::string str = "";

		o << "Reference size: " << refWidth << "x" << refHeight << std::endl;
		o << "Patch size: " << patchSize << std::endl;
		o << "Bin size: " << binSize << std::endl;
		o << "Number of TN per TP/TPFlipped : " << nrOfTN << std::endl;
		o << "Training/test set split : " << (100 - (100 / testSampleEvery)) << "/" << (100 / testSampleEvery) << std::endl;
		o << "Add S^2 of histograms to features: " << (addS2 ? "yes" : "no") << std::endl;
	}

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

