#pragma once
#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"

struct RawEvaluationEntry {
	int imageNumber;
	cv::Rect region;
	double response;
	bool correct;
	RawEvaluationEntry(int imageNumber, cv::Rect& region, double response, bool correct) : imageNumber(imageNumber), region(region), response(response), correct(correct) { }

	bool operator<(RawEvaluationEntry& other) const {
		return this->response < other.response;
	}
};

struct ClassifierEvaluation {
	int nrOfTruePositives = 0;
	int nrOfTrueNegatives = 0;
	int nrOfFalsePositives = 0;
	int nrOfFalseNegatives = 0;

	double valueShift = 0;
	double evaluationSpeedPerRegionMS = 0;

	std::vector<RawEvaluationEntry> rawValues;

	void print(std::ostream& out) {

		int totalNegatives = nrOfTrueNegatives + nrOfFalseNegatives;
		int totalPositives = nrOfTruePositives + nrOfFalsePositives;

		out << std::fixed << std::setprecision(2);
		out << "Evaluation result " << std::endl;
		out << "-------------------" << std::endl;
		out << "# True Positives " << nrOfTruePositives << std::endl;
		out << "# True Negatives " << nrOfTrueNegatives << std::endl;
		out << "# False Positives " << nrOfFalsePositives << std::endl;
		out << "# False Negatives " << nrOfFalseNegatives << std::endl;
		out << "-------------------" << std::endl;
		out << "TPR " << getTPR() * 100 << "%" << std::endl;
		out << "FPR " << getFPR() * 100 << "%" << std::endl;
		out << "-------------------" << std::endl;
		out << std::setprecision(6) << std::endl;
		out << "Precision " << getPrecision() << std::endl;
		out << "Recall " << getRecall() << std::endl;
		out << "F1-score " << getFScore(1) << std::endl;
		out << "-------------------" << std::endl;
		out << "Evaluation time " << evaluationSpeedPerRegionMS << "ms" << std::endl;

	}

	double getTPR() {
		return 1.0 * nrOfTruePositives / (nrOfTruePositives + nrOfFalseNegatives);
	}
	double getFPR() {
		return 1.0 *nrOfFalsePositives / (nrOfFalsePositives + nrOfTrueNegatives);
	}
	double getPrecision() {
		return 1.0 *nrOfTruePositives / (nrOfTruePositives + nrOfFalsePositives);
	}
	double getRecall() {
		return getTPR();
	}
	double getFScore(double beta) {
		double betabeta = beta * beta;
		double precision = getPrecision();
		double recall = getRecall();
		return (1 + betabeta) * (precision * recall) / ((betabeta * precision) + recall);
	}

	void toCSVLine(std::ostream& out, bool header) {
		if (header)
			out << "TP;TN;FP;FN;TPR;FPR;Precision;Recall;F1;EvaluationTimePerWindowMS;ValueShift";
		else
			out << std::fixed << nrOfTruePositives << ";" << nrOfTrueNegatives << ";" << nrOfFalsePositives << ";" << nrOfFalseNegatives
			<< ";" << getTPR() << ";" << getFPR() << ";" << getPrecision() << ";" << getRecall() << ";" << getFScore(1)
			<< ";" << evaluationSpeedPerRegionMS << ";" << valueShift;
	}
};