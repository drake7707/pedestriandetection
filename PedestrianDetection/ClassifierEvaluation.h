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

	int nrOfImagesEvaluated = 0;

	double valueShift = 0;
	double evaluationSpeedPerRegionMS = 0;

	std::vector<int> falsePositivesPerImage;

	std::vector<RawEvaluationEntry> rawValues;

	ClassifierEvaluation(int nrOfImages = 0) {
		falsePositivesPerImage = std::vector<int>(nrOfImages, 0);
	}

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
		out << "FPPI " << getFPPI() << std::endl;
		out << "Miss rate " << getMissRate() << std::endl;
		out << "-------------------" << std::endl;
		out << "Evaluation time " << evaluationSpeedPerRegionMS << "ms" << std::endl;

	}

	double getTPR() const {
		return 1.0 * nrOfTruePositives / (nrOfTruePositives + nrOfFalseNegatives);
	}
	double getFPR() const {
		return 1.0 *nrOfFalsePositives / (nrOfFalsePositives + nrOfTrueNegatives);
	}
	double getPrecision() const {
		return 1.0 *nrOfTruePositives / (nrOfTruePositives + nrOfFalsePositives);
	}
	double getRecall()const {
		return getTPR();
	}
	double getFScore(double beta) const {
		double betabeta = beta * beta;
		double precision = getPrecision();
		double recall = getRecall();
		return (1 + betabeta) * (precision * recall) / ((betabeta * precision) + recall);
	}

	double getMissRate() {
		return 1 - getTPR();
	}

	double getFPPI() {
		if (falsePositivesPerImage.size() == 0)
			return -1;

		int sum = 0;
		for (int i : falsePositivesPerImage)
			sum += i;
		return 1.0 * sum / nrOfImagesEvaluated;
	}

	std::vector<int> getWorstPerformingImages() {
		std::map<int, std::vector<int>> map;
		for (int i = 0; i < falsePositivesPerImage.size(); i++)
		{

			map[falsePositivesPerImage[i]].push_back(i);
		}

		std::vector<int> worstPerforming;
		auto it = map.rbegin();
		while (it != map.rend()) {
			auto& v = it->second;
			for (int imgNr : v)
				worstPerforming.push_back(imgNr);
			it++;
		}
		return worstPerforming;
	}

	void toCSVLine(std::ostream& out, bool header) {
		if (header)
			out << "TP;TN;FP;FN;TPR;FPR;Precision;Recall;F1;FPPI;MissRate;EvaluationTimePerWindowMS;ValueShift;WorstFalsePositiveImages";
		else
		{
			std::vector<int> worstPerforming = getWorstPerformingImages();
			std::string str = "";
			for (int i = 0; i < worstPerforming.size(); i++) {
				str += std::to_string(worstPerforming[i]);

				if (i != worstPerforming.size() - 1)
					str += ",";
			}
			out << std::fixed << nrOfTruePositives << ";" << nrOfTrueNegatives << ";" << nrOfFalsePositives << ";" << nrOfFalseNegatives
				<< ";" << getTPR() << ";" << getFPR() << ";" << getPrecision() << ";" << getRecall() << ";" << getFScore(1)
				<< ";" << getFPPI() << ";" << getMissRate()
				<< ";" << evaluationSpeedPerRegionMS << ";" << valueShift << ";" << str;
		}
	}
};