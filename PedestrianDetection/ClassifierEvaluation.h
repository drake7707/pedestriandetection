#pragma once
#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"

struct ClassifierEvaluation {
	int nrOfTruePositives = 0;
	int nrOfTrueNegatives = 0;
	int nrOfFalsePositives = 0;
	int nrOfFalseNegatives = 0;

	int nrOfImagesEvaluated = 0;

	double valueShift = 0;
	double evaluationSpeedPerRegionMS = 0;

	std::vector<int> falsePositivesPerImage;

	ClassifierEvaluation(int nrOfImages = 0);

	void print(std::ostream& out);

	double getTPR() const;

	double getFPR() const;

	double getPrecision() const;

	double getRecall()const;

	double getFScore(double beta) const;

	double getMissRate() const;

	double getFPPI() const;

	std::vector<int> getWorstPerformingImages() const; 

	void toCSVLine(std::ostream& out, bool header) const;
};