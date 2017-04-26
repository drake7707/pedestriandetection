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

	/// <summary>
	/// Calculates the true positive rate
	/// </summary>
	double getTPR() const;

	/// <summary>
	/// Calculates the false positive rate
	/// </summary>
	double getFPR() const;

	/// <summary>
	/// Calculates the precision
	/// </summary>
	double getPrecision() const;

	/// <summary>
	/// Calculates the recall
	/// </summary>
	double getRecall()const;

	/// <summary>
	/// Calculates the F-score with given weight
	/// </summary>
	double getFScore(double beta) const;

	/// <summary>
	/// Calculates the miss rate
	/// </summary>
	double getMissRate() const;

	/// <summary>
	/// Calculates the number of false positives per image
	/// </summary>
	double getFPPI() const;

	/// <summary>
	/// Returns a list of the worst performing images if available
	/// </summary>
	std::vector<int> getWorstPerformingImages() const; 

	/// <summary>
	/// Writes the data to the given stream in a csv format, separated with ';'
	/// </summary>
	void toCSVLine(std::ostream& out, bool header) const;
};