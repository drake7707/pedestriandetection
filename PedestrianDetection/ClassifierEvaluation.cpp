#include "ClassifierEvaluation.h"


ClassifierEvaluation::ClassifierEvaluation(int nrOfImages) {
	falsePositivesPerImage = std::vector<int>(nrOfImages, 0);
}

void ClassifierEvaluation::print(std::ostream& out) {

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

double ClassifierEvaluation::getTPR() const {
	return 1.0 * nrOfTruePositives / (nrOfTruePositives + nrOfFalseNegatives);
}
double ClassifierEvaluation::getFPR() const {
	return 1.0 *nrOfFalsePositives / (nrOfFalsePositives + nrOfTrueNegatives);
}
double ClassifierEvaluation::getPrecision() const {
	return 1.0 *nrOfTruePositives / (nrOfTruePositives + nrOfFalsePositives);
}
double ClassifierEvaluation::getRecall() const {
	return getTPR();
}
double ClassifierEvaluation::getFScore(double beta) const {
	double betabeta = beta * beta;
	double precision = getPrecision();
	double recall = getRecall();
	return (1 + betabeta) * (precision * recall) / ((betabeta * precision) + recall);
}

double ClassifierEvaluation::getMissRate() const {
	return 1 - getTPR();
}

double ClassifierEvaluation::getFPPI() const {
	if (falsePositivesPerImage.size() == 0)
		return -1;

	int sum = 0;
	for (int i : falsePositivesPerImage)
		sum += i;
	return 1.0 * sum / nrOfImagesEvaluated;
}

std::vector<int> ClassifierEvaluation::getWorstPerformingImages() const {
	std::map<int, std::vector<int>> map;
	for (int i = 0; i < falsePositivesPerImage.size(); i++)
	{

		map[falsePositivesPerImage[i]].push_back(i);
	}

	std::vector<int> worstPerforming;
	auto it = map.rbegin();

	int i = 0;
	while (it != map.rend() && i < 50) {
		auto& v = it->second;
		for (int imgNr : v) {
			worstPerforming.push_back(imgNr);
			i++;
			if (i >= 50)
				break;
		}
		it++;
	}
	return worstPerforming;
}

void ClassifierEvaluation::toCSVLine(std::ostream& out, bool header) const {
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