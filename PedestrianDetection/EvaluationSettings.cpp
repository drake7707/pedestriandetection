#include "EvaluationSettings.h"


void EvaluationSettings::read(std::string& iniPath) {

	INIReader reader(iniPath);


	if (reader.ParseError() < 0) {
		return;
	}


	refWidth = reader.GetInteger("general", "refWidth", refWidth);
	refHeight = reader.GetInteger("general", "refHeight", refHeight);

	nrOfEvaluations = reader.GetInteger("general", "nrOfEvaluations", nrOfEvaluations);
	maxNrOfFalsePosOrNeg = reader.GetInteger("general", "maxNrOfFalsePosOrNeg", maxNrOfFalsePosOrNeg);
	maxNrOfFPPerImage = reader.GetInteger("general", "maxNrOfFPPerImage", maxNrOfFPPerImage);
	
	requiredTPRRate = reader.GetReal("general", "requiredTPRRate", requiredTPRRate);

	maxWeakClassifiers = reader.GetInteger("general", "maxWeakClassifiers", maxWeakClassifiers);
	nrOfTrainingRounds = reader.GetInteger("general", "nrOfTrainingRounds", nrOfTrainingRounds);


	int mod = reader.GetInteger("general", "sampleSetEvery", 2);


	evaluationRange = reader.GetInteger("general", "evaluationRange", evaluationRange);
	slidingWindowParallelization = reader.GetInteger("general", "slidingWindowParallelization", slidingWindowParallelization);
	baseWindowStride = reader.GetInteger("general", "baseWindowStride", baseWindowStride);

	kittiDataSetPath = reader.Get("general", "kittiDataSetPath", "");
	kaistDataSetPath = reader.Get("general", "kaistDataSetPath", "");


	addFlippedInTrainingSet = reader.GetBoolean("general", "addFlippedInTrainingSet", true);

	trainingCriteria = [=](int imageNumber) -> bool { return imageNumber % mod == 0; };
	testCriteria = [=](int imageNumber) -> bool { return imageNumber % mod == 1; };


	std::string windowSizesStr = reader.Get("general", "windowSizes", "");
	if (windowSizesStr != "") {
		windowSizes.clear();
		auto parts = splitString(windowSizesStr, ',');
		for (std::string sizeStr : parts) {
			auto sizeParts = splitString(sizeStr, 'x');
			cv::Size s(atoi(sizeParts[0].c_str()), atoi(sizeParts[1].c_str()));
			windowSizes.push_back(s);
		}
	}
}