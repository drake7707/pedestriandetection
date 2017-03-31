#include "EvaluationSettings.h"


void EvaluationSettings::read(std::string& iniPath) {

	INIReader reader(iniPath);


	if (reader.ParseError() < 0) {
		return;
	}


	refWidth = reader.GetInteger("general", "refWidth", refWidth);
	refHeight = reader.GetInteger("general", "refHeight", refHeight);

	nrOfEvaluations = reader.GetInteger("general", "nrOfEvaluations", nrOfEvaluations);
	maxNrWorstPosNeg = reader.GetInteger("general", "maxNrWorstPosNeg", maxNrWorstPosNeg);
	requiredTPRRate = reader.GetReal("general", "requiredTPRRate", requiredTPRRate);

	maxWeakClassifiers = reader.GetInteger("general", "maxWeakClassifiers", maxWeakClassifiers);
	nrOfTrainingRounds = reader.GetInteger("general", "nrOfTrainingRounds", nrOfTrainingRounds);


	int mod = reader.GetInteger("general", "sampleSetEvery", 2);

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