#pragma once
#include "INIReader.h"
#include <string>
#include "Helper.h"
#include <functional>
#include "opencv2/opencv.hpp"

struct EvaluationSettings
{
public:
	int refWidth = 64;
	int refHeight = 128;

	int nrOfEvaluations = 300;

	int maxNrOfFalsePosOrNeg = 4000;
	int maxNrOfFPPerImage = 10;

	float requiredTPRRate = 0.95;
	int maxWeakClassifiers = 500;
	int nrOfTrainingRounds = 4;

	int evaluationRange = 60;
	int slidingWindowParallelization = 8;
	int baseWindowStride = 16;

	bool addFlippedInTrainingSet = true;

	float vehicleSpeedKMh = 50;
	float tireRoadFriction = 0.7;

	std::string kittiDataSetPath = "";
	std::string kaistDataSetPath = "";

	//int slidingWindowEveryXImage = 1;


	std::vector<cv::Size> windowSizes = {
		/*cv::Size(24,48),*/
		cv::Size(32,64),
		cv::Size(48,96),
		cv::Size(64,128),
		cv::Size(80,160),
		cv::Size(96,192),
		cv::Size(104,208),
		cv::Size(112,224),
		cv::Size(120,240),
		cv::Size(128,256)
	};

	std::function<bool(int)> trainingCriteria = [](int imageNumber) -> bool { return imageNumber % 2 == 0; };
	std::function<bool(int)> testCriteria = [](int imageNumber) -> bool { return imageNumber % 2 == 1; };

	/// <summary>
	/// Reads the settings.ini file from the given path
	/// </summary>
	void read(std::string& iniPath);
};

