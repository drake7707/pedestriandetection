#pragma once
#include <unordered_map>
#include "IFeatureCreator.h"
#include "FeatureSet.h"
#include <set>
#include <queue>
#include "Helper.h"
#include "ClassifierEvaluation.h"
#include "TrainingDataSet.h"
#include <functional>
#include <memory>
#include "FeatureTester.h"
#include <mutex>
#include "EvaluatorCascade.h"
#include "JetHeatMap.h"

class FeatureTester;

class FeatureTesterJob {



	std::set<std::string> set;
	std::vector<cv::Size>& windowSizes;
	std::string baseDataSetPath;
	int nrOfEvaluations;
	int nrOfTrainingRounds;
	bool evaluateOnSlidingWindow;

	FeatureTester* tester;

public:
	FeatureTesterJob(FeatureTester* tester, std::vector<cv::Size>& windowSizes, std::set<std::string>& set, std::string& baseDataPath, int nrOfEvaluations, int nrOfTrainingRounds, bool evaluateOnSlidingWindow);

	std::string FeatureTesterJob::getFeatureName() const;
	void run() const;


	void generateFeatureImportanceImage(EvaluatorCascade& cascade, std::unique_ptr<FeatureSet>& fset) const;

};