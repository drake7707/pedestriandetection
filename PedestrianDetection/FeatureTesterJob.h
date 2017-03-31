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
#include "EvaluationSettings.h"

class FeatureTester;

class FeatureTesterJob {



	std::set<std::string> set;
	std::string baseDataSetPath;
	
	EvaluationSettings settings;

	FeatureTester* tester;

private:
	void FeatureTesterJob::evaluateTrainingWithSlidingWindow(EvaluatorCascade& cascade, std::string& featureSetName, TrainingDataSet& trainingDataSet, std::unique_ptr<FeatureSet>& fset) const ;

public:
	FeatureTesterJob(FeatureTester* tester, std::set<std::string> set, std::string& baseDataPath, EvaluationSettings settings);

	std::string FeatureTesterJob::getFeatureName() const;
	void run() const;


	void generateFeatureImportanceImage(EvaluatorCascade& cascade, std::unique_ptr<FeatureSet>& fset) const;

};