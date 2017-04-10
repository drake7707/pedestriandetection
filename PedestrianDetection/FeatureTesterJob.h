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
#include "RiskAnalysis.h"


class FeatureTester;



class FeatureTesterJob {

	std::set<std::string> set;
	DataSet* dataSet;

	EvaluationSettings settings;

	FeatureTester* tester;

private:

	/// <summary>
	/// Evaluates the cascade with sliding window on the training set
	/// </summary>
	void FeatureTesterJob::evaluateTrainingWithSlidingWindow(EvaluatorCascade& cascade, std::string& featureSetName, TrainingDataSet& trainingDataSet, std::unique_ptr<FeatureSet>& fset) const;

	/// <summary>
	/// Evaluates the cascade with sliding window and NMS on the test set 
	/// </summary>
	void FeatureTesterJob::evaluateTestSet(EvaluatorCascade& cascade, std::string& cascadeFile, std::unique_ptr<FeatureSet>& featureSet, std::string& featureSetName) const;

	/// <summary>
	/// Runs the risk analysis and generates the classification result with the various risk categories if depth information is available
	/// </summary>
	void doRiskAnalysis(FinalEvaluationSlidingWindowResult& finalresult, std::string& featureSetName) const;

	/// <summary>
	/// Generates the feature explanation images of all the models in the cascade and save it to a file 
	/// </summary>
	void generateFeatureImportanceImage(EvaluatorCascade& cascade, std::unique_ptr<FeatureSet>& fset) const;

public:
	FeatureTesterJob(FeatureTester* tester, std::set<std::string> set, DataSet* dataSet, EvaluationSettings settings);

	/// <summary>
	/// Returns the concatenated feature name from the feature set
	/// </summary>
	std::string FeatureTesterJob::getFeatureName() const;

	/// <summary>
	/// Runs the job 
	/// </summary>
	void run() const;

	/// <summary>
	/// Returns the data set that is used for the job
	/// </summary>
	DataSet* FeatureTesterJob::getDataSet() const;

	
};