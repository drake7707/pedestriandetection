#pragma once
#include <unordered_map>
#include "IFeatureCreator.h"
#include "FeatureSet.h"
#include <set>
#include <queue>
#include "Helper.h"
#include "ClassifierEvaluation.h"
#include "TrainingDataSet.h"

class FeatureTester;

class FeatureTesterJob {

	
	TrainingDataSet trainingDataSet;

	FeatureSet featureSet;
	int nrOfEvaluations;

public:
	std::string featureSetName;

	FeatureTesterJob(std::string& featureSetName, FeatureSet& set, TrainingDataSet& trainingDataSet, int nrOfEvaluations);
	std::vector<ClassifierEvaluation> run() const;

};


class FeatureTester
{
private:
	std::unordered_map<std::string, IFeatureCreator*> creators;
	TrainingDataSet trainingDataSet;

	std::queue<FeatureTesterJob> jobs;


	std::set<std::string> processedFeatureSets;
	void loadProcessedFeatureSets();
	void markFeatureSetProcessed(std::string& featureSetName);

	void prepareCreators();

public:
	FeatureTester(TrainingDataSet& trainingDataSet);
	~FeatureTester();

	int nrOfConcurrentJobs = 4;

	void addAvailableCreator(IFeatureCreator* creator);
	std::vector<IFeatureCreator*> getAvailableCreators() const;

	void addJob(std::set<std::string>& set, int nrOfEvaluations);

	void runJobs(std::string& resultsFile = std::string("results.csv"));
};

