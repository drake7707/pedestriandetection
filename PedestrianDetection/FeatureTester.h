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

class FeatureTester;

struct FactoryCreator {
	typedef  std::function<std::unique_ptr<IFeatureCreator>(std::string& name)> FactoryMethod;
	std::string name;
	FactoryMethod createInstance;
	FactoryCreator(std::string& name, FactoryMethod createInstance) : name(name), createInstance(createInstance) { }
};


class FeatureTesterJob {

	
	std::unordered_map<std::string, FactoryCreator> creators;
	std::set<std::string> set;
	int nrOfEvaluations;
	int nrOfTrainingRounds;

public:
	std::string featureSetName;

	FeatureTesterJob(std::unordered_map<std::string, FactoryCreator>& creators, std::set<std::string>& set, int nrOfEvaluations, int nrOfTrainingRounds);
	void run();

};


class FeatureTester
{
private:
	std::unordered_map<std::string, FactoryCreator> creators;
	
	std::queue<FeatureTesterJob> jobs;


	std::set<std::string> processedFeatureSets;
	void loadProcessedFeatureSets();
	void markFeatureSetProcessed(std::string& featureSetName);

public:
	FeatureTester();
	~FeatureTester();

	int nrOfConcurrentJobs = 4;

	void addAvailableCreator(FactoryCreator& creator);
	FactoryCreator getAvailableCreator(std::string& name) const;

	void addJob(std::set<std::string>& set, int nrOfEvaluations, int nrOfTrainingRounds = 1);

	void runJobs();
};

