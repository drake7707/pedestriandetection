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
	std::string baseDataSetPath;
	int nrOfEvaluations;
	int nrOfTrainingRounds;
	bool evaluateOnSlidingWindow;

public:
	FeatureTesterJob(std::unordered_map<std::string, FactoryCreator>& creators, std::set<std::string>& set, std::string& baseDataPath, int nrOfEvaluations, int nrOfTrainingRounds, bool evaluateOnSlidingWindow);
	std::string FeatureTesterJob::getFeatureName() const;
	void run() const;

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

	void addFeatureCreatorFactory(FactoryCreator& creator);
	FactoryCreator getAvailableCreator(std::string& name) const;

	std::vector<std::string> getFeatureCreatorFactories() const;

	void addJob(std::set<std::string>& set, std::string& baseDataSetPath, int nrOfEvaluations, int nrOfTrainingRounds = 1, bool evaluateOnSlidingWindow = true);

	void runJobs();

	FeatureSet getFeatureSet(std::set<std::string>& set);
};

