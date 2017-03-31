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
#include "FeatureTesterJob.h"
#include "EvaluationSettings.h"

class FeatureTesterJob;

struct FactoryCreator {
	typedef  std::function<std::unique_ptr<IFeatureCreator>(std::string& name)> FactoryMethod;
	std::string name;
	FactoryMethod createInstance;
	FactoryCreator(std::string& name, FactoryMethod createInstance) : name(name), createInstance(createInstance) { }
};


class FeatureTester
{
private:
	std::unordered_map<std::string, FactoryCreator> creators;

	std::queue<FeatureTesterJob*> jobs;

	std::set<std::string> processedFeatureSets;
	void loadProcessedFeatureSets();
	void markFeatureSetProcessed(std::string& featureSetName);



	std::mutex singletonLock;




public:
	FeatureTester();
	~FeatureTester();

	int nrOfConcurrentJobs = 4;

	void addFeatureCreatorFactory(FactoryCreator& creator);
	FactoryCreator getAvailableCreator(std::string& name) const;

	std::vector<std::string> getFeatureCreatorFactories() const;

	void FeatureTester::addJob(std::set<std::string>& set, DataSet* dataSet, EvaluationSettings& settings);

	void runJobs();

	std::unique_ptr<FeatureSet> getFeatureSet(const std::set<std::string>& set);

	std::mutex* getLock() {
		return &singletonLock;
	}
};

