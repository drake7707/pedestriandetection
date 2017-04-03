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
#include "Semaphore.h"

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
	std::mutex singletonLock;

	std::set<std::string> processedFeatureSets;

	/// <summary>
	/// Loads the already processed  feature sets from the processedsets.txt file
	/// </summary>
	void loadProcessedFeatureSets();

	/// <summary>
	/// Marks a feature set as processed
	/// </summary>
	void FeatureTester::markFeatureSetProcessed(DataSet* dataSet, std::string& featureSetName);


public:
	FeatureTester();
	~FeatureTester();

	int nrOfConcurrentJobs = 4;

	/// <summary>
	/// Add a feature descriptor factory to use in building feature sets
	/// </summary>
	void addFeatureCreatorFactory(FactoryCreator& creator);

	/// <summary>
	/// Returns the the given factory creator with given name
	/// </summary>
	FactoryCreator getAvailableCreator(std::string& name) const;

	/// <summary>
	/// Returns all the available feature creator factories
	/// </summary>
	std::vector<std::string> getFeatureCreatorFactories() const;

	/// <summary>
	/// Queues a job to run with the given feature set and settings on given the data set
	/// </summary>
	void FeatureTester::addJob(std::set<std::string>& set, DataSet* dataSet, EvaluationSettings& settings);

	/// <summary>
	/// Run all the queued jobs
	/// </summary>
	void runJobs();

	/// <summary>
	/// Returns the feature set of the given feature names
	/// </summary>
	std::unique_ptr<FeatureSet> getFeatureSet(const std::set<std::string>& set);

	/// <summary>
	/// Returns a lock to synchronize between multiple jobs
	/// </summary>
	std::mutex* getLock();
};

