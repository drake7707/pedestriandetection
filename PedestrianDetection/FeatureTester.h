#pragma once
#include <unordered_map>
#include "IFeatureCreator.h"
#include "FeatureSet.h"
#include <set>
#include <queue>
#include "Helper.h"
#include "ClassifierEvaluation.h"

class FeatureTester;

class FeatureTesterJob {

	
	std::string baseDatasetPath;
	FeatureSet set;

public:
	std::string featureSetName;

	FeatureTesterJob(std::string& featureSetName, FeatureSet& set, std::string& baseDatasetPath);
	std::vector<ClassifierEvaluation> run() const;

};


class FeatureTester
{
private:
	std::unordered_map<std::string, IFeatureCreator*> creators;
	std::string baseDatasetPath;
	std::queue<FeatureTesterJob> jobs;

public:
	FeatureTester(std::string& baseDatasetPath);
	~FeatureTester();


	void addAvailableCreator(std::string& name, IFeatureCreator* creator);

	void addJob(std::set<std::string>& set);

	void runJobs();
};

