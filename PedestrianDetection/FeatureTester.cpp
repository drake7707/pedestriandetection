#include "FeatureTester.h"
#include "ModelEvaluator.h"
#include "VariableNumberFeatureCreator.h"
#include <fstream>
#include "ProgressWindow.h"
#include "EvaluatorCascade.h"



//-------------------------------------------------



FeatureTester::FeatureTester()
{
	loadProcessedFeatureSets();
}


FeatureTester::~FeatureTester()
{
}


void FeatureTester::loadProcessedFeatureSets() {
	std::ifstream str("processedsets.txt");
	if (!str.is_open())
		return;

	std::string line;
	while (std::getline(str, line)) {
		processedFeatureSets.emplace(line);
	}
}

void FeatureTester::markFeatureSetProcessed(DataSet* dataSet, std::string& featureSetName) {
	processedFeatureSets.emplace(dataSet->getName() + "_" + featureSetName);
	std::ofstream str("processedsets.txt", std::ofstream::out | std::ofstream::app);
	if (!str.is_open())
		return;
	str << dataSet->getName() + "_" + featureSetName << std::endl;
	str.flush();
}

void FeatureTester::addFeatureCreatorFactory(FactoryCreator& creator) {
	creators.emplace(creator.name, creator);
}

std::vector<std::string> FeatureTester::getFeatureCreatorFactories() const {
	std::vector<std::string> factories;
	for (auto& pair : creators) {
		factories.push_back(pair.first);
	}
	return factories;
}


FactoryCreator FeatureTester::getAvailableCreator(std::string& name) const {
	//@pre name is added in the map
	return creators.find(name)->second;
}

void FeatureTester::addJob(std::set<std::string>& set, DataSet* dataSet, EvaluationSettings& settings) {
	FeatureTester* tester = this;
	FeatureTesterJob* job = new FeatureTesterJob(tester, set, dataSet, settings);
	jobs.push(job);
}

void FeatureTester::runJobs() {

	std::mutex resultsFileMutex;

	// run multiple jobs concurrently 
	Semaphore semaphore(nrOfConcurrentJobs);

	std::vector<FeatureTesterJob*> finishedJobs;
	std::vector<std::thread> threads;
	while (jobs.size() > 0) {
		FeatureTesterJob* job = jobs.front();
		finishedJobs.push_back(job);
		jobs.pop();


		std::string featureSetName = job->getFeatureName();
		if (processedFeatureSets.find(job->getDataSet()->getName() + "_" + featureSetName) != processedFeatureSets.end()) {
			std::cout << "Job " << featureSetName << " is already processed and will be skipped" << std::endl;
		}
		else {
			semaphore.wait();

			FeatureTester* ft = this;
			threads.push_back(std::thread([job, &resultsFileMutex, &semaphore, &ft, featureSetName]() -> void {
				try {

					std::cout << "Starting job " << featureSetName << std::endl;
					job->run();

					// mark a job is processed so it won't be tried again on next run
					resultsFileMutex.lock();
					ft->markFeatureSetProcessed(job->getDataSet(), std::string(featureSetName));
					resultsFileMutex.unlock();
					semaphore.notify();
				}
				catch (std::exception e) {

					std::cout << "Error: " << e.what() << std::endl;
					semaphore.notify();
				}
			}));

			
		}
	}

	// clean up all threads by joining htem
	for (auto& t : threads)
		t.join();

	// clean up jobs from heap
	for (auto j : finishedJobs)
		delete j;
}

std::unique_ptr<FeatureSet> FeatureTester::getFeatureSet(const std::set<std::string>& set) {
	FeatureSet* featureSet = new FeatureSet();
	for (auto& name : set) {
		FactoryCreator creator = creators.find(name)->second;

		std::unique_ptr<IFeatureCreator> featureCreator = creator.createInstance(creator.name);
		featureSet->addCreator(std::move(featureCreator));
	}
	return std::unique_ptr<FeatureSet>(featureSet);
}

std::mutex* FeatureTester::getLock() {
	return &singletonLock;
}