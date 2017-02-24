#include "FeatureTester.h"
#include "ModelEvaluator.h"
#include <fstream>



FeatureTesterJob::FeatureTesterJob(std::string& featureSetName, FeatureSet& set, std::string& baseDatasetPath, int nrOfEvaluations) : featureSetName(featureSetName), set(set), baseDatasetPath(baseDatasetPath), nrOfEvaluations(nrOfEvaluations) {

}

std::vector<ClassifierEvaluation>  FeatureTesterJob::run() const {

	std::vector<ClassifierEvaluation> evaluations;

	ModelEvaluator evaluator(baseDatasetPath, set);
	evaluator.train();


	return evaluator.evaluate(nrOfEvaluations);
	return evaluations;
}


FeatureTester::FeatureTester(std::string& baseDatasetPath) : baseDatasetPath(baseDatasetPath)
{
	loadProcessedFeatureSets();
}


FeatureTester::~FeatureTester()
{
	// delete creators
	for (auto& pair : creators)
		delete pair.second;
}

void FeatureTester::loadProcessedFeatureSets() {
	std::ifstream str("processedsets.txt");
	if (!str.is_open())
		return;

	std::string line;
	while (std::getline(str, line))	{
		processedFeatureSets.emplace(line);
	}
}

void FeatureTester::markFeatureSetProcessed(std::string& featureSetName) {
	processedFeatureSets.emplace(featureSetName);
	std::ofstream str("processedsets.txt", std::ofstream::out | std::ofstream::app);
	if (!str.is_open())
		return;
	str << featureSetName << std::endl;
}

void FeatureTester::addAvailableCreator(std::string& name, IFeatureCreator* creator) {
	creators.emplace(name, creator);
}

void FeatureTester::addJob(std::set<std::string>& set, int nrOfEvaluations) {
	FeatureSet featureSet;
	std::string featureSetName("");
	for (auto& name : set) {
		if (creators.find(name) == creators.end())
			throw std::exception(("Unable for find feature creator with name " + name).c_str());

		featureSet.addCreator(creators[name]);

		if (name != *(set.begin()))
			featureSetName += "+" + name;
		else
			featureSetName += name;
	}

	FeatureTesterJob job(featureSetName, featureSet, baseDatasetPath, nrOfEvaluations);
	jobs.push(job);
}

void FeatureTester::runJobs() {

	std::ofstream resultStream("results.csv", std::ofstream::out | std::ofstream::app);
	if (!resultStream.is_open())
		throw std::exception("Unable to open or create results file");

	resultStream << "Name;";
	ClassifierEvaluation().toCSVLine(resultStream, true);
	resultStream << ";";
	resultStream << std::endl;


	std::mutex resultsFileMutex;

	// run 4 jobs concurrently
	Semaphore semaphore(4);

	std::vector<std::thread> threads;
	while (jobs.size() > 0) {
		FeatureTesterJob job = jobs.front();
		jobs.pop();

		if (processedFeatureSets.find(job.featureSetName) != processedFeatureSets.end()) {
			std::cout << "Job " << job.featureSetName << " is already processed and will be skipped" << std::endl;
		}
		else {
			semaphore.wait();

			FeatureTester* ft = this;
			threads.push_back(std::thread([job, &resultsFileMutex, &resultStream, &semaphore, &ft]() -> void {
				try {

					std::cout << "Starting job " << job.featureSetName << std::endl;
					std::vector<ClassifierEvaluation> results = job.run();
					
					resultsFileMutex.lock();
					for (auto& eval : results) {
						resultStream << job.featureSetName << ";";
						eval.toCSVLine(resultStream, false);
						resultStream << ";" << std::endl;
					}

					resultsFileMutex.unlock();

					std::string name = job.featureSetName;
					ft->markFeatureSetProcessed(name);

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
}
