#include "FeatureTester.h"
#include "ModelEvaluator.h"
#include "VariableNumberFeatureCreator.h"
#include <fstream>



FeatureTesterJob::FeatureTesterJob(std::string& featureSetName, FeatureSet& set, TrainingDataSet& trainingDataSet, int nrOfEvaluations)
	: featureSetName(featureSetName), featureSet(set), trainingDataSet(trainingDataSet), nrOfEvaluations(nrOfEvaluations) {

}

std::vector<ClassifierEvaluation> FeatureTesterJob::run() const {

	std::vector<ClassifierEvaluation> evaluations;

	ModelEvaluator evaluator(trainingDataSet, featureSet);

	std::cout << "Started training of " << this->featureSetName << std::endl;
	long elapsedTrainingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
		evaluator.train();
	});
	std::cout << "Training complete after " << elapsedTrainingTime << "ms for " << this->featureSetName << std::endl;

	// save it to a file
	evaluator.saveModel("models\\" + this->featureSetName + ".xml");

	std::cout << "Started evaluation of " << this->featureSetName << std::endl;
	long elapsedEvaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
		evaluations = evaluator.evaluateDataSet(nrOfEvaluations, false);
	});
	std::cout << "Evaluation complete after " << elapsedEvaluationTime << "ms for " << this->featureSetName << std::endl;
	return evaluations;
}


//-------------------------------------------------



FeatureTester::FeatureTester(TrainingDataSet& trainingDataSet) : trainingDataSet(trainingDataSet)
{
	loadProcessedFeatureSets();
}


FeatureTester::~FeatureTester()
{
	// delete creators
	for (auto& pair : creators)
		delete pair.second;
}


void FeatureTester::prepareCreators() {
	// all variable number feature creators need to be prepared with the k-means centroids
	for (auto& pair : creators) {
		if (dynamic_cast<VariableNumberFeatureCreator*>(pair.second) != nullptr) {
			std::cout << "Preparing feature creator " << pair.second->getName() << std::endl;
			dynamic_cast<VariableNumberFeatureCreator*>(pair.second)->prepare(trainingDataSet);
		}
	}
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

void FeatureTester::markFeatureSetProcessed(std::string& featureSetName) {
	processedFeatureSets.emplace(featureSetName);
	std::ofstream str("processedsets.txt", std::ofstream::out | std::ofstream::app);
	if (!str.is_open())
		return;
	str << featureSetName << std::endl;
	str.flush();
}

void FeatureTester::addAvailableCreator(IFeatureCreator* creator) {
	creators.emplace(creator->getName(), creator);
}

std::vector<IFeatureCreator*> FeatureTester::getAvailableCreators() const {
	std::vector<IFeatureCreator*> items;
	for (auto& pair : creators)
		items.push_back(pair.second);
	return items;
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

	FeatureTesterJob job(featureSetName, featureSet, trainingDataSet, nrOfEvaluations);
	jobs.push(job);
}

void FeatureTester::runJobs(std::string& resultsFile) {

	std::ofstream resultStream(resultsFile, std::ofstream::out | std::ofstream::app);
	if (!resultStream.is_open())
		throw std::exception("Unable to open or create results file");

	resultStream << "Name;";
	ClassifierEvaluation().toCSVLine(resultStream, true);
	resultStream << ";";
	resultStream << std::endl;

	// make sure all creates are ready to roll
	prepareCreators();

	std::mutex resultsFileMutex;

	// run 4 jobs concurrently
	Semaphore semaphore(nrOfConcurrentJobs);

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

					std::string name = job.featureSetName;
					ft->markFeatureSetProcessed(name);

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
}
