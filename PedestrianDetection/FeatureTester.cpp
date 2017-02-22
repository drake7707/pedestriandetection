#include "FeatureTester.h"
#include "ModelEvaluator.h"
#include <fstream>



FeatureTesterJob::FeatureTesterJob(std::string& featureSetName, FeatureSet& set, std::string& baseDatasetPath) : featureSetName(featureSetName), set(set), baseDatasetPath(baseDatasetPath) {

}

std::vector<ClassifierEvaluation>  FeatureTesterJob::run() const {

	std::vector<ClassifierEvaluation> evaluations;
	{
		std::cout << "Testing with 0.8/0.2 prior weights" << std::endl;
		ModelEvaluator evaluator(baseDatasetPath, set, 0.8, 0.2);
		evaluator.train();
		ClassifierEvaluation eval = evaluator.evaluate();
		eval.print(std::cout);
		evaluations.push_back(eval);
	}
	{
		std::cout << "Testing with 0.5/0.5 prior weights" << std::endl;
		ModelEvaluator evaluator(baseDatasetPath, set, 0.5, 0.5);
		evaluator.train();
		ClassifierEvaluation eval = evaluator.evaluate();
		eval.print(std::cout);
		evaluations.push_back(eval);
	}
	{
		std::cout << "Testing with 0.2/0.8 prior weights" << std::endl;
		ModelEvaluator evaluator(baseDatasetPath, set, 0.2, 0.8);
		evaluator.train();
		ClassifierEvaluation eval = evaluator.evaluate();

		eval.print(std::cout);
		evaluations.push_back(eval);
	}
	return evaluations;
}


FeatureTester::FeatureTester(std::string& baseDatasetPath) : baseDatasetPath(baseDatasetPath)
{
}


FeatureTester::~FeatureTester()
{
	// delete creators
	for (auto& pair : creators)
		delete pair.second;
}

void FeatureTester::addAvailableCreator(std::string& name, IFeatureCreator* creator) {
	creators.emplace(name, creator);
}

void FeatureTester::addJob(std::set<std::string>& set) {
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

	FeatureTesterJob job(featureSetName, featureSet, baseDatasetPath);
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

		
		semaphore.wait();

		threads.push_back(std::thread([job, &resultsFileMutex, &resultStream, &semaphore]() -> void {
			try {

				std::cout << "Starting job " << job.featureSetName << std::endl;
				std::vector<ClassifierEvaluation> results = job.run();
				std::this_thread::sleep_for(std::chrono::milliseconds(5000));


				resultsFileMutex.lock();
				for (auto& eval : results) {
					resultStream << job.featureSetName << ";";
					eval.toCSVLine(resultStream, false);
					resultStream << ";" << std::endl;
				}

				resultsFileMutex.unlock();


				semaphore.notify();
			}
			catch (std::exception e) {

				std::cout << "Error: " << e.what() << std::endl;
				semaphore.notify();
			}
		}));
	}

	// clean up all threads by joining htem
	for (auto& t : threads)
		t.join();
}
