#include "FeatureTester.h"
#include "ModelEvaluator.h"
#include "VariableNumberFeatureCreator.h"
#include <fstream>
#include "ProgressWindow.h"



FeatureTesterJob::FeatureTesterJob(std::unordered_map<std::string, FactoryCreator>& creators, std::set<std::string>& set, std::string& baseDataPath, int nrOfEvaluations, int nrOfTrainingRounds)
	: creators(creators), baseDataSetPath(baseDataPath), set(set), nrOfEvaluations(nrOfEvaluations), nrOfTrainingRounds(nrOfTrainingRounds) {

}


std::string FeatureTesterJob::getFeatureName() const {
	std::string featureSetName("");
	for (auto& name : set) {
		FactoryCreator creator = creators.find(name)->second;
		if (name != *(set.begin()))
			featureSetName += "+" + name;
		else
			featureSetName += name;
	}
	return featureSetName;
}

void FeatureTesterJob::run() const {

	int maxNrWorstPosNeg = 4000;

	std::string featureSetName = getFeatureName();
	int trainingRound = 0;

	// ---------------- Load a training set -----------------------
	std::string dataSetPath = std::string(baseDataSetPath);
	TrainingDataSet trainingDataSet(dataSetPath);
	trainingDataSet.load(trainingRound == 0 ? (std::string("trainingsets") + PATH_SEPARATOR + "train0.txt") : (std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(trainingRound) + ".txt"));

	while (trainingRound < nrOfTrainingRounds) {
		// ---------------- Build a feature set & prepare variable feature creators --------------------
		FeatureSet featureSet;
		for (auto& name : set) {
			FactoryCreator creator = creators.find(name)->second;

			std::unique_ptr<IFeatureCreator> featureCreator = creator.createInstance(creator.name);
			if (dynamic_cast<VariableNumberFeatureCreator*>(featureCreator.get()) != nullptr)
				(dynamic_cast<VariableNumberFeatureCreator*>(featureCreator.get()))->prepare(trainingDataSet, trainingRound);
			featureSet.addCreator(std::move(featureCreator));
		}

		ModelEvaluator evaluator(featureSetName + " round " + std::to_string(trainingRound), trainingDataSet, featureSet);

		
		// ---------------- Training -----------------------
		std::string modelFile = trainingRound == 0 ? std::string("models") + PATH_SEPARATOR + featureSetName + ".xml" : std::string("models") + PATH_SEPARATOR + featureSetName + "_round" + std::to_string(trainingRound) + ".xml";
		if (FileExists(modelFile)) {
			std::cout << "Skipped training of " << featureSetName << ", loading from existing model instead" << std::endl;
			evaluator.loadModel(modelFile);
		}
		else {
			std::cout << "Started training of " << featureSetName << ", round " << trainingRound << std::endl;
			long elapsedTrainingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluator.train();
			});
			std::cout << "Training round " << trainingRound << " complete after " << elapsedTrainingTime << "ms for " << featureSetName << std::endl;

			// save it to a file
			evaluator.saveModel(modelFile);
		}

		// --------------- Evaluation --------------------
		std::vector<ClassifierEvaluation> evaluations;
		std::string evaluationFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_round" + std::to_string(trainingRound) + ".csv";
		if (FileExists(evaluationFile)) {
			std::cout << "Skipped evaluation of " << featureSetName << ", evaluation was already done." << std::endl;
		}
		else {
			std::cout << "Started evaluation of " << featureSetName << std::endl;
			long elapsedEvaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluations = evaluator.evaluateDataSet(nrOfEvaluations, false);
			});
			std::ofstream str(evaluationFile);
			ClassifierEvaluation().toCSVLine(str, true);
			for (auto& result : evaluations) {
				str << featureSetName << "_round" << trainingRound << ";";
				result.toCSVLine(str, false);
				str << std::endl;
			}
			std::cout << "Evaluation complete after " << elapsedEvaluationTime << "ms for " << featureSetName << std::endl;
		}
		// --------------- Evaluation sliding window --------------------
		std::string evaluationSlidingFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_sliding_round" + std::to_string(trainingRound) + ".csv";
		std::string nextRoundTrainingFile = std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(trainingRound + 1) + ".txt";
		if (FileExists(evaluationSlidingFile) && FileExists(nextRoundTrainingFile)) {
			std::cout << "Skipped evaluation with sliding window of " << featureSetName << ", evaluation was already done and next training set was already present." << std::endl;
			// load the training set for next round
			trainingDataSet.load(nextRoundTrainingFile);
		}
		else {
			std::cout << "Started evaluation with sliding window of " << featureSetName << std::endl;
			EvaluationSlidingWindowResult result;
			long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				result = evaluator.evaluateWithSlidingWindow(nrOfEvaluations, trainingRound, 0, maxNrWorstPosNeg);
			});
			std::cout << "Evaluation with sliding window complete after " << elapsedEvaluationSlidingTime << "ms for " << featureSetName << std::endl;

			std::ofstream str = std::ofstream(evaluationSlidingFile);
			ClassifierEvaluation().toCSVLine(str, true);
			for (auto& result : result.evaluations) {
				str << featureSetName << "[S]_round" << trainingRound << ";";
				result.toCSVLine(str, false);
				str << std::endl;
			}

			// --------------- New training set --------------------
			TrainingDataSet newTrainingSet = trainingDataSet;
			for (auto& swregion : result.worstFalsePositives) {

				TrainingRegion r;
				r.region = swregion.bbox;
				r.regionClass = -1; // it was a negative but positive was specified
				newTrainingSet.addTrainingRegion(swregion.imageNumber, r);
			}

			for (auto& swregion : result.worstFalseNegatives) {

				TrainingRegion r;
				r.region = swregion.bbox;
				r.regionClass = 1; // it was a positive but negative was specified
				newTrainingSet.addTrainingRegion(swregion.imageNumber, r);
			}
			newTrainingSet.save(std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(trainingRound + 1) + ".txt");
			trainingDataSet = newTrainingSet;
		}

		ProgressWindow::getInstance()->finish(featureSetName + " round " + std::to_string(trainingRound));

		trainingRound++;
	}
}


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

void FeatureTester::markFeatureSetProcessed(std::string& featureSetName) {
	processedFeatureSets.emplace(featureSetName);
	std::ofstream str("processedsets.txt", std::ofstream::out | std::ofstream::app);
	if (!str.is_open())
		return;
	str << featureSetName << std::endl;
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

void FeatureTester::addJob(std::set<std::string>& set, std::string& baseDataSetPath, int nrOfEvaluations, int nrOfTrainingRounds) {
	FeatureTesterJob job(creators,set, baseDataSetPath, nrOfEvaluations,  nrOfTrainingRounds);
	jobs.push(job);
}

void FeatureTester::runJobs() {

	
	// make sure all creates are ready to roll
	//prepareCreators();

	std::mutex resultsFileMutex;

	// run 4 jobs concurrently
	Semaphore semaphore(nrOfConcurrentJobs);

	std::vector<std::thread> threads;
	while (jobs.size() > 0) {
		FeatureTesterJob job = jobs.front();
		jobs.pop();


		std::string featureSetName = job.getFeatureName();
		if (processedFeatureSets.find(featureSetName) != processedFeatureSets.end()) {
			std::cout << "Job " << featureSetName << " is already processed and will be skipped" << std::endl;
		}
		else {
			semaphore.wait();

			FeatureTester* ft = this;
			threads.push_back(std::thread([job, &resultsFileMutex, &semaphore, &ft, featureSetName]() -> void {
				try {

					std::cout << "Starting job " << featureSetName << std::endl;
					job.run();
					
					resultsFileMutex.lock();
					ft->markFeatureSetProcessed(std::string(featureSetName));
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
