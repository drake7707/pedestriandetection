#include "FeatureTesterJob.h"
#include "EvaluatorCascade.h"




FeatureTesterJob::FeatureTesterJob(FeatureTester* tester, std::vector<cv::Size>& windowSizes, std::set<std::string>& set, std::string& baseDataPath, int nrOfEvaluations, int nrOfTrainingRounds, bool evaluateOnSlidingWindow)
	: tester(tester), baseDataSetPath(baseDataPath), windowSizes(windowSizes), set(set), nrOfEvaluations(nrOfEvaluations), nrOfTrainingRounds(nrOfTrainingRounds), evaluateOnSlidingWindow(evaluateOnSlidingWindow) {

	if (nrOfTrainingRounds > 1 && !evaluateOnSlidingWindow) {
		throw std::exception("Evaluate on sliding window is required when running multiple training rounds");
	}
}


std::string FeatureTesterJob::getFeatureName() const {
	std::string featureSetName("");
	for (auto& name : set) {
		if (name != *(set.begin()))
			featureSetName += "+" + name;
		else
			featureSetName += name;
	}
	return featureSetName;
}



void FeatureTesterJob::run() const {

	int maxNrWorstPosNeg = 4000;
	float requiredTPRRate = 0.95;

	// faster, change later
	std::function<bool(int)> trainingCriteria = [](int imageNumber) -> bool { return imageNumber % 20 == 0; };
	std::function<bool(int)> testCriteria = [](int imageNumber) -> bool { return imageNumber % 20 == 1; };


	std::string featureSetName = getFeatureName();


	EvaluatorCascade cascade(featureSetName);

	std::string cascadeFile = std::string("models") + PATH_SEPARATOR + featureSetName + "_cascade.xml";

	// load and continue if it already exists
	if (FileExists(cascadeFile))
		cascade.load(cascadeFile, std::string("models"));

	// ---------------- Load a training set -----------------------
	std::string dataSetPath = std::string(baseDataSetPath);
	TrainingDataSet trainingDataSet(dataSetPath);
	trainingDataSet.load(cascade.trainingRound == 0 ? (std::string("trainingsets") + PATH_SEPARATOR + "train0.txt") : (std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound) + ".txt"));

	while (cascade.trainingRound < nrOfTrainingRounds) {
		// ---------------- Build a feature set & prepare variable feature creators --------------------
		auto featureSet = tester->getFeatureSet(set);
		featureSet->prepare(trainingDataSet, 0); // always use base
		//featureSet->prepare(trainingDataSet, cascade.trainingRound);

		ModelEvaluator evaluator(featureSetName + " round " + std::to_string(cascade.trainingRound));

		std::string modelFile = std::string("models") + PATH_SEPARATOR + evaluator.getName() + ".xml";
		// ---------------- Training -----------------------
		if (FileExists(modelFile)) {
			std::cout << "Skipped training of " << featureSetName << ", loading from existing model instead" << std::endl;
			evaluator.loadModel(modelFile);
		}
		else {
			std::cout << "Started training of " << featureSetName << ", round " << cascade.trainingRound << std::endl;
			long elapsedTrainingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluator.train(trainingDataSet, *featureSet, trainingCriteria);
			});
			std::cout << "Training round " << cascade.trainingRound << " complete after " << elapsedTrainingTime << "ms for " << featureSetName << std::endl;

			// save it to a file
			evaluator.saveModel(modelFile);
		}
		cascade.addModelEvaluator(evaluator, 0);


		// --------------- Evaluation --------------------

		double valueShiftRequiredForTPR95Percent = 0;

		std::vector<ClassifierEvaluation> evaluations;
		std::string evaluationFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_round" + std::to_string(cascade.trainingRound) + ".csv";
		if (FileExists(evaluationFile)) {
			std::cout << "Skipped evaluation of " << featureSetName << ", evaluation was already done." << std::endl;
		}
		else {
			std::cout << "Started evaluation of " << featureSetName << std::endl;
			long elapsedEvaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluations = cascade.evaluateDataSet(trainingDataSet, *featureSet, nrOfEvaluations, false, trainingCriteria);
			});
			std::ofstream str(evaluationFile);
			str << "Name" << ";";
			ClassifierEvaluation().toCSVLine(str, true);
			str << std::endl;
			for (auto& result : evaluations) {
				str << featureSetName << "_round" << cascade.trainingRound << ";";
				result.toCSVLine(str, false);
				str << std::endl;
			}
			std::cout << "Evaluation complete after " << elapsedEvaluationTime << "ms for " << featureSetName << std::endl;
		}

		// --------------- Evaluation sliding window --------------------
		if (evaluateOnSlidingWindow) {

			// do not try and evaluate multiple jobs together with sliding window. The sliding window evaluation is already parallelized will just interfere with each other then
			tester->getLock()->lock();

			std::string evaluationSlidingFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_sliding_round" + std::to_string(cascade.trainingRound) + ".csv";
			std::string nextRoundTrainingFile = std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound + 1) + ".txt";
			if (FileExists(evaluationSlidingFile) && FileExists(nextRoundTrainingFile)) {
				std::cout << "Skipped evaluation with sliding window of " << featureSetName << ", evaluation was already done and next training set was already present." << std::endl;
				// load the training set for next round
				trainingDataSet.load(nextRoundTrainingFile);
			}
			else {
				std::cout << "Started training evaluation with sliding window of " << featureSetName << std::endl;
				EvaluationSlidingWindowResult result;
				long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
					result = cascade.evaluateWithSlidingWindow(windowSizes, trainingDataSet.getDataSet(), *featureSet, nrOfEvaluations, cascade.trainingRound, requiredTPRRate, maxNrWorstPosNeg,
						trainingCriteria);
				});
				std::cout << "Evaluation with sliding window complete after " << elapsedEvaluationSlidingTime << "ms for " << featureSetName << std::endl;

				std::ofstream str = std::ofstream(evaluationSlidingFile);
				str << "Name" << ";";
				ClassifierEvaluation().toCSVLine(str, true);
				str << std::endl;
				for (auto& result : result.evaluations) {
					str << featureSetName << "[S]_round" << cascade.trainingRound << ";";
					result.toCSVLine(str, false);
					str << std::endl;
				}

				cascade.updateLastModelValueShift(result.evaluations[result.evaluationIndexWhenTPRThresholdIsReached].valueShift);


				std::string worstFalsePositivesFile = std::string("models") + PATH_SEPARATOR + featureSetName + "_" + "worstfalsepositives_" + std::to_string(cascade.trainingRound) + ".csv";
				std::ofstream wfpStr = std::ofstream(worstFalsePositivesFile);

				wfpStr << std::fixed;
				wfpStr << "Name;Score;ImageNumber;X;Y;Width;Height" << std::endl;
				// --------------- New training set --------------------
				TrainingDataSet newTrainingSet = trainingDataSet;
				for (auto& swregion : result.worstFalsePositives) {

					TrainingRegion r;
					r.region = swregion.bbox;
					r.regionClass = -1; // it was a negative but positive was specified
					newTrainingSet.addTrainingRegion(swregion.imageNumber, r);

					wfpStr << featureSetName << ";" << swregion.score << ";" << swregion.imageNumber << ";" << swregion.bbox.x << ";" << swregion.bbox.y << ";" << swregion.bbox.width << ";" << swregion.bbox.height << std::endl;
				}
				wfpStr.close();

				newTrainingSet.save(std::string("trainingsets") + PATH_SEPARATOR + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound + 1) + ".txt");
				trainingDataSet = newTrainingSet;
			}

			tester->getLock()->unlock();
		}

		// round is finished
		ProgressWindow::getInstance()->finish(featureSetName + " round " + std::to_string(cascade.trainingRound));

		cascade.trainingRound++;
		cascade.save(cascadeFile);
	}


	// Note: no true negatives will be tracked due to NMS
	std::string finalEvaluationSlidingFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_sliding_final.csv";
	if (!FileExists(finalEvaluationSlidingFile)) {
		auto featureSet = tester->getFeatureSet(set);
		featureSet->prepare(trainingDataSet, 0); // always use base bag of words

		std::cout << "Started final evaluation on test set with sliding window and NMS of " << featureSetName << std::endl;
		FinalEvaluationSlidingWindowResult finalresult;
		long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			finalresult = cascade.evaluateWithSlidingWindowAndNMS(windowSizes, trainingDataSet.getDataSet(), *featureSet, nrOfEvaluations, testCriteria);

		});
		std::cout << "Evaluation with sliding window and NMS complete after " << elapsedEvaluationSlidingTime << "ms for " << featureSetName << std::endl;

		std::ofstream str = std::ofstream(finalEvaluationSlidingFile);
		str << "Name" << ";";
		ClassifierEvaluation().toCSVLine(str, true);
		str << std::endl;
		for (auto& result : finalresult.evaluations["easy"]) {
			str << featureSetName << "[S][E]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.evaluations["moderate"]) {
			str << featureSetName << "[S][M]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.evaluations["hard"]) {
			str << featureSetName << "[S][H]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.combinedEvaluations) {
			str << featureSetName << "[S]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}
	}


}
