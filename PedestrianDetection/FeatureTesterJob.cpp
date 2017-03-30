#include "FeatureTesterJob.h"




FeatureTesterJob::FeatureTesterJob(FeatureTester* tester, std::vector<cv::Size>& windowSizes, std::set<std::string> set, std::string& baseDataPath, int nrOfEvaluations, int nrOfTrainingRounds, bool evaluateOnSlidingWindow)
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
	int maxWeakClassifiers = 500;

	// faster, change later
	std::function<bool(int)> trainingCriteria = [](int imageNumber) -> bool { return imageNumber % 2 == 0; };
	std::function<bool(int)> testCriteria = [](int imageNumber) -> bool { return imageNumber % 2 == 1; };

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
				evaluator.train(trainingDataSet, *featureSet, maxWeakClassifiers, trainingCriteria);
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

		// --------------- Evaluation of training with sliding window --------------------
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

	// --------------- Evaluation sliding window and NMS on test set --------------------
	// Note: no true negatives will be tracked due to NMS
	std::string finalEvaluationSlidingFile = std::string("results") + PATH_SEPARATOR + featureSetName + "_sliding_final.csv";
	if (!FileExists(finalEvaluationSlidingFile)) {

		// reset classifier hit count
		cascade.resetClassifierHitCount();

		auto featureSet = tester->getFeatureSet(set);
		featureSet->prepare(trainingDataSet, 0); // always use base bag of words

		std::cout << "Started final evaluation on test set with sliding window and NMS of " << featureSetName << std::endl;
		FinalEvaluationSlidingWindowResult finalresult;
		long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			finalresult = cascade.evaluateWithSlidingWindowAndNMS(windowSizes, trainingDataSet.getDataSet(), *featureSet, nrOfEvaluations, testCriteria);

		});

		// save the hit count on the cascade
		cascade.save(cascadeFile);
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


	// ---- Generate feature importance images ----

	auto featureSet = tester->getFeatureSet(set);
	featureSet->prepare(trainingDataSet, 0);
	generateFeatureImportanceImage(cascade, featureSet);
}


void FeatureTesterJob::generateFeatureImportanceImage(EvaluatorCascade& cascade, std::unique_ptr<FeatureSet>& fset) const {
	int refWidth = 64;
	int refHeight = 128;

	auto classifierHits = cascade.getClassifierHitCount();

	int classifierHitSum = 0;
	for (auto val : classifierHits) classifierHitSum += val;

	int rounds = cascade.size();
	int padding = 5;

	// don't initialize directly or it will point to the same data
	std::vector<cv::Mat> imgs;// (set.size(), cv::Mat(cv::Size(refWidth, refHeight * 4), CV_32FC1, cv::Scalar(0)));
	for (int i = 0; i < cascade.size(); i++)
		imgs.push_back(cv::Mat(cv::Size((refWidth + padding)*rounds + 4 * padding + refWidth, refHeight), CV_32FC1, cv::Scalar(0)));


	std::vector<cv::Mat> totalImgs;
	for (int i = 0; i < cascade.size(); i++)
		totalImgs.push_back(cv::Mat(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0)));

	cv::Mat rgb(128, 64, CV_8UC3, cv::Scalar(0));
	cv::Mat depth(128, 64, CV_32FC1, cv::Scalar(0));


	std::string featureSetName = getFeatureName();

	for (int i = 0; i < rounds; i++)
	{
		ModelEvaluator model(featureSetName);
		model.loadModel(std::string("models") + PATH_SEPARATOR + featureSetName + " round " + std::to_string(i) + ".xml");

		auto cur = model.explainModel(fset, refWidth, refHeight);

		for (int j = 0; j < cur.size(); j++) {
			if (cur[j].rows > 0 && cur[j].cols > 0) {
				cv::normalize(cur[j], cur[j], 0, 1, cv::NormTypes::NORM_MINMAX);

				totalImgs[j] += cur[j] * (1.0 * classifierHits[j] / classifierHitSum);

				cv::Mat& dst = imgs[j](cv::Rect(i*(refWidth + padding), 0, refWidth, refHeight));
				cur[j].copyTo(dst);
			}
		}
	}


	auto it = set.begin();
	for (int i = 0; i < imgs.size(); i++) {
		cv::Mat img;

		img = imgs[i];
		img = heatmap::toHeatMap(img);


		cv::Mat totalimage = totalImgs[i];
		cv::normalize(totalimage, totalimage, 0, 1, cv::NormTypes::NORM_MINMAX);
		cv::Mat heatmapTotalImage = heatmap::toHeatMap(totalimage);
		heatmapTotalImage.copyTo(img(cv::Rect(img.cols - refWidth, 0, refWidth, refHeight)));

		std::string featureImportanceFilename = std::string("results") + PATH_SEPARATOR + featureSetName + "_" + *it + "_featureimportance.png";
		cv::imwrite(featureImportanceFilename, img);
		it++;
	}

}