#include "FeatureTesterJob.h"




FeatureTesterJob::FeatureTesterJob(FeatureTester* tester, std::set<std::string> set, DataSet* dataSet, EvaluationSettings settings)
	: tester(tester), dataSet(dataSet), set(set), settings(settings) {
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

DataSet* FeatureTesterJob::getDataSet() const {
	return FeatureTesterJob::dataSet;
}


void FeatureTesterJob::run() const {
	std::string featureSetName = getFeatureName();


	EvaluatorCascade cascade(featureSetName);

	std::string cascadeFile = std::string("models") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_cascade.xml";

	// load and continue if it already exists
	if (FileExists(cascadeFile))
		cascade.load(cascadeFile, std::string("models"));

	// ---------------- Load the initial training set -----------------------
	TrainingDataSet trainingDataSet(dataSet);
	trainingDataSet.load(cascade.trainingRound == 0 ? (std::string("trainingsets") + PATH_SEPARATOR + dataSet->getName() + "_" + "train0.txt") : (std::string("trainingsets") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound) + ".txt"));

	auto featureSet = tester->getFeatureSet(set);

	std::vector<bool> fullfillment = dataSet->getFullfillsRequirements();
	std::vector<bool> requirements = featureSet->getRequirements();
	bool isFullfilled = true;
	// check if the requirements of the feature set matches the fullfillment of the dataset
	for (int i = 0; i < requirements.size(); i++) {
		if (requirements[i] && !fullfillment[i]) { // feature set needs req i, but the dataset doesn't provide this data
			isFullfilled = false;
			break;
		}
	}
	if (!isFullfilled) {
		std::cout << "Can't evaluate the feature set " << featureSetName << " against the data set " << dataSet->getName() << ", requirements not met" << std::endl;
		return;
	}

	featureSet->prepare(trainingDataSet);

	while (cascade.trainingRound < settings.nrOfTrainingRounds) {
		// ---------------- Build a feature set & prepare variable feature creators --------------------

		ModelEvaluator evaluator(dataSet->getName() + "_" + featureSetName + " round " + std::to_string(cascade.trainingRound));

		std::string modelFile = std::string("models") + PATH_SEPARATOR + evaluator.getName() + ".xml";
		// ---------------- Training -----------------------
		if (FileExists(modelFile)) {
			std::cout << "Skipped training of " << featureSetName << ", loading from existing model instead" << std::endl;
			evaluator.loadModel(modelFile);
		}
		else {
			std::cout << "Started training of " << featureSetName << ", round " << cascade.trainingRound << std::endl;
			long elapsedTrainingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluator.train(trainingDataSet, *featureSet, settings.maxWeakClassifiers, settings.trainingCriteria);
			});
			std::cout << "Training round " << cascade.trainingRound << " complete after " << elapsedTrainingTime << "ms for " << featureSetName << std::endl;

			// save it to a file
			evaluator.saveModel(modelFile);
		}
		cascade.addModelEvaluator(evaluator, 0);


		// --------------- Evaluation --------------------
		std::vector<ClassifierEvaluation> evaluations;
		std::string evaluationFile = std::string("results") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_round" + std::to_string(cascade.trainingRound) + ".csv";
		if (FileExists(evaluationFile)) {
			std::cout << "Skipped evaluation of " << featureSetName << ", evaluation was already done." << std::endl;
		}
		else {
			std::cout << "Started evaluation of " << featureSetName << std::endl;
			long elapsedEvaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
				evaluations = cascade.evaluateDataSet(trainingDataSet, *featureSet, settings, false, settings.trainingCriteria);
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
		evaluateTrainingWithSlidingWindow(cascade, featureSetName, trainingDataSet, featureSet);


		// round is finished
		ProgressWindow::getInstance()->finish(featureSetName + " round " + std::to_string(cascade.trainingRound));

		cascade.trainingRound++;
		cascade.save(cascadeFile);
	}

	// --------------- Evaluation sliding window and NMS on test set --------------------
	// Note: no true negatives will be tracked due to NMS
	evaluateTestSet(cascade, cascadeFile, featureSet, featureSetName);


	// ---- Generate feature importance images ----
	generateFeatureImportanceImage(cascade, featureSet);
}

void FeatureTesterJob::evaluateTestSet(EvaluatorCascade& cascade, std::string& cascadeFile, std::unique_ptr<FeatureSet>& featureSet, std::string& featureSetName) const {
	std::string finalEvaluationSlidingFile = std::string("results") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_sliding_final.csv";


	if (!FileExists(finalEvaluationSlidingFile)) {

		// reset classifier hit count
		cascade.resetClassifierHitCount();
		cascade.setTrackClassifierHitCountEnabled(true); // enable hit count tracking so feature importance weight can be collected

		std::cout << "Started final evaluation on test set with sliding window and NMS of " << featureSetName << std::endl;
		FinalEvaluationSlidingWindowResult finalresult;
		long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			finalresult = cascade.evaluateWithSlidingWindowAndNMS(settings, dataSet, *featureSet, settings.testCriteria);

		});

		// save the hit count on the cascade
		cascade.save(cascadeFile);
		std::cout << "Evaluation with sliding window and NMS complete after " << elapsedEvaluationSlidingTime << "ms for " << featureSetName << std::endl;

		std::ofstream str = std::ofstream(finalEvaluationSlidingFile);
		str << "Name" << ";";
		ClassifierEvaluation().toCSVLine(str, true);
		str << std::endl;

		for (auto& category : dataSet->getCategories()) {
			for (auto& result : finalresult.evaluations[category]) {
				str << featureSetName << "[S][" + category + "]" << ";";
				result.toCSVLine(str, false);
				str << std::endl;
			}
		}

		for (auto& result : finalresult.combinedEvaluations) {
			str << featureSetName << "[S]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}
		str.close();

		std::string finalEvaluationMissedPositivesFile = std::string("results") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_missedpositives.csv";

		str = std::ofstream(finalEvaluationMissedPositivesFile);
		for (auto& pair : finalresult.missedPositivesPerImage) {
			int imgNr = pair.first;
			auto& missedPositivesPerShift = pair.second;
			for (int i = 0; i < missedPositivesPerShift.size(); i++) {

				for (int tp = 0; tp < missedPositivesPerShift[i].size(); tp++)
				{
					auto& region = missedPositivesPerShift[i][tp];
					str << imgNr << ";" << cascade.getValueShift(i, settings.nrOfEvaluations, settings.evaluationRange) << ";" <<
						region.x << "," << region.y << "," << region.width << "," << region.height << std::endl;
				}
			}
		}
		str.close();
	}
}

void FeatureTesterJob::evaluateTrainingWithSlidingWindow(EvaluatorCascade& cascade, std::string& featureSetName, TrainingDataSet& trainingDataSet, std::unique_ptr<FeatureSet>& featureSet) const {
	// do not try and evaluate multiple jobs together with sliding window. The sliding window evaluation is already parallelized will just interfere with each other then
	tester->getLock()->lock();

	std::string evaluationSlidingFile = std::string("results") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_sliding_round" + std::to_string(cascade.trainingRound) + ".csv";
	std::string nextRoundTrainingFile = std::string("trainingsets") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound + 1) + ".txt";
	if (FileExists(evaluationSlidingFile) && FileExists(nextRoundTrainingFile)) {
		std::cout << "Skipped evaluation with sliding window of " << featureSetName << ", evaluation was already done and next training set was already present." << std::endl;
		// load the training set for next round
		trainingDataSet.load(nextRoundTrainingFile);
	}
	else {
		std::cout << "Started training evaluation with sliding window of " << featureSetName << std::endl;
		EvaluationSlidingWindowResult result;
		cascade.setTrackClassifierHitCountEnabled(false);
		long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			result = cascade.evaluateWithSlidingWindow(settings, trainingDataSet.getDataSet(), *featureSet, cascade.trainingRound, settings.trainingCriteria);
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


		std::string worstFalsePositivesFile = std::string("models") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_" + "worstfalsepositives_" + std::to_string(cascade.trainingRound) + ".csv";
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

		newTrainingSet.save(std::string("trainingsets") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_" + "train" + std::to_string(cascade.trainingRound + 1) + ".txt");
		trainingDataSet = newTrainingSet;
	}

	tester->getLock()->unlock();
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
		ModelEvaluator& model = cascade.getModelEvaluator(i);

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

		std::string featureImportanceFilename = std::string("results") + PATH_SEPARATOR + dataSet->getName() + "_" + featureSetName + "_" + *it + "_featureimportance.png";
		cv::imwrite(featureImportanceFilename, img);
		it++;
	}

}