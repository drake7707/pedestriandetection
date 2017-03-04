#include "ModelEvaluator.h"
#include "Helper.h"
#include <queue>

ModelEvaluator::ModelEvaluator(std::string& name, const TrainingDataSet& trainingDataSet, const FeatureSet& set)
	: trainingDataSet(trainingDataSet), set(set), name(name)
{
}


ModelEvaluator::~ModelEvaluator()
{
}




void ModelEvaluator::train()
{

	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;


	trainingDataSet.iterateDataSet([&](int idx) -> bool { return idx % trainEveryXImage != 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Building feature vectors (") + std::to_string(imageNumber) + ")");

		FeatureVector v = set.getFeatures(rgb, depth);
		if (resultClass == 1)
			truePositiveFeatures.push_back(v);
		else
			trueNegativeFeatures.push_back(v);
	});

	int featureSize = truePositiveFeatures[0].size();

	// build mean and sigma vector
	int N = truePositiveFeatures.size() + trueNegativeFeatures.size();
	model.meanVector = std::vector<float>(featureSize, 0);
	model.sigmaVector = std::vector<float>(featureSize, 0);

	for (auto& featureVector : truePositiveFeatures) {
		for (int f = 0; f < featureSize; f++)
			model.meanVector[f] += featureVector[f];
	}
	for (auto& featureVector : trueNegativeFeatures) {
		for (int f = 0; f < featureSize; f++)
			model.meanVector[f] += featureVector[f];
	}

	for (int f = 0; f < featureSize; f++)
		model.meanVector[f] /= N;

	std::vector<double> sigmaSum(featureSize, 0);
	for (auto& featureVector : truePositiveFeatures) {
		for (int f = 0; f < featureSize; f++)
			sigmaSum[f] += (featureVector[f] - model.meanVector[f]) * (featureVector[f] - model.meanVector[f]);
	}
	for (auto& featureVector : trueNegativeFeatures) {
		for (int f = 0; f < featureSize; f++)
			sigmaSum[f] += (featureVector[f] - model.meanVector[f]) * (featureVector[f] - model.meanVector[f]);
	}

	for (int f = 0; f < featureSize; f++)
		model.sigmaVector[f] = sigmaSum[f] / (N - 1);

	// now apply the normalization on the feature arrays
	for (auto& featureVector : truePositiveFeatures)
		featureVector.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	for (auto& featureVector : trueNegativeFeatures)
		featureVector.applyMeanAndVariance(model.meanVector, model.sigmaVector);


	// build training dataset
	cv::Mat trainingMat(truePositiveFeatures.size() + trueNegativeFeatures.size(), featureSize, CV_32FC1);
	cv::Mat trainingLabels(truePositiveFeatures.size() + trueNegativeFeatures.size(), 1, CV_32SC1);

	int idx = 0;
	for (int i = 0; i < truePositiveFeatures.size(); i++)
	{
		for (int f = 0; f < featureSize; f++) {

			if (isnan(truePositiveFeatures[i][f]))
				throw std::exception("Feature contains NaN");

			trainingMat.at<float>(idx, f) = truePositiveFeatures[i][f];

		}
		trainingLabels.at<int>(idx, 0) = 1;
		idx++;
	}

	for (int i = 0; i < trueNegativeFeatures.size(); i++) {
		for (int f = 0; f < featureSize; f++) {
			if (isnan(trueNegativeFeatures[i][f]))
				throw std::exception("Feature contains NaN");

			trainingMat.at<float>(idx, f) = trueNegativeFeatures[i][f];
		}
		trainingLabels.at<int>(idx, 0) = -1;
		idx++;
	}

	cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingMat, cv::ml::SampleTypes::ROW_SAMPLE, trainingLabels,
		cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());


	std::vector<double>	priors(2);
	priors[0] = trueNegativeFeatures.size();
	priors[1] = truePositiveFeatures.size();

	boost->setPriors(cv::Mat(priors));
	boost->setWeakCount(200);
	//boost->setWeightTrimRate(0.95);
	//boost->setMaxDepth(5);
	//boost->setUseSurrogates(false);

	ProgressWindow::getInstance()->updateStatus(name, 0, std::string("Training boost classifier"));

	std::cout << "Training boost classifier on " << N << " samples, feature size " << featureSize << std::endl;

	double trainingTimeMS = measure<std::chrono::milliseconds>::execution([&]() -> void {
		boost->train(tdata);
	});
	//std::cout << "Done training, took " << trainingTimeMS << "ms" << std::endl;

	if (!boost->isTrained())
		throw std::exception("Boost training failed");

	model.boost = boost;

	//auto& roots = boost->getRoots();
	//auto& nodes = boost->getNodes();
	//auto& splits = boost->getSplits();
	//for (auto& r : roots) {
	//	std::cout << "Root split " << nodes[r].split << " variable index " << splits[nodes[r].split].varIdx << " quality: " << splits[nodes[r].split].quality << " explain: " << set.explainFeature(splits[nodes[r].split].varIdx, nodes[r].value) << std::endl;
	//}
}

std::vector<ClassifierEvaluation> ModelEvaluator::evaluateDataSet(int nrOfEvaluations, bool includeRawResponses) {
	std::vector<ClassifierEvaluation> evals(nrOfEvaluations, ClassifierEvaluation());

	std::vector<double> sumTimes(nrOfEvaluations, 0);
	std::vector<int> nrRegions(nrOfEvaluations, 0);
	double featureBuildTime = 0;

	trainingDataSet.iterateDataSet([&](int idx) -> bool { return idx % trainEveryXImage == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating training set regions (") + std::to_string(imageNumber) + ")");

		FeatureVector v;
		featureBuildTime += measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth);
			v.applyMeanAndVariance(model.meanVector, model.sigmaVector);
		});


		// get datapoints, ranging from -10 to 10
		for (int i = 0; i < nrOfEvaluations; i++)
		{
			double valueShift = 1.0 * i / nrOfEvaluations * 20 - 10;
			evals[i].valueShift = valueShift;
			sumTimes[i] += measure<std::chrono::milliseconds>::execution([&]() -> void {

				EvaluationResult result = evaluateFeatures(v, valueShift);
				bool correct;
				if (resultClass == result.resultClass) {
					if (resultClass == -1)
						evals[i].nrOfTrueNegatives++;
					else
						evals[i].nrOfTruePositives++;

					correct = true;
				}
				else {
					if (resultClass == -1 && result.resultClass == 1)
						evals[i].nrOfFalsePositives++;
					else
						evals[i].nrOfFalseNegatives++;

					correct = false;
				}

				if (includeRawResponses)
					evals[i].rawValues.push_back(RawEvaluationEntry(imageNumber, region, result.rawResponse, correct));
			});
			nrRegions[i]++;
		}
	});

	for (int i = 0; i < nrOfEvaluations; i++)
		evals[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes[i]) / nrRegions[i];

	return evals;
}

EvaluationResult ModelEvaluator::evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift) const {

	FeatureVector v = set.getFeatures(rgb, depth);
	v.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	return evaluateFeatures(v, valueShift);
}


EvaluationResult ModelEvaluator::evaluateFeatures(FeatureVector& v, double valueShift) const {

	auto& nodes = model.boost->getNodes();
	auto& roots = model.boost->getRoots();
	auto& splits = model.boost->getSplits();

	double sum = 0.;
	for (auto& r : roots) {

		int nidx = r, prev = nidx, c = 0;


		while (true) {
			auto& n = nodes[r];
			prev = nidx;
			const cv::ml::DTrees::Node& node = nodes[nidx];
			if (node.split < 0)
				break;
			const cv::ml::DTrees::Split& split = splits[node.split];
			int vi = split.varIdx;
			float val = v[vi];
			nidx = val <= split.c ? node.left : node.right;
		}
		sum += nodes[prev].value;
	}

	//float result = model.boost->predict(v.toMat(), cv::noArray());

	return EvaluationResult((sum + valueShift) > 0 ? 1 : -1, sum);
}


EvaluationSlidingWindowResult ModelEvaluator::evaluateWithSlidingWindow(int nrOfEvaluations, int trainingRound, float valueShiftForFalsePosOrNegCollection, int maxNrOfFalsePosOrNeg) {

	EvaluationSlidingWindowResult swresult;
	swresult.evaluations = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation());

	std::vector<double> sumTimes(nrOfEvaluations, 0);
	std::vector<int> nrRegions(nrOfEvaluations, 0);
	double featureBuildTime = 0;

	auto comp = [](std::pair<float, SlidingWindowRegion> a, std::pair<float, SlidingWindowRegion> b) { return a.first > b.first; };
	std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)> worstFalsePositives(comp);
	std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)> worstFalseNegatives(comp);

	std::mutex mutex;
	std::function<void(std::function<void()>)> lock = [&](std::function<void()> func) -> void {
		mutex.lock();
		func();
		mutex.unlock();
	};;

	trainingDataSet.iterateDataSetWithSlidingWindow([&](int idx) -> bool { return (idx + trainingRound) % slidingWindowEveryXImage == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating with sliding window (") + std::to_string(imageNumber) + ")");

		FeatureVector v;
		long buildTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth);
			v.applyMeanAndVariance(model.meanVector, model.sigmaVector);
		});

		lock([&]() -> void { // lock everything that is outside the function body as the iteration is done over multiple threads
			featureBuildTime += buildTime;
		});

		// prepare vectors locally to a single thread
		std::vector<EvaluationResult> results;
		std::vector<long> resultTimes;
		results.reserve(nrOfEvaluations);
		resultTimes.reserve(nrOfEvaluations);
		for (int i = 0; i < nrOfEvaluations; i++)
		{
			// get datapoints, ranging from -10 to 10
			double valueShift = 1.0 * i / nrOfEvaluations * 20 - 10;
			long time = measure<std::chrono::milliseconds>::execution([&]() -> void {
				EvaluationResult result = evaluateFeatures(v, valueShift);
				results.push_back(result);
			});
			resultTimes.push_back(time);
		}

		// now update everything that is shared across threads from the built vectors
		lock([&]() -> void {
			for (int i = 0; i < nrOfEvaluations; i++)
			{
				sumTimes[i] += resultTimes[i];
				double valueShift = 1.0 * i / nrOfEvaluations * 20 - 10;
				swresult.evaluations[i].valueShift = valueShift;

				EvaluationResult result = results[i];
				bool correct;
				if (resultClass == result.resultClass) {
					if (resultClass == -1)
						swresult.evaluations[i].nrOfTrueNegatives++;
					else {
						swresult.evaluations[i].nrOfTruePositives++;
						//if (valueShift == valueShiftForFalsePosOrNegCollection) // TODO TMP REMOVE
						//	cv::rectangle(fullrgb, region, cv::Scalar(0, 255, 0), 1);
					}

					correct = true;
				}
				else {
					if (resultClass == -1 && result.resultClass == 1) {
						swresult.evaluations[i].nrOfFalsePositives++;

						//if (valueShift == valueShiftForFalsePosOrNegCollection) // TODO TMP REMOVE
						//	cv::rectangle(fullrgb, region, cv::Scalar(0, 0, 255), 1);
					}
					else
						swresult.evaluations[i].nrOfFalseNegatives++;

					correct = false;
				}

				if (valueShift == valueShiftForFalsePosOrNegCollection && !correct) {
					if (result.resultClass == 1) {
						worstFalsePositives.push(std::pair<float, SlidingWindowRegion>(abs(result.rawResponse), SlidingWindowRegion(imageNumber, region)));
						// smallest numbers will be popped
						if (worstFalsePositives.size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
							worstFalsePositives.pop();
					}
					else {
						worstFalseNegatives.push(std::pair<float, SlidingWindowRegion>(abs(result.rawResponse), SlidingWindowRegion(imageNumber, region)));
						// smallest numbers will be popped
						if (worstFalsePositives.size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
							worstFalsePositives.pop();
					}
				}
				nrRegions[i]++;
			}
		});
	});

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives
	for (int i = 0; i < nrOfEvaluations; i++)
		swresult.evaluations[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes[i]) / nrRegions[i];

	swresult.worstFalsePositives.reserve(worstFalsePositives.size());
	swresult.worstFalseNegatives.reserve(worstFalseNegatives.size());

	while (worstFalsePositives.size() > 0) {
		auto pair = worstFalsePositives.top();
		worstFalsePositives.pop();
		std::cout << "Worst FP region score= " << pair.first << " image= " << pair.second.imageNumber << " bbox=" << pair.second.bbox.x << "," << pair.second.bbox.y << " " << pair.second.bbox.width << "x" << pair.second.bbox.height << std::endl;
		swresult.worstFalsePositives.push_back(pair.second);
	}
	while (worstFalseNegatives.size() > 0) {
		auto pair = worstFalseNegatives.top();
		worstFalseNegatives.pop();
		std::cout << "Worst FN region score= " << pair.first << " image= " << pair.second.imageNumber << " bbox=" << pair.second.bbox.x << "," << pair.second.bbox.y << " " << pair.second.bbox.width << "x" << pair.second.bbox.height << std::endl;
		swresult.worstFalseNegatives.push_back(pair.second);
	}

	return swresult;
}


void ModelEvaluator::saveModel(std::string& path) {
	if (!model.boost->isTrained())
		throw std::exception("Model is not trained");

	std::string filename = path;

	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	model.boost->write(fs);
	fs << "mean" << model.meanVector;
	fs << "sigma" << model.sigmaVector;
	//model.boost->save(filename + ".boost.xml");
	fs.release();

}

void ModelEvaluator::loadModel(std::string& path) {

	std::string filename = path;

	//model.boost = cv::Algorithm::load<cv::ml::Boost>(filename + ".boost.xml");
	cv::FileStorage fsRead(filename, cv::FileStorage::READ);

	model.boost = cv::Algorithm::read<cv::ml::Boost>(fsRead.root());
	fsRead["mean"] >> model.meanVector;
	fsRead["sigma"] >> model.sigmaVector;
	fsRead.release();
}