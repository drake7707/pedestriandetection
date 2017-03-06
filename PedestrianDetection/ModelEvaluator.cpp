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

	double tnWeight = trueNegativeFeatures.size();// *1.0 / N;
	double tpWeight = truePositiveFeatures.size();// *1.0 / N;
	priors[0] = tnWeight;
	priors[1] = tpWeight;

	boost->setPriors(cv::Mat(priors));
	boost->setWeakCount(500);
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

	double sumTimes = 0;
	int nrRegions = 0;
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

		double resultSum;
		sumTimes += measure<std::chrono::milliseconds>::execution([&]() -> void {
			resultSum = evaluateFeatures(v);
		});
		nrRegions++;
		for (int i = 0; i < nrOfEvaluations; i++)
		{
			double valueShift = 1.0 * i / nrOfEvaluations * 20 - 10;
			evals[i].valueShift = valueShift;

			int evaluationClass = resultSum+valueShift > 0 ? 1 : -1;

			bool correct;
			if (resultClass == evaluationClass) {
				if (resultClass == -1)
					evals[i].nrOfTrueNegatives++;
				else
					evals[i].nrOfTruePositives++;

				correct = true;
			}
			else {
				if (resultClass == -1 && evaluationClass == 1)
					evals[i].nrOfFalsePositives++;
				else
					evals[i].nrOfFalseNegatives++;

				correct = false;
			}

			if (includeRawResponses)
				evals[i].rawValues.push_back(RawEvaluationEntry(imageNumber, region, resultSum+valueShift, correct));

			
		}
	});

	for (int i = 0; i < nrOfEvaluations; i++)
		evals[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes) / nrRegions;

	return evals;
}

double ModelEvaluator::evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const {

	FeatureVector v = set.getFeatures(rgb, depth);
	v.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	return evaluateFeatures(v);
}


double ModelEvaluator::evaluateFeatures(FeatureVector& featureVector) const {

	auto& nodes = model.boost->getNodes();
	auto& roots = model.boost->getRoots();
	auto& splits = model.boost->getSplits();

	double sum = 0.;
	// iterate all trees starting from their root
	for (auto& r : roots) {

		int nidx = r, prev = nidx;

		// traverse path to leaf, selecting the corresponding value from the feature vector
		while (true) {
			prev = nidx;
			const cv::ml::DTrees::Node& node = nodes[nidx];
			if (node.split < 0) // no more split, reached leaf
				break;
			const cv::ml::DTrees::Split& split = splits[node.split];
			float val = featureVector[split.varIdx];
			nidx = (val <= split.c) ? node.left : node.right;
		}
		sum += nodes[prev].value;
	}

	//float result = model.boost->predict(v.toMat(), cv::noArray());
	return sum;
}


EvaluationSlidingWindowResult ModelEvaluator::evaluateWithSlidingWindow(int nrOfEvaluations, int trainingRound, float valueShiftForFalsePosOrNegCollection, int maxNrOfFalsePosOrNeg) {

	EvaluationSlidingWindowResult swresult;
	swresult.evaluations = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation());

	double sumTimesRegions = 0;
	int nrRegions = 0;
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

	double worstFalsePositiveValue = std::numeric_limits<double>().min();

	trainingDataSet.iterateDataSetWithSlidingWindow([&](int idx) -> bool { return (idx + trainingRound) % slidingWindowEveryXImage == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating with sliding window (") + std::to_string(imageNumber) + ")");

		FeatureVector v;
		long buildTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth);
			v.applyMeanAndVariance(model.meanVector, model.sigmaVector);
		});

		lock([&]() -> void { // lock everything that is outside the function body as the iteration is done over multiple threads
			featureBuildTime += buildTime;
			nrRegions++;
		});
	
		// just evaluate it once and shift it
		double baseEvaluationSum;
		long time = measure<std::chrono::milliseconds>::execution([&]() -> void {
			baseEvaluationSum = evaluateFeatures(v);
		});

		// now update everything that is shared across threads from the built vectors
		lock([&]() -> void {
			for (int i = 0; i < nrOfEvaluations; i++)
			{
				double valueShift = 1.0 * i / nrOfEvaluations * 20 - 10;
				swresult.evaluations[i].valueShift = valueShift;

				double evaluationResult = baseEvaluationSum + valueShift;
				double evaluationClass = evaluationResult > 0 ? 1 : -1;
				bool correct;
				if (resultClass == evaluationClass) {
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
					if (resultClass == -1 && evaluationClass == 1) {
						swresult.evaluations[i].nrOfFalsePositives++;

						//if (valueShift == valueShiftForFalsePosOrNegCollection) // TODO TMP REMOVE
						//	cv::rectangle(fullrgb, region, cv::Scalar(0, 0, 255), 1);
					}
					else
						swresult.evaluations[i].nrOfFalseNegatives++;

					correct = false;
				}

				// don't consider adding hard negatives or positives if it overlaps with a true positive
				// because it will add samples to the negative set that are really really strong and should probably be considered positive
				if (!overlapsWithTruePositive) {
					// only add worst false positives if valueShift is the value that was tested on and it was incorrect
					if (abs(valueShift - valueShiftForFalsePosOrNegCollection) < 0.0001 && !correct) {
						if (evaluationClass == 1) {

							if (abs(evaluationResult) > worstFalsePositiveValue)
								worstFalsePositiveValue = abs(evaluationResult);

							worstFalsePositives.push(std::pair<float, SlidingWindowRegion>(abs(evaluationResult), SlidingWindowRegion(imageNumber, region)));
							// smallest numbers will be popped
							if (worstFalsePositives.size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
								worstFalsePositives.pop();
						}
						else {
							worstFalseNegatives.push(std::pair<float, SlidingWindowRegion>(abs(evaluationResult), SlidingWindowRegion(imageNumber, region)));
							// smallest numbers will be popped
							if (worstFalsePositives.size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
								worstFalsePositives.pop();
						}
					}
				}
				
			}
		});
	});

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives
	for (int i = 0; i < nrOfEvaluations; i++)
		swresult.evaluations[i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;

	std::cout << "Worst false positive result: " << worstFalsePositiveValue << std::endl;
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
	//	std::cout << "Worst FN region score= " << pair.first << " image= " << pair.second.imageNumber << " bbox=" << pair.second.bbox.x << "," << pair.second.bbox.y << " " << pair.second.bbox.width << "x" << pair.second.bbox.height << std::endl;
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