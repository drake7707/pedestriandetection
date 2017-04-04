#include "ModelEvaluator.h"
#include "Helper.h"
#include <queue>

ModelEvaluator::ModelEvaluator(std::string& name)
	: IEvaluator(name)
{
}


ModelEvaluator::~ModelEvaluator()
{
}




void ModelEvaluator::train(const TrainingDataSet& trainingDataSet, const FeatureSet& set, const EvaluationSettings& settings, std::function<bool(int number)> canSelectFunc)
{

	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;

	// don't use prepared data for training
	 std::vector<IPreparedData*> preparedData;

	trainingDataSet.iterateDataSet(canSelectFunc,
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Building feature vectors (") + std::to_string(imageNumber) + ")");

		FeatureVector v = set.getFeatures(rgb, depth, thermal, region, preparedData);
		if (resultClass == 1)
			truePositiveFeatures.push_back(v);
		else
			trueNegativeFeatures.push_back(v);
	}, settings.addFlippedInTrainingSet, settings.refWidth, settings.refHeight);

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
	cv::Mat trainingMat(N, featureSize, CV_32FC1);
	cv::Mat trainingLabels(N, 1, CV_32SC1);

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

	boost->setBoostType(cv::ml::Boost::Types::REAL);
	boost->setPriors(cv::Mat(priors));
	boost->setWeakCount(settings.maxWeakClassifiers);
	//boost->setMaxDepth(3);

	ProgressWindow::getInstance()->updateStatus(name, 0, std::string("Training boost classifier"));

	std::cout << "Training boost classifier on " << N << " samples (" << truePositiveFeatures.size() << " TP, " << trueNegativeFeatures.size() << " TN), feature size " << featureSize << std::endl;

	double trainingTimeMS = measure<std::chrono::milliseconds>::execution([&]() -> void {
		boost->train(tdata);
	});
	//std::cout << "Done training, took " << trainingTimeMS << "ms" << std::endl;

	if (!boost->isTrained())
		throw std::exception("Boost training failed");

	model.boost = boost;

	ProgressWindow::getInstance()->finish(name);
	//auto& roots = boost->getRoots();
	//auto& nodes = boost->getNodes();
	//auto& splits = boost->getSplits();
	//for (auto& r : roots) {
	//	std::cout << "Root split " << nodes[r].split << " variable index " << splits[nodes[r].split].varIdx << " quality: " << splits[nodes[r].split].quality << " explain: " << set.explainFeature(splits[nodes[r].split].varIdx, nodes[r].value) << std::endl;
	//}
}


double ModelEvaluator::evaluateFeatures(FeatureVector& v) {

	FeatureVector featureVector = v; // copy
	featureVector.applyMeanAndVariance(model.meanVector, model.sigmaVector);

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



std::vector<cv::Mat> ModelEvaluator::explainModel(const std::unique_ptr<FeatureSet>& set, int refWidth, int refHeight) const {
	auto& roots = model.boost->getRoots();
	auto& nodes = model.boost->getNodes();
	auto& splits = model.boost->getSplits();

	std::vector<float> weightPerFeature(set->getNumberOfFeatures(), 0);
	std::vector<float> occurrencePerFeature(set->getNumberOfFeatures(), 0);
	for (auto& r : roots) {

		if (nodes[r].split != -1) {
			int varIdx = splits[nodes[r].split].varIdx;
			float quality = splits[nodes[r].split].quality;
			weightPerFeature[varIdx] += quality;
			occurrencePerFeature[varIdx]++;
		}
	}

	return set->explainFeatures(weightPerFeature, occurrencePerFeature, refWidth, refHeight);
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

std::string ModelEvaluator::getName() const {
	return name;
}