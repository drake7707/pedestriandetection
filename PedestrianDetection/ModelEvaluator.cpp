#include "ModelEvaluator.h"
#include "Helper.h"

ModelEvaluator::ModelEvaluator(const std::string& baseDatasetPath, const FeatureSet& set)
	: baseDatasetPath(baseDatasetPath), set(set)
{
}


ModelEvaluator::~ModelEvaluator()
{
}




void ModelEvaluator::train()
{

	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;


	iterateDataSet(baseDatasetPath,[](int idx) -> bool { return idx % 2 != 0; },
		[&](int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth) -> void {

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
	//boost->setWeakCount(10);
	//boost->setWeightTrimRate(0.95);
	//boost->setMaxDepth(5);
	//boost->setUseSurrogates(false);

	std::cout << "Training boost classifier on " << N << " samples, feature size " << featureSize << std::endl;

	trainingTimeMS = measure<std::chrono::milliseconds>::execution([&]() -> void {
		boost->train(tdata);
	});
	std::cout << "Done training, took " << trainingTimeMS << "ms" << std::endl;

	if (!boost->isTrained())
		throw std::exception("Boost training failed");

	model.boost = boost;

	auto& roots = boost->getRoots();
	auto& nodes = boost->getNodes();
	auto& splits = boost->getSplits();
	for (auto& r : roots) {
		std::cout << "Root split " << nodes[r].split << " variable index " << splits[nodes[r].split].varIdx << " quality: " << splits[nodes[r].split].quality << " explain: " << set.explainFeature(splits[nodes[r].split].varIdx, nodes[r].value) << std::endl;
	}
}

std::vector<ClassifierEvaluation> ModelEvaluator::evaluate(int nrOfEvaluations) {
	std::vector<ClassifierEvaluation> evals(nrOfEvaluations, ClassifierEvaluation());

	std::vector<double> sumTimes(nrOfEvaluations, 0);
	std::vector<int> nrRegions(nrOfEvaluations, 0);
	double featureBuildTime = 0;

	iterateDataSet(baseDatasetPath,[](int idx) -> bool { return idx % 2 == 0; },
		[&](int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth) -> void {

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

				int result = evaluateFeatures(v, valueShift);
				if (resultClass == result) {
					if (resultClass == -1)
						evals[i].nrOfTrueNegatives++;
					else
						evals[i].nrOfTruePositives++;
				}
				else {
					if (resultClass == -1 && result == 1)
						evals[i].nrOfFalsePositives++;
					else
						evals[i].nrOfFalseNegatives++;
				}
			});
			nrRegions[i]++;
		}
	});

	for (int i = 0; i < nrOfEvaluations; i++)
		evals[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes[i]) / nrRegions[i];

	return evals;
}

int ModelEvaluator::evaluateWindow(cv::Mat& rgb, cv::Mat& depth, double valueShift) const {

	FeatureVector v = set.getFeatures(rgb, depth);
	v.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	return evaluateFeatures(v, valueShift);
}


int ModelEvaluator::evaluateFeatures(FeatureVector& v, double valueShift) const {

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

	float result = model.boost->predict(v.toMat(), cv::noArray());

	return sum + valueShift > 0 ? 1 : -1;
}


void ModelEvaluator::saveModel(std::string& path) {
	std::string filename = path;

	cv::FileStorage fs(filename + ".data.xml", cv::FileStorage::WRITE);
	fs << "mean" << model.meanVector;
	fs << "sigma" << model.sigmaVector;
	model.boost->save(filename + ".boost.xml");
	fs.release();

}

void ModelEvaluator::loadModel(std::string& path) {

	std::string filename = path;

	model.boost = cv::Algorithm::load<cv::ml::Boost>(filename + ".boost.xml");
	cv::FileStorage fsRead(filename + ".data.xml", cv::FileStorage::READ);
	fsRead["mean"] >> model.meanVector;
	fsRead["sigma"] >> model.sigmaVector;





	fsRead.release();
}