#include "ModelEvaluator.h"


ModelEvaluator::ModelEvaluator(std::string& baseDatasetPath, FeaturesSet& set) : baseDatasetPath(baseDatasetPath), set(set)
{
}


ModelEvaluator::~ModelEvaluator()
{
}



void ModelEvaluator::iterateDataSet(std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth)> func) const {
	int i = 0;
	bool stop = false;
	while (!stop) {
		if (canSelectFunc(i)) {
			std::string rgbTP = baseDatasetPath + PATH_SEPARATOR + "tp" + PATH_SEPARATOR + "rgb" + std::to_string(i) + ".png";
			std::string rgbTN = baseDatasetPath + PATH_SEPARATOR + "tn" + PATH_SEPARATOR + "rgb" + std::to_string(i) + ".png";
			std::string depthTP = baseDatasetPath + PATH_SEPARATOR + "tp" + PATH_SEPARATOR + "depth" + std::to_string(i) + ".png";
			std::string depthTN = baseDatasetPath + PATH_SEPARATOR + "tn" + PATH_SEPARATOR + "depth" + std::to_string(i) + ".png";

			cv::Mat rgb;
			cv::Mat depth;
			rgb = cv::imread(rgbTP);
			depth = cv::imread(depthTP);
			if (rgb.cols == 0 || rgb.rows == 0 || depth.cols == 0 || depth.rows == 0) {
				stop = true;
				break;
			}

			func(i, 1, rgb, depth);

			rgb = cv::imread(rgbTN);
			depth = cv::imread(depthTN);
			if (rgb.cols == 0 || rgb.rows == 0 || depth.cols == 0 || depth.rows == 0) {
				stop = true;
				break;
			}

			func(i, -1, rgb, depth);
		}
		i++;
	}
}


void ModelEvaluator::train()
{

	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;


	iterateDataSet([](int idx) -> bool { return idx % 5 != 0; },
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


	//boost->setBoostType(cv::ml::Boost::GENTLE);

	std::vector<double>	priors(2);
	priors[0] = trueNegativeFeatures.size();
	priors[1] = truePositiveFeatures.size();

	boost->setPriors(cv::Mat(priors));
	//boost->setWeakCount(10);
	//boost->setWeightTrimRate(0.95);
	//boost->setMaxDepth(5);
	//boost->setUseSurrogates(false);

	std::cout << "Training boost classifier on " << N << " samples, feature size " << featureSize << std::endl;
	boost->train(tdata);

	if (!boost->isTrained())
		throw std::exception("Boost training failed");

	model.boost = boost;
}

ClassifierEvaluation ModelEvaluator::evaluate() const {
	ClassifierEvaluation eval;

	iterateDataSet([](int idx) -> bool { return idx % 5 == 0; },
		[&](int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth) -> void {

		FeatureVector v = set.getFeatures(rgb, depth);
		v.applyMeanAndVariance(model.meanVector, model.sigmaVector);

		int result = model.boost->predict(v.toMat());
		if (resultClass == result) {
			if (resultClass == -1)
				eval.nrOfTrueNegatives++;
			else
				eval.nrOfTruePositives++;
		}
		else {
			if (resultClass == -1 && result == 1)
				eval.nrOfFalsePositives++;
			else
				eval.nrOfFalseNegatives++;
		}
	});
	return eval;
}
