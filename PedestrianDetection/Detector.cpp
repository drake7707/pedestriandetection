#include "Detector.h"
#include "KITTIDataSet.h"
#include <fstream>
#include <string>
#include "HistogramOfOrientedGradients.h"
#include "FeatureVector.h"




void Detector::buildModel(std::vector<FeatureVector>& truePositiveFeatures, std::vector<FeatureVector>& trueNegativeFeatures) {

	buildWeakHoGSVMClassifier(truePositiveFeatures, trueNegativeFeatures);
	loadSVMEvaluationParameters();
	modelReady = true;
}

void Detector::loadSVMEvaluationParameters() {
	cv::Mat sv = model.svm->getSupportVectors();
	std::vector<float> alpha;
	std::vector<float> svidx;
	model.bias = model.svm->getDecisionFunction(0, alpha, svidx);
	model.weightVector = cv::Mat(1, sv.cols, CV_32F, cv::Scalar(0));
	for (int r = 0; r < sv.rows; ++r)
	{
		for (int c = 0; c < sv.cols; ++c)
			model.weightVector.at<float>(0, c) += alpha[r] * sv.at<float>(r, c);
	}
}


void Detector::buildWeakHoGSVMClassifier(std::vector<FeatureVector>& truePositiveFeatures, std::vector<FeatureVector>& trueNegativeFeatures) {

	int featureSize = truePositiveFeatures[0].size();

	// + 1 for the responses
	cv::Mat trainingMat(truePositiveFeatures.size() + trueNegativeFeatures.size(), featureSize, CV_32FC1);
	cv::Mat trainingLabels(truePositiveFeatures.size() + trueNegativeFeatures.size(), 1, CV_32SC1);

	int idx = 0;
	for (int i = 0; i < truePositiveFeatures.size(); i++)
	{
		for (int f = 0; f < featureSize; f++) {

			if (isnan(truePositiveFeatures[i][f]))
				throw std::exception("HoG feature contains NaN");

			trainingMat.at<float>(idx, f) = truePositiveFeatures[i][f];

		}
		trainingLabels.at<int>(idx, 0) = 1;
		idx++;
	}

	for (int i = 0; i < trueNegativeFeatures.size(); i++) {
		for (int f = 0; f < featureSize; f++) {
			if (isnan(trueNegativeFeatures[i][f]))
				throw std::exception("HoG feature contains NaN");

			trainingMat.at<float>(idx, f) = trueNegativeFeatures[i][f];
		}
		trainingLabels.at<int>(idx, 0) = -1;
		idx++;
	}

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

	// now correct the feature arrays
	//for (auto& featureVector : truePositiveFeatures)
	//	featureVector.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	//for (auto& featureVector : trueNegativeFeatures)
	//	featureVector.applyMeanAndVariance(model.meanVector, model.sigmaVector);


	/*cv::Mat var_type(1, featureSize, CV_8U);
	var_type.setTo(cv::Scalar::all(cv::ml::VariableTypes::VAR_ORDERED));
	var_type.at<uchar>(featureSize) = var_type.at<uchar>(featureSize + 1) = cv::ml::VariableTypes::VAR_CATEGORICAL;
*/

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
	/*boost->train(tdata);

	if (!boost->isTrained())
		throw std::exception("Boost training failed");*/

	model.boost = boost;

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setC(0.001);
	svm->setDegree(2);

	//cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingMat, cv::ml::SampleTypes::ROW_SAMPLE, trainingLabels);
	//svm->trainAuto(tdata);
	auto& criteria = svm->getTermCriteria();
	//criteria.maxCount = 4000;

	cv::Mat classWeights(2, 1, CV_32FC1);

	classWeights.at<float>(0, 0) = 0.5;
	classWeights.at<float>(1, 0) = 0.5;
	svm->setClassWeights(classWeights);


	std::cout << "Training SVM with options: " << std::endl;
	std::cout << "Type: " << svm->getType() << std::endl;
	std::cout << "Kernel: " << svm->getKernelType() << std::endl;
	std::cout << "C: " << svm->getC() << std::endl;
	std::cout << "Iterations: " << criteria.maxCount << std::endl;

	if (svm->getKernelType() == cv::ml::SVM::POLY)
		std::cout << "Degree: " << svm->getDegree() << std::endl;
	if (svm->getClassWeights().rows > 0)
		std::cout << "Class weights: [" << svm->getClassWeights().at<float>(0, 0) << "," << svm->getClassWeights().at<float>(1, 0) << "]" << std::endl;

	std::cout << "Number of features per sample: " << featureSize << std::endl;
	std::cout << "Number of training samples: " << trainingMat.rows << std::endl;
	std::cout << "Number of true positive samples: " << truePositiveFeatures.size() << std::endl;
	std::cout << "Number of true negative samples: " << trueNegativeFeatures.size() << std::endl;


	bool success = svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);

	if (!success || !svm->isTrained())
		throw std::exception("SVM training failed");

	model.svm = svm;
}


double Detector::evaluate(FeatureVector& vec) const {
	//vec.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	/*int resultClass = model.boost->predict(vec.toMat(), cv::noArray());
	double result =  model.boost->predict(vec.toMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
	*///std::cout << resultClass << " " << std::fixed << result << std::endl;

	//	int svmResult = model.svm->predict(vec.toMat());
		//return svmResult;
		//if (resultClass == svmResult && resultClass == 1)
		//	return 1;
		//else
		//	return -1;

	if (model.svm->getKernelType() == cv::ml::SVM::LINEAR)
		return -(model.weightVector.dot(vec.toMat()) - (model.bias + biasShift));
	else
		return model.svm->predict(vec.toMat());
}


void Detector::saveModel(std::string& path) {

	if (!modelReady)
		throw std::exception("Detector does not contain a model");
	//if (!model.boost->isTrained())
	//	throw std::exception("Boost is not trained");

	std::string weakClassifierSVMFile = path + PATH_SEPARATOR + "kittitrainingsvm.xml";
	std::string weakClassifierBoostFile = path + PATH_SEPARATOR + "kittitrainingboost.xml";
	std::string filename = path + PATH_SEPARATOR + "model.xml";

	//model.boost->save(weakClassifierBoostFile);
	model.svm->save(weakClassifierSVMFile);

	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "mean" << model.meanVector;
	fs << "sigma" << model.sigmaVector;
	fs.release();

}

void Detector::loadModel(std::string& path) {

	std::string weakClassifierSVMFile = path + PATH_SEPARATOR + "kittitrainingsvm.xml";
	std::string weakClassifierBoostFile = path + PATH_SEPARATOR + "kittitrainingboost.xml";
	std::string filename = path + PATH_SEPARATOR + "model.xml";

	//model.boost = cv::Algorithm::load<cv::ml::Boost>(weakClassifierBoostFile);
	model.svm = cv::Algorithm::load<cv::ml::SVM>(weakClassifierSVMFile);
	loadSVMEvaluationParameters();
	modelReady = true;

	cv::FileStorage fsRead(filename, cv::FileStorage::READ);
	fsRead["mean"] >> model.meanVector;
	fsRead["sigma"] >> model.sigmaVector;
	fsRead.release();
}

//--------------------------------------------







