#include "Detector.h"
#include "KITTIDataSet.h"
#include <fstream>
#include <string>



int sizeVariance = 4;

void Detector::iterateDataset(std::function<void(cv::Mat&)> tpFunc, std::function<void(cv::Mat&)> tnFunc, std::function<bool(int)> includeSample) {
	KITTIDataSet dataset(kittiDatasetPath);
	srand(7707);


	std::vector<DataSetLabel> labels = dataset.getLabels();

	std::string currentNumber = "";
	std::vector<cv::Mat> currentImages;

	int idx = 0;
	for (auto& l : labels) {

		if (includeSample(idx)) {
			if (currentNumber != l.getNumber()) {
				currentNumber = l.getNumber();
				currentImages = dataset.getImagesForNumber(currentNumber);
			}

			// get true positive and true negative image
			// -----------------------------------------

			cv::Mat rgbTP;
			cv::Rect2d& r = l.getBbox();
			if (r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {

				//for (auto& img : currentImages) {
				auto& img = currentImages[0];
				img(l.getBbox()).copyTo(rgbTP);
				cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

				// build training mat

				tpFunc(rgbTP);
				//truePositiveFeatures.push_back(resultTP.getFeatureArray());

				cv::Mat rgbTPFlip;
				cv::flip(rgbTP, rgbTPFlip, 1);

				tpFunc(rgbTPFlip);
			
				// take a number of true negative patches that don't overlap
				for (int k = 0; k < nrOfTN; k++)
				{
					// take an equivalent patch at random for a true negative
					cv::Mat rgbTN;
					cv::Rect2d rTN;

					int iteration = 0;
					do {
						double width = l.getBbox().width * (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
						double height = l.getBbox().height * (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
						rTN = cv::Rect2d(randBetween(0, img.cols - width), randBetween(0, img.rows - height), width, height);
					} while (iteration++ < 100 && ((rTN & l.getBbox()).area() > 0 || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= img.cols || rTN.y + rTN.height >= img.rows));


					if (iteration < 100) {
						img(rTN).copyTo(rgbTN);
						cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));
						tnFunc(rgbTN);
					}
				}
				//}
			}
		}
		idx++;
	}
}


void Detector::saveSVMLightFiles() {

	std::ofstream trainingFile("training.dat");
	if (!trainingFile.is_open())
		throw std::exception("Unable to create training file");


	iterateDataset([&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		trainingFile << "+1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		trainingFile << "#";
		trainingFile << std::endl;

	}, [&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		trainingFile << "-1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		trainingFile << "#";
		trainingFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery != 0); });


	std::ofstream testFile("test.dat");
	if (!testFile.is_open())
		throw std::exception("Unable to create training file");

	iterateDataset([&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		testFile << "+1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		testFile << "#";
		testFile << std::endl;

	}, [&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();
		testFile << "-1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		testFile << "#";
		testFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery == 0); });

}



void Detector::buildModel() {

	buildWeakHoGSVMClassifier();
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

void Detector::buildWeakHoGSVMClassifier() {

	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;
	int featureSize;

	iterateDataset([&](cv::Mat& mat) -> void {
		truePositiveFeatures.push_back(this->getFeatures(mat));
	}, [&](cv::Mat& mat) -> void {
		trueNegativeFeatures.push_back(this->getFeatures(mat));
	}, [&](int idx) -> bool { return idx % testSampleEvery != 0; });

	featureSize = truePositiveFeatures[0].size();

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
	boost->train(tdata);
	
	if (!boost->isTrained())
		throw std::exception("Boost training failed");

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




ClassifierEvaluation Detector::evaluateWeakHoGSVMClassifier(bool onTrainingSet) {


	ClassifierEvaluation evalResult;

	iterateDataset([&](cv::Mat& mat) -> void {
		// should be positive
		double svmResult = evaluate(mat);
		int svmClass = svmResult < 0 ? -1 : 1;
		if (!isnan(svmResult)) {
			if (svmClass == 1)
				evalResult.nrOfTruePositives++;
			else
				evalResult.nrOfFalsePositives++;
		}

	}, [&](cv::Mat& mat) -> void {
		// should be negative
		double svmResult = evaluate(mat);
		int svmClass = svmResult < 0 ? -1 : 1;
		if (!isnan(svmResult)) {
			if (svmClass == -1)
				evalResult.nrOfTrueNegatives++;
			else
				evalResult.nrOfFalseNegatives++;
		}
	}, [&](int idx) -> bool { return onTrainingSet ? (idx % testSampleEvery != 0) : (idx % testSampleEvery == 0); });

	return evalResult;
}

double Detector::evaluate(cv::Mat& mat) {

	FeatureVector vec = getFeatures(mat);
	//vec.applyMeanAndVariance(model.meanVector, model.sigmaVector);

	int resultClass = model.boost->predict(vec.toMat(), cv::noArray());
	double result =  model.boost->predict(vec.toMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
	//std::cout << resultClass << " " << std::fixed << result << std::endl;
	
	int svmResult = model.svm->predict(vec.toMat());
	
	if (resultClass == svmResult && resultClass == 1)
		return 1;
	else
		return -1;

		/*if (model.svm->getKernelType() == cv::ml::SVM::LINEAR)
			return -(model.weightVector.dot(vec.toMat()) - (model.bias + biasShift));
		else
			return model.svm->predict(vec.toMat());*/
}


void Detector::saveModel(std::string& path) {

	if (!modelReady)
		throw std::exception("Detector does not contain a model");
	if (!model.boost->isTrained())
		throw std::exception("Boost is not trained");

	std::string weakClassifierSVMFile = "kittitrainingsvm.xml";
	std::string weakClassifierBoostFile = "kittitrainingboost.xml";
	std::string filename = "model.xml";

	model.boost->save(weakClassifierBoostFile);
	model.svm->save(weakClassifierSVMFile);

	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "mean" << model.meanVector;
	fs << "sigma" << model.sigmaVector;
	fs.release();

}

void Detector::loadModel(std::string& path) {

	std::string weakClassifierSVMFile = "kittitrainingsvm.xml";
	std::string weakClassifierBoostFile = "kittitrainingboost.xml";
	std::string filename = "model.xml";

	model.boost = cv::Algorithm::load<cv::ml::Boost>(weakClassifierBoostFile);
	model.svm = cv::Algorithm::load<cv::ml::SVM>(weakClassifierSVMFile);
	loadSVMEvaluationParameters();
	modelReady = true;

	cv::FileStorage fsRead(filename, cv::FileStorage::READ);
	fsRead["mean"] >> model.meanVector;
	fsRead["sigma"] >> model.sigmaVector;
	fsRead.release();
}

//--------------------------------------------




FeatureVector Detector::getFeatures(cv::Mat& mat) {
	auto result = getHistogramsOfOrientedGradient(mat, patchSize, binSize, false, true);
	FeatureVector vector = result.getFeatureArray(addS2);
	return vector;
}


