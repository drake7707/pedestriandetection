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
					//truePositiveFeatures.push_back(resultTPFlip.getFeatureArray());

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

							//trueNegativeFeatures.push_back(resultTN.getFeatureArray());
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
			trainingFile << (i+1) << ":" << features.at<float>(0,i) << " ";
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

cv::Ptr<cv::ml::SVM> Detector::buildWeakHoGSVMClassifier() {

	std::vector<std::vector<float>> truePositiveFeatures;
	std::vector<std::vector<float>> trueNegativeFeatures;
	int featureSize;

	iterateDataset([&](cv::Mat& mat) -> void {
		truePositiveFeatures.push_back(this->getFeatures(mat));
	}, [&](cv::Mat& mat) -> void {
		trueNegativeFeatures.push_back(this->getFeatures(mat));
	}, [&](int idx) -> bool { return idx % testSampleEvery != 0; });

	featureSize = truePositiveFeatures[0].size();

	cv::Mat trainingMat(truePositiveFeatures.size() + trueNegativeFeatures.size(), featureSize, CV_32FC1);
	cv::Mat trainingLabels(truePositiveFeatures.size() + trueNegativeFeatures.size(), 1, CV_32SC1);

	int idx = 0;
	for (int i = 0; i < truePositiveFeatures.size(); i++)
	{
		for (int f = 0; f < trainingMat.cols; f++) {

			if (isnan(truePositiveFeatures[i][f]))
				throw std::exception("HoG feature contains NaN");

			trainingMat.at<float>(idx, f) = truePositiveFeatures[i][f];

		}
		trainingLabels.at<int>(idx, 0) = 1;
		idx++;
	}

	for (int i = 0; i < trueNegativeFeatures.size(); i++) {
		for (int f = 0; f < trainingMat.cols; f++) {

			if (isnan(trueNegativeFeatures[i][f]))
				throw std::exception("HoG feature contains NaN");

			trainingMat.at<float>(idx, f) = trueNegativeFeatures[i][f];
		}

		trainingLabels.at<int>(idx, 0) = -1;
		idx++;
	}


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
	std::cout << "Evaluating with bias (rho) shift: " << biasShift << std::endl;

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

	svm->save(weakClassifierSVMFile);

	return svm;
}

ClassifierEvaluation Detector::evaluateWeakHoGSVMClassifier(bool onTrainingSet) {
	cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(weakClassifierSVMFile);
	
	cv::Mat sv = svm->getSupportVectors();
	std::vector<float> alpha;
	std::vector<float> svidx;
	double b = svm->getDecisionFunction(0, alpha, svidx);
	cv::Mat wT(1, sv.cols, CV_32F, cv::Scalar(0));
	for (int r = 0; r < sv.rows; ++r)
	{
		for (int c = 0; c < sv.cols; ++c)
			wT.at<float>(0, c) += alpha[r] * sv.at<float>(r, c);
	}

	ClassifierEvaluation evalResult;

	iterateDataset([&](cv::Mat& mat) -> void {
		// should be positive
		cv::Mat features = this->getFeatures(mat).toMat();
		double svmResult = evaluateSVM(svm, wT, b, this->getFeatures(mat));
		int svmClass = svmResult < 0 ? -1 : 1;
		if (!isnan(svmResult)) {
			if (svmClass == 1)
				evalResult.nrOfTruePositives++;
			else
				evalResult.nrOfFalsePositives++;
		}

	}, [&](cv::Mat& mat) -> void {
		// should be negative
		
		double svmResult = evaluateSVM(svm, wT, b, this->getFeatures(mat));
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

double Detector::evaluateSVM(cv::Ptr<cv::ml::SVM> svm, cv::Mat& wT, double b, FeatureVector& vec) {
	if (svm->getKernelType() == cv::ml::SVM::LINEAR)
		return -(wT.dot(vec) - (b+biasShift));
	else
		return svm->predict(vec.toMat());
}


void Detector::saveModel() {

}

void Detector::loadModel(std::string& path) {

}

//--------------------------------------------




FeatureVector Detector::getFeatures(cv::Mat& mat) {
	auto result = getHistogramsOfOrientedGradient(mat, patchSize, binSize, false, true);
	FeatureVector vector= result.getFeatureArray(addS2);
	//vector.normalize();
	return vector;
}


