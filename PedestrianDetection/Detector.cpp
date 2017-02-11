#include "Detector.h"
#include "KITTIDataSet.h"
#include <fstream>

int nrOfTN = 2;
int testSampleEvery = 5;


void Detector::iterateDataset(std::function<void(cv::Mat&, HoGResult&)> tpFunc, std::function<void(cv::Mat&, HoGResult&)> tnFunc, std::function<bool(int)> includeSample) {
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

				for (auto& img : currentImages) {
					//auto& img = currentImages[0];
					img(l.getBbox()).copyTo(rgbTP);
					cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

					// build training mat

					auto resultTP = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, false, true);
					tpFunc(rgbTP, resultTP);
					//truePositiveFeatures.push_back(resultTP.getFeatureArray());

					cv::Mat rgbTPFlip;
					cv::flip(rgbTP, rgbTPFlip, 1);

					auto resultTPFlip = getHistogramsOfOrientedGradient(rgbTPFlip, patchSize, binSize, false, true);
					tpFunc(rgbTPFlip, resultTPFlip);
					//truePositiveFeatures.push_back(resultTPFlip.getFeatureArray());

					// take a number of true negative patches that don't overlap
					for (int k = 0; k < nrOfTN; k++)
					{
						// take an equivalent patch at random for a true negative
						cv::Mat rgbTN;
						cv::Rect2d rTN;

						int iteration = 0;
						do {
							double width = l.getBbox().width * (1 + rand() * 1.0 / RAND_MAX * 4);
							double height = l.getBbox().height * (1 + rand() * 1.0 / RAND_MAX * 4);
							rTN = cv::Rect2d(randBetween(0, img.cols - width), randBetween(0, img.rows - height), width, height);
						} while (iteration++ < 100 && ((rTN & l.getBbox()).area() > 0 || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= img.cols || rTN.y + rTN.height >= img.rows));


						if (iteration < 100) {
							img(rTN).copyTo(rgbTN);
							cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));

							auto resultTN = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, false, true);
							tnFunc(rgbTN, resultTN);

							//trueNegativeFeatures.push_back(resultTN.getFeatureArray());
						}
					}
				}
			}
		}
		idx++;
	}
}


void Detector::saveSVMLightFiles() {

	std::ofstream trainingFile("training.dat");
	if (!trainingFile.is_open())
		throw std::exception("Unable to create training file");

	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		std::vector<float> featureArray = result.getFeatureArray();
		trainingFile << "+1 ";
		for (int i = 0; i < featureArray.size(); i++)
			trainingFile << (i+1) << ":" << featureArray[i] << " ";
		trainingFile << "#";
		trainingFile << std::endl;

	}, [&](cv::Mat& mat, HoGResult& result) -> void {
		std::vector<float> featureArray = result.getFeatureArray();
		trainingFile << "-1 ";
		for (int i = 0; i < featureArray.size(); i++)
			trainingFile << (i + 1) << ":" << featureArray[i] << " ";
		trainingFile << "#";
		trainingFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery != 0); });


	std::ofstream testFile("test.dat");
	if (!testFile.is_open())
		throw std::exception("Unable to create training file");

	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		std::vector<float> featureArray = result.getFeatureArray();
		testFile << "+1 ";
		for (int i = 0; i < featureArray.size(); i++)
			testFile << (i + 1) << ":" << featureArray[i] << " ";
		testFile << "#";
		testFile << std::endl;

	}, [&](cv::Mat& mat, HoGResult& result) -> void {
		std::vector<float> featureArray = result.getFeatureArray();
		testFile << "-1 ";
		for (int i = 0; i < featureArray.size(); i++)
			testFile << (i + 1) << ":" << featureArray[i] << " ";
		testFile << "#";
		testFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery == 0); });

}

cv::Ptr<cv::ml::SVM> Detector::buildWeakHoGSVMClassifier() {

	std::vector<std::vector<float>> truePositiveFeatures;
	std::vector<std::vector<float>> trueNegativeFeatures;
	int featureSize;

	std::cout << "Iterating dataset with test sample every " << testSampleEvery << " samples" << std::endl;
	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		truePositiveFeatures.push_back(result.getFeatureArray());
	}, [&](cv::Mat& mat, HoGResult& result) -> void {
		trueNegativeFeatures.push_back(result.getFeatureArray());
	}, [](int idx) -> bool { return idx % testSampleEvery != 0; });

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
	svm->setKernel(cv::ml::SVM::POLY);
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

	svm->save(weakClassifierSVMFile);

	return svm;
}



ClassifierEvaluation Detector::evaluateWeakHoGSVMClassifier(bool onTrainingSet) {
	cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(weakClassifierSVMFile);

	ClassifierEvaluation evalResult;

	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		// should be positive
		int svmClass = svm->predict(result.getFeatureMat(), cv::noArray());
		float svmResult = svm->predict(result.getFeatureMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
		if (!isnan(svmResult)) {
			if (svmClass == 1)
				evalResult.nrOfTruePositives++;
			else
				evalResult.nrOfFalsePositives++;
		}

	}, [&](cv::Mat& mat, HoGResult& result) -> void {

		// should be negative
		int svmClass = svm->predict(result.getFeatureMat(), cv::noArray());
		float svmResult = svm->predict(result.getFeatureMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
		if (!isnan(svmResult)) {
			if (svmClass == -1)
				evalResult.nrOfTrueNegatives++;
			else
				evalResult.nrOfFalseNegatives++;
		}
	}, [&](int idx) -> bool { return onTrainingSet ? (idx % testSampleEvery != 0) : (idx % testSampleEvery == 0); });

	return evalResult;
}
