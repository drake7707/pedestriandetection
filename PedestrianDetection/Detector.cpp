#include "Detector.h"
#include "KITTIDataSet.h"


int nrOfTN = 2;
int testSampleEvery = 500000;


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
				currentImages[0](l.getBbox()).copyTo(rgbTP);
				cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

				// build training mat

				auto resultTP = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, false);
				tpFunc(rgbTP, resultTP);
				//truePositiveFeatures.push_back(resultTP.getFeatureArray());

				cv::Mat rgbTPFlip;
				cv::flip(rgbTP, rgbTPFlip, 1);

				auto resultTPFlip = getHistogramsOfOrientedGradient(rgbTPFlip, patchSize, binSize, false);
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
						rTN = cv::Rect2d(randBetween(0, currentImages[0].cols - l.getBbox().width), randBetween(0, currentImages[0].rows - l.getBbox().height), l.getBbox().width, l.getBbox().height);
					} while ((rTN & l.getBbox()).area() > 0 && iteration++ < 100);

					if (iteration < 100) {
						currentImages[0](rTN).copyTo(rgbTN);
						cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));

						auto resultTN = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, false);
						tnFunc(rgbTN, resultTN);
						//trueNegativeFeatures.push_back(resultTN.getFeatureArray());
					}
				}
			}
		}
		idx++;
	}
}
cv::Ptr<cv::ml::SVM> Detector::buildWeakHoGSVMClassifier() {

	std::vector<std::vector<float>> truePositiveFeatures;
	std::vector<std::vector<float>> trueNegativeFeatures;
	int featureSize;


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
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setC(0.001);
	//svm->setDegree(2);

	//cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingMat, cv::ml::SampleTypes::ROW_SAMPLE, trainingLabels);
	//svm->trainAuto(tdata);

	//cv::Mat classWeights(2, 1, CV_32FC1);
	//// each TP has 2, 1 normal , 1 flipped
	//classWeights.at<float>(0, 0) = 2 - 2.0 / nrOfTN;
	//classWeights.at<float>(1, 0) = 2.0/nrOfTN;
	//svm->setClassWeights(classWeights);
	std::cout << "Training SVM" << std::endl;
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
