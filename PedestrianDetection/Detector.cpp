#include "Detector.h"
#include "KITTIDataSet.h"





void Detector::iterateDataset(std::function<void(cv::Mat&, HoGResult&)> tpFunc, std::function<void(cv::Mat&, HoGResult&)> tnFunc)  {
	KITTIDataSet dataset(kittiDatasetPath);
	int nrOfTN = 2;
	srand(7707);


	std::vector<DataSetLabel> labels = dataset.getLabels();

	std::string currentNumber = "";
	std::vector<cv::Mat> currentImages;

	for (auto& l : labels) {

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
			
			auto resultTP = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, true);
			tpFunc(rgbTP, resultTP);
			//truePositiveFeatures.push_back(resultTP.getFeatureArray());

			cv::Mat rgbTPFlip;
			cv::flip(rgbTP, rgbTPFlip, 1);

			auto resultTPFlip = getHistogramsOfOrientedGradient(rgbTPFlip, patchSize, binSize, true);
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
}
cv::Ptr<cv::ml::SVM> Detector::buildWeakHoGSVMClassifier() {

	int nrOfTN = 2;

	std::vector<std::vector<float>> truePositiveFeatures;
	std::vector<std::vector<float>> trueNegativeFeatures;
	int featureSize;


	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		truePositiveFeatures.push_back(result.getFeatureArray());
	}, [&](cv::Mat& mat, HoGResult& result) -> void {
		trueNegativeFeatures.push_back(result.getFeatureArray());
	});

	featureSize = truePositiveFeatures[0].size();

	cv::Mat trainingMat(truePositiveFeatures.size() + trueNegativeFeatures.size(), featureSize, CV_32FC1);
	cv::Mat trainingLabels(truePositiveFeatures.size() + trueNegativeFeatures.size(), 1, CV_32SC1);



	int idx = 0;
	for (int i = 0; i < truePositiveFeatures.size(); i++)
	{
		for (int f = 0; f < trainingMat.cols; f++)
			trainingMat.at<float>(idx, f) = truePositiveFeatures[i][f];
		trainingLabels.at<int>(idx, 0) = 1;
		idx++;
	}

	for (int i = 0; i < trueNegativeFeatures.size(); i++) {
		for (int f = 0; f < trainingMat.cols; f++)
			trainingMat.at<float>(idx, f) = trueNegativeFeatures[i][f];
		trainingLabels.at<int>(idx, 0) = -1;
		idx++;
	}


	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setC(0.01);
	svm->setDegree(2);

	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);

	svm->save(weakClassifierSVMFile);

	return svm;
}



ClassifierEvaluation Detector::evaluateWeakHoGSVMClassifier() {
	cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(weakClassifierSVMFile);

	ClassifierEvaluation evalResult;

	iterateDataset([&](cv::Mat& mat, HoGResult& result) -> void {
		// should be positive
		float svmResult = svm->predict(result.getFeatureMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);

		if (svmResult > 0)
			evalResult.nrOfTruePositives++;
		else
			evalResult.nrOfFalsePositives++;

	}, [&](cv::Mat& mat, HoGResult& result) -> void {

		// should be negative
		float svmResult = svm->predict(result.getFeatureMat(), cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);

		if (svmResult < 0)
			evalResult.nrOfTrueNegatives++;
		else
			evalResult.nrOfFalseNegatives++;
	});

	return evalResult;
}
