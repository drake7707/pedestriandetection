// PedestrianDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

#include "KITTIDataSet.h"
#include "HistogramOfOrientedGradients.h"
#include "Helper.h"
#include "Detector.h"

#define __cplusplus


using namespace cv;

int refWidth = 64;
int refHeight = 128;

int patchSize = 8;
int binSize = 9;




void dumpTrainingMat(cv::Mat& trainingMat) {

	cv::Mat dmp = trainingMat.clone();
	double min;
	double max;
	minMaxLoc(dmp, &min, &max);
	float range = (float)(max - min);
	for (int j = 0; j < dmp.rows; j++)
	{
		for (int i = 0; i < dmp.cols; i++)
			dmp.at<float>(j, i) = (dmp.at<float>(j, i) - (float)min) / range * 255;
	}
	dmp.convertTo(dmp, CV_8U);

	imwrite("D:\\test.png", dmp);
}

//cv::Ptr<ml::SVM> buildModel() {
//
//	int nrOfTN = 3;
//
//	std::vector<std::vector<float>> truePositiveFeatures;
//	std::vector<std::vector<float>> trueNegativeFeatures;
//	int featureSize;
//
//	KITTIDataSet dataset("D:\\PedestrianDetectionDatasets\\kitti");
//
//	std::vector<DataSetLabel> labels = dataset.getLabels();
//
//	std::string currentNumber = "";
//	std::vector<cv::Mat> currentImages;
//
//	//namedWindow("HoGTest");
//
//	for (auto& l : labels) {
//
//		if (currentNumber != l.getNumber()) {
//			currentNumber = l.getNumber();
//			currentImages = dataset.getImagesForNumber(currentNumber);
//		}
//
//		// get true positive and true negative image
//		// -----------------------------------------
//		cv::Mat rgbTP;
//		cv::Rect2d& r = l.getBbox();
//		if (r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {
//			currentImages[0](l.getBbox()).copyTo(rgbTP);
//			cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));
//
//			// take an equivalent patch at random for a true negative
//			cv::Mat rgbTN;
//			cv::Rect2d rTN;
//
//			// build training mat
//
//			auto resultTP = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, true);
//			truePositiveFeatures.push_back(resultTP.getFeatureArray(true));
//
//			/*			cv::imshow("Original", rgbTP);
//						cv::imshow("HoGTest", resultTP.hogImage);
//						setMouseCallback("HoGTest", [](int event, int x, int y, int flags, void* userdata) -> void {
//							HoGResult* r = (HoGResult*)userdata;
//
//							int cx = x / patchSize;
//							int cy = y / patchSize;
//							if (cx >= 0 && cy >= 0 && cx < r->width && cy < r->height)
//								showHistogram(r->data[cy][cx]);
//						}, &resultTP);
//						waitKey(0);
//						*/
//
//
//
//			for (int k = 0; k < nrOfTN; k++)
//			{
//				int iteration = 0;
//				do {
//					rTN = cv::Rect2d(randBetween(0, currentImages[0].cols - l.getBbox().width), randBetween(0, currentImages[0].rows - l.getBbox().height), l.getBbox().width, l.getBbox().height);
//				} while ((rTN & l.getBbox()).area() > 0 && iteration++ < 100);
//
//				currentImages[0](rTN).copyTo(rgbTN);
//				cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));
//
//				auto resultTN = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, false);
//				trueNegativeFeatures.push_back(resultTN.getFeatureArray(true));
//			}
//
//			if (truePositiveFeatures.size() % 10 == 0)
//				std::cout << l.getNumber() << "   " << l.getBbox().x << "," << l.getBbox().y << " / " << l.getBbox().width << "x" << l.getBbox().height << std::endl;
//		}
//	}
//
//	featureSize = truePositiveFeatures[0].size();
//
//	cv::Mat trainingMat(truePositiveFeatures.size() + trueNegativeFeatures.size(), featureSize, CV_32FC1);
//	cv::Mat trainingLabels(truePositiveFeatures.size() + trueNegativeFeatures.size(), 1, CV_32SC1);
//
//	int idx = 0;
//	for (int i = 0; i < truePositiveFeatures.size(); i++)
//	{
//		for (int f = 0; f < trainingMat.cols; f++)
//			trainingMat.at<float>(idx, f) = truePositiveFeatures[i][f];
//		trainingLabels.at<int>(idx, 0) = 1;
//		idx++;
//	}
//
//	for (int i = 0; i < trueNegativeFeatures.size(); i++) {
//		for (int f = 0; f < trainingMat.cols; f++)
//			trainingMat.at<float>(idx, f) = trueNegativeFeatures[i][f];
//		trainingLabels.at<int>(idx, 0) = -1;
//		idx++;
//	}
//
//
//	Ptr<ml::SVM> svm = ml::SVM::create();
//	svm->setType(cv::ml::SVM::C_SVC);
//	svm->setKernel(cv::ml::SVM::POLY);
//	svm->setC(0.01);
//	svm->setDegree(2);
//
//	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);
//
//
//
//	svm->save("kittitraining.xml");
//
//	return svm;
//}


void testSVM() {

	// if the pattern is 010101010 then 1 (anything above 0.5 is seen as 1)
	// if the pattern is anything else then -1
	int featureSize = 20;
	int trainingSize = 1000;
	int testIterations = 1000;

	cv::Mat trainingMat(trainingSize, featureSize, CV_32FC1);
	cv::Mat trainingLabels(trainingSize, 1, CV_32SC1);

	for (int i = 0; i < trainingSize; i++)
	{

		bool isTrue = true;
		for (int f = 0; f < featureSize; f++)
		{
			double val = (rand() / (double)RAND_MAX);
			trainingMat.at<float>(i, f) = (float)val;
			isTrue = isTrue && ((f % 2 == 0 && val < 0.5) ||
				(f % 2 == 1 && val >= 0.5));
		}
		trainingLabels.at<int>(i, 0) = isTrue ? 1 : -1;
	}

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setC(0.01);
	svm->setDegree(2);

	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);

	int nrOK = 0;
	int nrNOK = 0;

	for (int i = 0; i < testIterations; i++)
	{
		cv::Mat testMat(1, featureSize, CV_32F);
		bool isTrue = true;
		for (int f = 0; f < featureSize; f++)
		{
			double val = (rand() / (double)RAND_MAX);
			testMat.at<float>(0, f) = (float)val;
			isTrue = isTrue && ((f % 2 == 0 && val < 0.5) ||
				(f % 2 == 1 && val >= 0.5));
		}

		float result = svm->predict(testMat, noArray(), ml::StatModel::Flags::RAW_OUTPUT);
		if ((result > 0 && isTrue)
			|| (result < 0 && !isTrue)) {
			// OK
			nrOK++;
		}
		else {
			nrNOK++;
		}
	}
	std::cout << "nr OK " << nrOK << " , " << " nr NOK " << nrNOK << std::endl;
}




void testSVMEvaluation() {


	int featureSize = 3;
	int trainingSize = 1000;
	int testIterations = 1000;

	cv::Mat trainingMat(trainingSize, featureSize, CV_32FC1);
	cv::Mat trainingLabels(trainingSize, 1, CV_32SC1);

	for (int i = 0; i < trainingSize; i++)
	{
		double val0 = (rand() / (double)RAND_MAX);
		double val1 = (rand() / (double)RAND_MAX);
		double val2 = (rand() / (double)RAND_MAX);

		trainingMat.at<float>(i, 0) = (float)val0;
		trainingMat.at<float>(i, 1) = (float)val1;
		trainingMat.at<float>(i, 2) = (float)val2;

		bool isTrue = val0 > val1 && val1 > val2;
		trainingLabels.at<float>(i, 0) = isTrue ? 1 : -1;
	}

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setC(1);
	//svm->setDegree(2);

	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);

	int nrOK = 0;
	int nrNOK = 0;

	int nrOK2 = 0;
	int nrNOK2 = 0;

	cv::Mat sv = svm->getSupportVectors();

	std::cout << "SV" << std::endl;
	for (int i = 0; i < sv.cols; i++)
	{
		std::cout << std::fixed << std::setprecision(2) << sv.at<float>(0, i) << " " << std::endl;
	}


	std::cout << "w^T" << std::endl;
	std::vector<float> alpha;
	std::vector<float> svidx;
	double b = svm->getDecisionFunction(0, alpha, svidx);
	Mat wT(1, sv.cols, CV_32F, Scalar(0));
	for (int r = 0; r < sv.rows; ++r)
	{
		for (int c = 0; c < sv.cols; ++c)
			wT.at<float>(0, c) += alpha[r] * sv.at<float>(r, c);
	}
//	cv::Mat wT;
//	cv::transpose(W, wT);

	for (int i = 0; i < wT.cols; i++)
	{
		std::cout << std::fixed << std::setprecision(2) << wT.at<float>(0, i) << " " << std::endl;
	}


	cv::Mat testImage(100, 100, CV_8UC3);

	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 100; j++)
		{


			cv::Mat testMat(1, featureSize, CV_32F);


			double val0 = i / 100.0;
			double val1 = j / 100.0;
			double val2 = (rand() / (double)RAND_MAX);

			testMat.at<float>(0, 0) = (float)val0;
			testMat.at<float>(0, 1) = (float)val1;
			testMat.at<float>(0, 2) = (float)val2;
			bool isTrue = val0 > val1 && val1 > val2;
			//	svm->save("test.xml");

			double value = -(wT.dot(testMat) - b);



			if ((value > 0 && isTrue)
				|| (value < 0 && !isTrue)) {
				// OK
				nrOK2++;
			//	testImage.at<cv::Vec3b>(99 - j, i) = Vec3b(0, 255, 0);

			}
			else {
				nrNOK2++;
				//testImage.at<cv::Vec3b>(99 - j, i) = Vec3b(0, 0, 255);
			}

			float result = svm->predict(testMat, noArray());
			if ((result > 0 && isTrue)
				|| (result < 0 && !isTrue)) {
				// OK
				nrOK++;
			}
			else {
				nrNOK++;
			}

			testImage.at<cv::Vec3b>(99 - j, i) = value > 0 ? Vec3b(0, 255, 0) : Vec3b(0, 0, 255);

		}
	}
	namedWindow("TestSVM");
	imshow("TestSVM", testImage);
	std::cout << "nr OK " << nrOK << " , " << " nr NOK " << nrNOK << std::endl;
	std::cout << "nr OK2 " << nrOK2 << " , " << " nr NOK2 " << nrNOK2 << std::endl;
	waitKey(0);
}





struct MatchRegion {
	cv::Rect2d region;
	float svmResult;
};

std::vector<MatchRegion> evaluateModelOnImage(cv::Mat& img, Detector& d, double threshold) {

	int slidingWindowWidth = 32;
	int slidingWindowHeight = 64;
	int slidingWindowStep = 8;
	int maxScaleReduction = 6; // 4 times reduction


	//namedWindow("Test2");
	std::vector<MatchRegion> matchingRegions;

	for (float invscale = 1; invscale <= maxScaleReduction; invscale += 0.3)
	{
		cv::Mat imgToTest;
		cv::resize(img, imgToTest, cv::Size2d(ceilTo(img.cols / invscale, slidingWindowWidth), ceilTo(img.rows / invscale, slidingWindowHeight)));

		for (int j = 0; j < imgToTest.rows / slidingWindowHeight - 1; j++)
		{
			for (float verticalStep = 0; verticalStep < slidingWindowWidth; verticalStep += slidingWindowStep/invscale)
			{
				for (int i = 0; i < imgToTest.cols / slidingWindowWidth - 1; i++)
				{

					for (float horizontalStep = 0; horizontalStep < slidingWindowWidth; horizontalStep += slidingWindowStep/invscale)
					{
						cv::Rect windowRect(i * slidingWindowWidth + horizontalStep, j * slidingWindowHeight + verticalStep, slidingWindowWidth, slidingWindowHeight);
						cv::Mat region;
						cv::resize(imgToTest(windowRect), region, Size2d(refWidth, refHeight));


						float svmResult = d.evaluate(region);
						int svmClass = svmResult < 0 ? -1 : 1;
						if (svmClass == 1 && abs(svmResult) > threshold) {

							double scaleX = img.cols / (double)imgToTest.cols;
							double scaleY = img.rows / (double)imgToTest.rows;

							MatchRegion mr;
							mr.region = Rect2d(windowRect.x * scaleX, windowRect.y * scaleY, windowRect.width * scaleX, windowRect.height * scaleY);
							mr.svmResult = abs(svmResult);
							matchingRegions.push_back(mr);
						}

						//cv::Mat tmp = imgToTest.clone();
						//cv::rectangle(tmp, windowRect, Scalar(0, 255, 0), 2);
						//cv::imshow("Test2", tmp);
						//waitKey(0);
						//std::cout << "TN result: " << tnResult << std::endl;


					}
				}
			}
		}
	}
	return matchingRegions;
}


void testDetection(Detector& d, double threshold) {
	//Ptr<ml::SVM> svm;
	//svm = buildModel();

	//svm = Algorithm::load<ml::SVM>("kittitraining.xml");

	KITTIDataSet dataset("C:\\Users\\dwight\\Downloads\\dwight\\kitti");

	std::vector<DataSetLabel> labels = dataset.getLabels();

	std::string currentNumber = "";
	std::vector<cv::Mat> currentImages;

	namedWindow("RGB");
	namedWindow("DEPTH");

	namedWindow("ElementTP");
	namedWindow("ElementResultTP");
	namedWindow("ElementTN");
	namedWindow("ElementResultTN");

	for (auto& l : labels) {

		if (currentNumber != l.getNumber()) {
			currentNumber = l.getNumber();
			currentImages = dataset.getImagesForNumber(currentNumber);
		}
		else
			continue;

		cv::Mat rgb = currentImages[0].clone();
		cv::Mat depth = currentImages[1].clone();
		//	cv::rectangle(rgb, l.getBbox(), Scalar(0, 0, 255), 2);
		//	cv::rectangle(depth, l.getBbox(), Scalar(0, 0, 255), 2);

		std::vector<MatchRegion> matchingRegions = evaluateModelOnImage(currentImages[0], d, threshold);

		for (auto& region : matchingRegions) {
			cv::rectangle(rgb, region.region, Scalar(0, 255, 0), 1);
		}
		std::cout << "Found " << matchingRegions.size() << " nr of matching regions " << std::endl;


		imshow("RGB", rgb);
		//	imshow("DEPTH", depth);

		/*cv::Mat rgbTP;
		currentImages[0](l.getBbox()).copyTo(rgbTP);
		cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));*/


		//// take an equivalent patch at random for a true negative
		//cv::Mat rgbTN;
		//cv::Rect2d rTN;

		//int iteration = 0;
		//do {
		//	rTN = cv::Rect2d(randBetween(0, rgb.cols - l.getBbox().x), randBetween(0, rgb.rows - l.getBbox().y), l.getBbox().width, l.getBbox().height);
		//} while ((rTN & l.getBbox()).area() > 0 && iteration++ < 100 && (rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= currentImages[0].cols || rTN.y + rTN.height >= currentImages[0].rows));

		////std::cout << rTN.x << " , " << rTN.y << "   " << rTN.width << " " << rTN.height << std::endl;

		//currentImages[0](rTN).copyTo(rgbTN);
		//cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));

		//imshow("ElementTP", rgbTP);

		//imshow("ElementTN", rgbTN);




		//auto result = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, true);
		//imshow("ElementTPResult", result.hogImage);

		//result = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, true);
		//imshow("ElementTNResult", result.hogImage);



		//float tpResult = d.evaluate(rgbTP);// svm->predict(result.getFeatureArray(true).toMat());
		////	std::cout << "TP result: " << tpResult << std::endl;

		//result = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, true);
		//float tnResult = d.evaluate(rgbTN);// svm->predict(result.getFeatureArray(true).toMat());
		////	std::cout << "TN result: " << tnResult << std::endl;

		waitKey(0);

		std::cout << l.getNumber() << "   " << l.getBbox().x << "," << l.getBbox().y << " / " << l.getBbox().width << "x" << l.getBbox().height << std::endl;
	}
}

int main()
{
	//testSVMEvaluation();

	std::cout << "--------------------- New console session -----------------------" << std::endl;
	Detector d;

	std::cout << "Detector Features options:" << std::endl;
	d.toString(std::cout);
	//d.buildWeakHoGSVMClassifier();
	d.loadModel(std::string("")); // TODO

	/*
	std::cout << "Training set evaluation" << std::endl;
	ClassifierEvaluation evalResult = d.evaluateWeakHoGSVMClassifier(true);
	evalResult.print(std::cout);

	std::cout << "Test set evaluation" << std::endl;
	evalResult = d.evaluateWeakHoGSVMClassifier(false);
	evalResult.print(std::cout);
	*/

	testDetection(d, 0.06);


	std::cout << "Done" << std::endl;

	getchar();

	//testDetection();


	//auto img = imread("D:\\circle.png");
	auto img = imread("C:\\Users\\dwight\\Downloads\\dwight\\kitti\\rgb\\000000.png");
	namedWindow("Test");
	imshow("Test", img);


	//	auto result = getHistogramsOfOrientedGradient(img, 8, 9, true);


	Mat hsl;
	cvtColor(img, hsl, CV_BGR2HLS);



	Mat hs(hsl.rows, hsl.cols, CV_32F, Scalar(0));
	for (int y = 0; y < hsl.rows; y++)
	{
		for (int x = 0; x < hsl.cols; x++)
		{
			cv::Vec3b v = hsl.at<cv::Vec3b>(y, x);
			float h = v[0] / 180.0;
			float s = v[1] / 255.0;
			hs.at<float>(y, x) = h*s;
		}
	}

	auto result = getHistogramsOfX(Mat(img.rows, img.cols, CV_32FC1, Scalar(1.0)), hs, patchSize, binSize, true, true);

	for (int y = 0; y < result.height; y++)
	{
		for (int x = 0; x < result.width; x++)
		{

		}
	}

	cv::normalize(hs, hs, 0, 1, CV_MINMAX);

	namedWindow("HoG");
	imshow("HoG", hs);

	setMouseCallback("HoG", [](int event, int x, int y, int flags, void* userdata) -> void {
		HoGResult* r = (HoGResult*)userdata;

		int cx = x / patchSize;
		int cy = y / patchSize;

		if (cx >= 0 && cy >= 0 && cy < r->data.size() && cx < r->data[cy].size())
			showHistogram(r->data[cy][cx]);
	}, &result);


	waitKey(0);


	getchar();

	return 0;
}

