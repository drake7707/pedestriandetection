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

cv::Ptr<ml::SVM> buildModel() {

	int nrOfTN = 3;

	std::vector<std::vector<float>> truePositiveFeatures;
	std::vector<std::vector<float>> trueNegativeFeatures;
	int featureSize;

	KITTIDataSet dataset("D:\\PedestrianDetectionDatasets\\kitti");

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

			// take an equivalent patch at random for a true negative
			cv::Mat rgbTN;
			cv::Rect2d rTN;

			// build training mat

			auto resultTP = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, false);
			truePositiveFeatures.push_back(resultTP.getFeatureArray());



			for (int k = 0; k < nrOfTN; k++)
			{
				int iteration = 0;
				do {
					rTN = cv::Rect2d(randBetween(0, currentImages[0].cols - l.getBbox().width), randBetween(0, currentImages[0].rows - l.getBbox().height), l.getBbox().width, l.getBbox().height);
				} while ((rTN & l.getBbox()).area() > 0 && iteration++ < 100);

				currentImages[0](rTN).copyTo(rgbTN);
				cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));

				auto resultTN = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, false);
				trueNegativeFeatures.push_back(resultTN.getFeatureArray());
			}

			if (truePositiveFeatures.size() % 10 == 0)
				std::cout << l.getNumber() << "   " << l.getBbox().x << "," << l.getBbox().y << " / " << l.getBbox().width << "x" << l.getBbox().height << std::endl;
		}
	}

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


	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setC(0.01);
	svm->setDegree(2);

	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);



	svm->save("kittitraining.xml");

	return svm;
}


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





struct MatchRegion {
	cv::Rect2d region;
	float svmResult;
};

std::vector<MatchRegion> evaluateModelOnImage(cv::Mat& img, Ptr<ml::SVM> svm, float threshold) {

	int slidingWindowWidth = 32;
	int slidingWindowHeight = 64;
	int slidingWindowStep = 8;
	int maxScaleReduction = 4; // 4 times reduction


	//namedWindow("Test2");
	std::vector<MatchRegion> matchingRegions;

	for (float invscale = 1; invscale <= maxScaleReduction; invscale+=0.5)
	{
		cv::Mat imgToTest;
		cv::resize(img, imgToTest, cv::Size2d(ceilTo(img.cols / invscale, slidingWindowWidth), ceilTo(img.rows / invscale, slidingWindowHeight)));

		for (int j = 0; j < imgToTest.rows / slidingWindowHeight - 1; j++)
		{
			for (int verticalStep = 0; verticalStep < slidingWindowWidth; verticalStep += slidingWindowStep)
			{
				for (int i = 0; i < imgToTest.cols / slidingWindowWidth - 1; i++)
				{

					for (int horizontalStep = 0; horizontalStep < slidingWindowWidth; horizontalStep += slidingWindowStep)
					{
						cv::Rect windowRect(i * slidingWindowWidth + horizontalStep, j * slidingWindowHeight + verticalStep, slidingWindowWidth, slidingWindowHeight);
						cv::Mat region;
						cv::resize(imgToTest(windowRect), region, Size2d(refWidth, refHeight));


						auto result = getHistogramsOfOrientedGradient(region, patchSize, binSize);
						result = getHistogramsOfOrientedGradient(region, patchSize, binSize, false);
						float svmResult = svm->predict(result.getFeatureMat(), noArray(), ml::StatModel::Flags::RAW_OUTPUT);
						if (svmResult > threshold) {

							double scaleX = img.cols / (double)imgToTest.cols;
							double scaleY = img.rows / (double)imgToTest.rows;

							MatchRegion mr;
							mr.region = Rect2d(windowRect.x * scaleX, windowRect.y * scaleY, windowRect.width * scaleX, windowRect.height * scaleY);
							mr.svmResult = svmResult;
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


void testDetection() {
	Ptr<ml::SVM> svm;
	//	svm = buildModel();

	svm = Algorithm::load<ml::SVM>("kittitraining.xml");

	KITTIDataSet dataset("D:\\PedestrianDetectionDatasets\\kitti");

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

		cv::Mat rgb = currentImages[0].clone();
		cv::Mat depth = currentImages[1].clone();
		cv::rectangle(rgb, l.getBbox(), Scalar(0, 0, 255), 2);
		cv::rectangle(depth, l.getBbox(), Scalar(0, 0, 255), 2);

		std::vector<MatchRegion> matchingRegions = evaluateModelOnImage(currentImages[0], svm, 0.7);

		for (auto& region : matchingRegions) {
			cv::rectangle(rgb, region.region, Scalar(0, 255 * region.svmResult, 0), 2);
		}
		std::cout << "Found " << matchingRegions.size() << " nr of matching regions " << std::endl;


		imshow("RGB", rgb);
		imshow("DEPTH", depth);

		cv::Mat rgbTP;
		currentImages[0](l.getBbox()).copyTo(rgbTP);
		cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

		// take an equivalent patch at random for a true negative
		cv::Mat rgbTN;
		cv::Rect2d rTN;

		int iteration = 0;
		do {
			rTN = cv::Rect2d(randBetween(0, rgb.cols - l.getBbox().x), randBetween(0, rgb.rows - l.getBbox().y), l.getBbox().width, l.getBbox().height);
		} while ((rTN & l.getBbox()).area() > 0 && iteration++ < 100);

		currentImages[0](rTN).copyTo(rgbTN);
		cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));

		imshow("ElementTP", rgbTP);

		imshow("ElementTN", rgbTN);




		auto result = getHistogramsOfOrientedGradient(rgbTP, patchSize, binSize, true);
		imshow("ElementTPResult", result.hogImage);

		result = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, true);
		imshow("ElementTNResult", result.hogImage);



		float tpResult = svm->predict(result.getFeatureMat());
		std::cout << "TP result: " << tpResult << std::endl;

		result = getHistogramsOfOrientedGradient(rgbTN, patchSize, binSize, true);
		float tnResult = svm->predict(result.getFeatureMat());
		std::cout << "TN result: " << tnResult << std::endl;

		waitKey(0);

		std::cout << l.getNumber() << "   " << l.getBbox().x << "," << l.getBbox().y << " / " << l.getBbox().width << "x" << l.getBbox().height << std::endl;
	}
}

int main()
{
	


	//auto img = imread("D:\\circle.png");
	auto img = imread("D:\\test.jpg");
	namedWindow("Test");
	imshow("Test", img);


	auto result = getHistogramsOfOrientedGradient(img, 8, 18, true);


	
	namedWindow("HoG");
	imshow("HoG", result.hogImage);

	setMouseCallback("HoG", [](int event, int x, int y, int flags, void* userdata) -> void {
		HoGResult* r = (HoGResult*)userdata;

		int cx = x / patchSize;
		int cy = y / patchSize;

		showHistogram(r->data[cy][cx]);
	}, &result);


	waitKey(0);


	getchar();

	return 0;
}

