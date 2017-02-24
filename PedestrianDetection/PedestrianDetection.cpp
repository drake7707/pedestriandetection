// PedestrianDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

#include "HistogramOfOrientedGradients.h"

#include "ModelEvaluator.h"
#include "IFeatureCreator.h"
#include "HOGRGBFeatureCreator.h"
#include "HOGDepthFeatureCreator.h"
#include "HOGRGBHistogramVarianceFeatureCreator.h"
#include "RGBCornerFeatureCreator.h"
#include "HistogramDepthFeatureCreator.h"


#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"


#include "KITTIDataSet.h"
#include "DataSet.h"

//std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
//std::string baseDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti\\regions";

std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";
std::string baseDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti\\regions";

int patchSize = 8;
int binSize = 9;
int refWidth = 64;
int refHeight = 128;




void cornerFeatureTest() {
	cv::Mat mRGB = cv::imread(kittiDatasetPath + "\\test.jpg");

	cv::cvtColor(mRGB, mRGB, cv::COLOR_BGR2GRAY);

	cv::Mat dst;
	cv::cornerHarris(mRGB, dst, 2, 3, 0);


	// Detecting corners
	cornerHarris(mRGB, dst, 7, 5, 0.05, cv::BORDER_DEFAULT);

	// Normalizing
	cv::Mat dst_norm;
	cv::Mat dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, cv::NormTypes::NORM_MINMAX, CV_32FC1, cv::Mat());

	Histogram h(101, 0);

	auto& mean = cv::mean(dst_norm);
	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			dst_norm.at<float>(j, i) = abs(dst_norm.at<float>(j, i) - mean[0]) *(dst_norm.at<float>(j, i) - mean[0]);
		}
	}
	convertScaleAbs(dst_norm, dst_norm_scaled);


	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			//	dst_norm_scaled.at<uchar>(j, i) = (dst_norm_scaled.at<uchar>(j, i) - mean[0])*(dst_norm_scaled.at<uchar>(j, i) - mean[0]);

			int idx = floor(dst_norm_scaled.at<uchar>(j, i) / 255.0 * (h.size() - 1));
			//	if (idx >= 0 && idx < h.size())
			h[idx]++;
		}
	}
	showHistogram(h);
	cv::Mat binningValues;
	dst_norm_scaled.convertTo(binningValues, CV_32F);
	normalize(binningValues, binningValues, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC1, cv::Mat());


	auto histResult = hog::getHistogramsOfX(cv::Mat(dst_norm_scaled.rows, dst_norm_scaled.cols, CV_32FC1, cv::Scalar(1)), binningValues, patchSize, binSize, true, true);


	cv::imshow("Test", histResult.hogImage);
	cv::waitKey(0);

}

bool overlaps(std::vector<DataSetLabel> labels, cv::Rect2d r, std::vector<cv::Rect2d> selectedRegions) {
	for (auto& l : labels) {
		if ((r & l.getBbox()).area() > 0)
			return true;
	}

	for (auto& region : selectedRegions) {
		if ((r & region).area() > 0)
			return true;
	}

	return false;
}

void saveTNTP() {

	KITTIDataSet* dataSet = new KITTIDataSet(kittiDatasetPath);

	srand(7707);


	std::vector<DataSetLabel> labels = dataSet->getLabels();

	std::map<std::string, std::vector<DataSetLabel>> labelsPerNumber;
	for (auto& l : labels)
		labelsPerNumber[l.getNumber()].push_back(l);



	int sizeVariance = 4;
	int tpIdxRGB = 0;
	int tnIdxRGB = 0;

	int tpIdxDepth = 0;
	int tnIdxDepth = 0;

	int idx = 0;
	for (auto& pair : labelsPerNumber) {


		std::vector<cv::Mat> currentImages = dataSet->getImagesForNumber(pair.first);


		// get true positive and true negative image
		// -----------------------------------------

		int nrOfTP = 0;
		for (auto& l : pair.second) {
			cv::Mat rgbTP;
			cv::Rect2d& r = l.getBbox();
			if (r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {

				//for (auto& img : currentImages) {

				for (int i = 0; i < currentImages.size(); i++)
				{
					auto& img = currentImages[i];

					img(l.getBbox()).copyTo(rgbTP);
					cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

					// build training mat

					//tpFunc(rgbTP);
					std::string path;
					path = "D:\\PedestrianDetectionDatasets\\kitti\\regions\\tp\\";
					path += i == 0 ? "rgb" : "depth";
					path += std::to_string(i == 0 ? tpIdxRGB : tpIdxDepth) + ".png";
					imwrite(path, rgbTP);
					if (i == 0) {
						tpIdxRGB++;
						nrOfTP++;
					}
					else
						tpIdxDepth++;
					//truePositiveFeatures.push_back(resultTP.getFeatureArray());

					cv::Mat rgbTPFlip;
					cv::flip(rgbTP, rgbTPFlip, 1);

					//tpFunc(rgbTPFlip);
					path = "D:\\PedestrianDetectionDatasets\\kitti\\regions\\tp\\";
					path += i == 0 ? "rgb" : "depth";
					path += std::to_string(i == 0 ? tpIdxRGB : tpIdxDepth) + ".png";
					imwrite(path, rgbTPFlip);
					if (i == 0) {
						tpIdxRGB++;
						nrOfTP++;
					}
					else
						tpIdxDepth++;
				}

			}
		}

		// take a number of true negative patches that don't overlap
		std::vector<cv::Rect2d> selectedTNRegions;
		for (int k = 0; k < nrOfTP; k++)
		{
			// take an equivalent patch at random for a true negative
			cv::Mat rgbTN;
			cv::Rect2d rTN;

			int iteration = 0;
			do {
				double sizeMultiplier = (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
				double width = refWidth * sizeMultiplier;
				double height = refHeight * sizeMultiplier;
				rTN = cv::Rect2d(randBetween(0, currentImages[0].cols - width), randBetween(0, currentImages[0].rows - height), width, height);
			} while (iteration++ < 10000 && (overlaps(pair.second, rTN, selectedTNRegions) || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= currentImages[0].cols || rTN.y + rTN.height >= currentImages[0].rows));


			if (iteration < 10000) {
				selectedTNRegions.push_back(rTN);
				for (int i = 0; i < currentImages.size(); i++)
				{
					auto& img = currentImages[i];

					img(rTN).copyTo(rgbTN);
					cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));
					//tnFunc(rgbTN);
					std::string path = "D:\\PedestrianDetectionDatasets\\kitti\\regions\\tn\\";
					path += i == 0 ? "rgb" : "depth";
					path += std::to_string(i == 0 ? tnIdxRGB : tnIdxDepth) + ".png";
					imwrite(path, rgbTN);
					if (i == 0)
						tnIdxRGB++;
					else
						tnIdxDepth++;
				}
			}
		}
	}
}



void testClassifier() {
	FeatureSet testSet;
	testSet.addCreator(new HOGRGBFeatureCreator(patchSize, binSize, refWidth, refHeight));
	testSet.addCreator(new HOGRGBHistogramVarianceFeatureCreator(patchSize, binSize, refWidth, refHeight));
	testSet.addCreator(new HOGDepthFeatureCreator(patchSize, binSize, refWidth, refHeight));

	ModelEvaluator model(baseDatasetPath, testSet);

	//model.train();
	//model.saveModel(std::string("testmodel.xml"));
	model.loadModel(std::string("testmodel.xml"));

	ClassifierEvaluation eval = model.evaluate(1)[0];
	eval.print(std::cout);

	cv::Mat mRGB = cv::imread(kittiDatasetPath + "\\rgb\\000000.png");
	cv::Mat mDepth = cv::imread(kittiDatasetPath + "\\depth\\000000.png");

	slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect2d bbox) -> void {
		cv::Mat regionRGB;
		cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));

		cv::Mat regionDepth;
		cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));

		FeatureVector v = testSet.getFeatures(regionRGB, regionDepth);
		int result = model.evaluateWindow(regionRGB, regionDepth, -5);
		if (result == 1)
			cv::rectangle(mRGB, bbox, cv::Scalar(0, 255, 0), 1);
	}, 0.25, 1);
	cv::imshow("Test", mRGB);

	// this will leak because creators are never disposed!
	cv::waitKey(0);
}


int main()
{
	//testClassifier();
	/*int nr = 0;
	while (true) {
		char nrStr[7];
		sprintf(nrStr, "%06d", nr);
		cv::Mat tp = cv::imread(kittiDatasetPath + "\\regions\\tp\\depth" + std::to_string(nr) + ".png");
		cv::Mat tn = cv::imread(kittiDatasetPath + "\\regions\\tn\\depth" + std::to_string(nr) + ".png");


		std::function<void(cv::Mat&,std::string)> func = [&](cv::Mat& img, std::string msg) -> void {
			cv::Mat gray;
			cv::cvtColor(img, gray, CV_BGR2GRAY);

			cv::normalize(img, img, 0, 255, cv::NormTypes::NORM_MINMAX);

			std::vector<cv::Point2f> corners;
			int maxCorners = 1000;
			float qualityLevel = 0.01;
			float minDistance = 5;

			cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);
			for (auto& c : corners) {
				cv::circle(img, c, 2, cv::Scalar(0, 255, 0), -1);
			}


			Histogram h(26, 0);
			for (int j = 0; j < img.rows; j++)
			{
				for (int i = 0; i < img.cols; i++)
				{
					int idx = img.at<uchar>(j, i) / 10;
					h[idx]++;
				}
			}
			showHistogram(h, std::string("Histogram_") + msg);
			cv::imshow(msg, img);
		};

		func(tp, "TP");

		func(tn, "TN");

		cv::waitKey(0);
		nr++;
	}
	*/
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	//testClassifier();
	//saveTNTP();
	//return 0;

	std::cout << hog::getNumberOfFeatures(64, 128, 8, 9) << std::endl;

	FeatureTester tester(baseDatasetPath);
	tester.addAvailableCreator(std::string("HoG(RGB)"), new HOGRGBFeatureCreator(patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(std::string("S2HoG(RGB)"), new HOGRGBHistogramVarianceFeatureCreator(patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(std::string("HoG(Depth)"), new HOGDepthFeatureCreator(patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(std::string("Corner(RGB)"), new RGBCornerFeatureCreator(patchSize, refWidth, refHeight));
	tester.addAvailableCreator(std::string("Histogram(Depth)"), new HistogramDepthFeatureCreator());

	int nrOfEvaluations = 100;
	std::set<std::string> set;

	set = { "Corner(RGB)" };
	tester.addJob(set, nrOfEvaluations);
	
	set = { "HoG(RGB)", "Corner(RGB)" };
	tester.addJob(set, nrOfEvaluations);


	set = { "HoG(RGB)","Histogram(Depth)" };
	tester.addJob(set, nrOfEvaluations);


	set = { "Histogram(Depth)" };
	tester.addJob(set, nrOfEvaluations);


	set = { "HoG(RGB)" };
	tester.addJob(set, nrOfEvaluations);

	set = { "HoG(RGB)", "S2HoG(RGB)" };
	tester.addJob(set, nrOfEvaluations);

	set = { "HoG(RGB)", "HoG(Depth)" };
	tester.addJob(set, nrOfEvaluations);

	set = { "HoG(RGB)",  "S2HoG(RGB)",  "HoG(Depth)" };
	tester.addJob(set, nrOfEvaluations);



	tester.runJobs();





	//FeatureSet set;
	//set.addCreator(&HOGRGBFeatureCreator());
	////set.addCreator(&HOGDepthFeatureCreator());



	//{
	//	std::cout << "Testing with 0.8/0.2 prior weights" << std::endl;
	//	ModelEvaluator evaluator(baseDatasetPath, set, 0.8, 0.2);
	//	evaluator.train();
	//	ClassifierEvaluation eval = evaluator.evaluate();
	//	eval.print(std::cout);
	//}
	//{
	//	std::cout << "Testing with 0.5/0.5 prior weights" << std::endl;
	//	ModelEvaluator evaluator(baseDatasetPath, set, 0.5, 0.5);
	//	evaluator.train();
	//	ClassifierEvaluation eval = evaluator.evaluate();
	//	eval.print(std::cout);
	//}
	//{
	//	std::cout << "Testing with 0.2/0.8 prior weights" << std::endl;
	//	ModelEvaluator evaluator(baseDatasetPath, set, 0.2, 0.8);
	//	evaluator.train();
	//	ClassifierEvaluation eval = evaluator.evaluate();
	//	eval.print(std::cout);
	//}


	//getchar();
	return 0;
}

