// PedestrianDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include "ProgressWindow.h"

#include "Helper.h"
#include "TrainingDataSet.h"

#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"

#include "ModelEvaluator.h"
#include "IFeatureCreator.h"
#include "HOGFeatureCreator.h"
#include "HOGHistogramVarianceFeatureCreator.h"
#include "CornerFeatureCreator.h"
#include "HistogramDepthFeatureCreator.h"
#include "SURFFeatureCreator.h"
#include "ORBFeatureCreator.h"
#include "SIFTFeatureCreator.h"
#include "CenSurEFeatureCreator.h"
#include "MSDFeatureCreator.h"
#include "BRISKFeatureCreator.h"
#include "FASTFeatureCreator.h"
#include "HDDFeatureCreator.h"
#include "LBPFeatureCreator.h"
#include "HONVFeatureCreator.h"
#include "CoOccurenceMatrixFeatureCreator.h"
#include "RAWRGBFeatureCreator.h"
#include "HOIFeatureCreator.h"
#include "SDDGFeatureCreator.h"
#include "RAWLUVFeatureCreator.h"


#include "EvaluatorCascade.h"

#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"

#include "CoOccurenceMatrix.h"

#include "KITTIDataSet.h"
#include "KAISTDataSet.h"

#include "DataSet.h"

#include "JetHeatMap.h"
#include "EvaluationSettings.h"
#include "RiskAnalysis.h"



//std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
//std::string kaistDatasetPath = "D:\\PedestrianDetectionDatasets\\kaist";
std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";

int patchSize = 8;
int binSize = 9;


TrainingDataSet buildInitialTrainingSet(EvaluationSettings& settings, DataSet* dataSet) {

	TrainingDataSet trainingSet(dataSet);

	srand(7707);

	ProgressWindow::getInstance()->updateStatus(std::string("Initial training set (train0)"), 0, "Loading dataset labels");

	std::vector<std::vector<DataSetLabel>> labelsPerNumber = dataSet->getLabelsPerNumber();

	int sizeVariance = 8; // from 0.25 to 2 times the refWidth and refHeight ( so anything between 16x32 - 32x64 - 64x128 - 128x256, more scales might be evaluated later )
	int nrOfTNPerImage = 2;

	for (int i = 0; i < labelsPerNumber.size(); i++)
	{
		TrainingImage tImg;
		tImg.number = i;


		std::vector<cv::Mat> currentImages = dataSet->getImagesForNumber(i);
		std::vector<cv::Rect2d> selectedRegions;


		ProgressWindow::getInstance()->updateStatus(std::string("Initial training set (train0)"), 1.0 * i / labelsPerNumber.size(), "Building training set (" + std::to_string(i) + ")");

		for (auto& l : labelsPerNumber[i]) {

			cv::Mat rgbTP;
			cv::Rect2d& r = l.getBbox();
			if (!l.isDontCareArea() && r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {

				TrainingRegion tr;
				tr.region = l.getBbox();
				tr.regionClass = 1;
				tImg.regions.push_back(tr);
				selectedRegions.push_back(r);
			}
			else {
				TrainingRegion tr;
				tr.region = l.getBbox();
				tr.regionClass = 0; // don't care
				tImg.regions.push_back(tr);
				selectedRegions.push_back(l.getBbox());
			}
		}

		for (int k = 0; k < nrOfTNPerImage; k++)
		{
			cv::Mat rgbTN;
			cv::Rect2d rTN;

			int iteration = 0;
			do {
				double sizeMultiplier = 0.25 * (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
				double width = settings.refWidth * sizeMultiplier;
				double height = settings.refHeight * sizeMultiplier;
				rTN = cv::Rect2d(randBetween(0, currentImages[0].cols - width), randBetween(0, currentImages[0].rows - height), width, height);
			} while (iteration++ < 10000 && (intersectsWith(rTN, selectedRegions) || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= currentImages[0].cols || rTN.y + rTN.height >= currentImages[0].rows));


			if (iteration < 10000) {

				TrainingRegion tr;
				tr.region = rTN;
				tr.regionClass = -1;
				tImg.regions.push_back(tr);

				selectedRegions.push_back(rTN);
			}
		}


		trainingSet.addTrainingImage(tImg);
	}

	ProgressWindow::getInstance()->finish(std::string("Initial training set (train0)"));
	return trainingSet;
}

cv::Mat getMask(cv::Mat& roi) {
	cv::Mat mask;
	roi.copyTo(mask);

	// normalize but replace any 0 values with the min

	float max = std::numeric_limits<float>().min();
	float min = std::numeric_limits<float>().max();
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{
			float val = mask.at<float>(j, i);
			if (val > 0) {
				if (val > max) max = val;
				if (val < min) min = val;
			}
		}
	}
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{
			float val = mask.at<float>(j, i);
			if (val > 0) {
				mask.at<float>(j, i) = (val - min) / (max - min);
			}
			else
				mask.at<float>(j, i) = min;
		}
	}
	//cv::normalize(mask, mask, 0, 1, cv::NormTypes::NORM_MINMAX);
	mask.convertTo(mask, CV_8UC1, 255);
	cv::threshold(mask, mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	return mask;
}

void testClassifier(FeatureTester& tester, EvaluationSettings& settings) {

	std::set<std::string> set = { "HOG(RGB)","HDD" };

	auto fset = tester.getFeatureSet(set);

	EvaluatorCascade cascade(std::string("Test"));
	cascade.load(std::string("models\\KITTI_HDD+HOG(RGB)_cascade.xml"), std::string("models"));

	double valueShift = -6;


	//ModelEvaluator modelFinal(std::string("Test"));
	//modelFinal.loadModel(std::string("models\\HOG(Depth)+HOG(RGB)+LBP(RGB) round 3.xml"));



	/*ClassifierEvaluation eval = model.evaluateDataSet(1, false)[0];
	eval.print(std::cout);*/

	KITTIDataSet dataSet(settings.kittiDataSetPath);

	//	cv::namedWindow("Test");

	auto& entries = cascade.getEntries();

	std::mutex m;
	int nr = 0;
	//while (true) {

	std::vector<DataSetLabel> labels = dataSet.getLabels();

	std::vector<std::vector<DataSetLabel>> labelsPerNumber = dataSet.getLabelsPerNumber();

	ClassifierEvaluation eval(dataSet.getNrOfImages());

	parallel_for(0, 4000, 6, [&](int i) -> void {
		ProgressWindow::getInstance()->updateStatus(std::string("Testing classifier"), 1.0 * i / 1000, std::to_string(i));

		if (labelsPerNumber[i].size() <= 0)
			return;

		auto imgs = dataSet.getImagesForNumber(i);
		cv::Mat mRGB = imgs[0];
		cv::Mat mDepth = imgs[1];
		cv::Mat mThermal = imgs[2];

		int nrOfWindowsEvaluated = 0;
		int nrOfWindowsSkipped = 0;
		long nrOfWindowsPositive = 0;

		std::vector<SlidingWindowRegion> predictedPositiveRegions;

		std::vector<SlidingWindowRegion> truepositiveregions;
		std::vector<SlidingWindowRegion> falsepositiveregions;


		long evaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {

			ROIManager roiManager;
			roiManager.prepare(mRGB, mDepth, mThermal);

			//std::vector<std::vector<IPreparedData*>> preparedData;

			//cv::Mat tmp(mRGB.rows, mRGB.cols, CV_8UC1, cv::Scalar(0));
			//for (int j = 0; j < mRGB.rows; j++)
			//{
			//	for (int i = 0; i < mRGB.cols; i++)
			//	{
			//		bool mustContinue = roiManager.needToEvaluate(cv::Rect(i,j,1,1), mRGB, mDepth, mThermal,
			//			[&](double height, double depthAvg) -> bool { return dataSet.isWithinValidDepthRange(height, depthAvg); });
			//		if (mustContinue)
			//			tmp.at<char>(j, i) = 255;
			//	}
			//}
			//cv::imshow("NeedToEvaluateRegions", tmp);

			for (int s = 0; s < settings.windowSizes.size(); s++) {

				double scale = 1.0  * settings.refWidth / settings.windowSizes[s].width;

				cv::Mat rgbScale;
				if (mRGB.cols > 0 && mRGB.rows > 0)
					cv::resize(mRGB, rgbScale, cv::Size2d(mRGB.cols * scale, mRGB.rows * scale));

				cv::Mat depthScale;
				if (mDepth.cols > 0 && mDepth.rows > 0)
					cv::resize(mDepth, depthScale, cv::Size2d(mDepth.cols * scale, mDepth.rows * scale));

				cv::Mat thermalScale;
				if (mThermal.cols > 0 && mThermal.rows > 0)
					cv::resize(mThermal, thermalScale, cv::Size2d(mThermal.cols * scale, mThermal.rows * scale));

				auto preparedData = fset->buildPreparedDataForFeatures(rgbScale, depthScale, thermalScale);


				slideWindow(rgbScale.cols, rgbScale.rows, [&](cv::Rect bbox) -> void {

					cv::Rect2d scaledBBox = cv::Rect2d(bbox.x / scale, bbox.y / scale, bbox.width / scale, bbox.height / scale);

					cv::Mat regionRGB;
					if (rgbScale.cols > 0 && rgbScale.rows > 0)
						regionRGB = rgbScale(bbox);

					cv::Mat regionDepth;
					if (depthScale.cols > 0 && depthScale.rows > 0)
						regionDepth = depthScale(bbox);

					cv::Mat regionThermal;
					if (thermalScale.cols > 0 && thermalScale.rows > 0)
						regionThermal = thermalScale(bbox);

					bool mustContinue = roiManager.needToEvaluate(scaledBBox, mRGB, mDepth, mThermal,
						[&](double height, double depthAvg) -> bool { return dataSet.isWithinValidDepthRange(height, depthAvg); });

					double result;
					bool predictedPositive = false;
					if (mustContinue) {
						FeatureVector v = fset->getFeatures(regionRGB, regionDepth, regionThermal, bbox, preparedData);
						result = cascade.evaluateFeatures(v);
						if ((result + valueShift > 0 ? 1 : -1) == 1) {
							nrOfWindowsPositive++;
							predictedPositive = true;

							predictedPositiveRegions.push_back(SlidingWindowRegion(i, scaledBBox, abs(result)));
						}
					}
					else
						nrOfWindowsSkipped++;

					nrOfWindowsEvaluated++;

				}, 16, settings.refWidth, settings.refHeight);
			}

			//// clean up of prepared data
			//for (auto& v : preparedData) {
			//	for (int i = 0; i < v.size(); i++) {
			//		if (v[i] != nullptr)
			//			delete v[i];
			//	}
			//}
		});


		auto nmsresult = applyNonMaximumSuppression(predictedPositiveRegions, 0.5);

		std::vector<cv::Rect2d> tpRegions;
		std::vector<cv::Rect2d> dontCareRegions;
		for (auto& l : labelsPerNumber[i]) {
			if (l.isDontCareArea())
				dontCareRegions.push_back(l.getBbox());
			else
				tpRegions.push_back(l.getBbox());
		}

		std::vector<SlidingWindowRegion> posInDontCareRegions;

		for (auto& predpos : nmsresult) {
			if (!intersectsWith(predpos.bbox, dontCareRegions)) {

				bool actualPositive = false;

				if (overlaps(predpos.bbox, tpRegions)) {
					// should be positive
					actualPositive = true;
				}

				bool predictedPositive = true;
				if (predictedPositive == actualPositive) {
					if (predictedPositive) {
						eval.nrOfTruePositives++;
						truepositiveregions.push_back(predpos);
					}
					else {
						eval.nrOfTrueNegatives++;
					}
				}
				else {
					if (predictedPositive && !actualPositive) {
						// false positive
						eval.nrOfFalsePositives++;
						eval.falsePositivesPerImage[i]++;
						falsepositiveregions.push_back(predpos);
					}
					else {

						eval.nrOfFalseNegatives++;
					}
				}
			}
			else {
				posInDontCareRegions.push_back(predpos);
			}
		}

		int nrMissed = 0;
		std::vector<cv::Rect2d> predictedPos;
		for (auto& r : nmsresult) {
			predictedPos.push_back(r.bbox);
		}
		for (auto pos : tpRegions) {
			if (!overlaps(pos, predictedPos))
				nrMissed++;
		}




		for (auto& pos : posInDontCareRegions) {
			// even though it's in the don't care region, let's check if there there is a pedestrian there
			if (overlaps(pos.bbox, tpRegions))
				cv::rectangle(mRGB, pos.bbox, cv::Scalar(128, 164, 128), 2);
			else
				cv::rectangle(mRGB, pos.bbox, cv::Scalar(128, 128, 164), 2);

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << pos.score;
			std::string s = stream.str();
			cv::rectangle(mRGB, cv::Rect(pos.bbox.x, pos.bbox.y - 10, pos.bbox.width - 5, 10), cv::Scalar(128, 128, 128), -1);
			cv::putText(mRGB, s, cv::Point(pos.bbox.x, pos.bbox.y - 2), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);
		}

		cv::Mat maskOverlay(mRGB.rows, mRGB.cols, CV_8UC3, cv::Scalar(0));
		for (auto& pos : falsepositiveregions) {
			cv::rectangle(mRGB, pos.bbox, cv::Scalar(0, 0, 255), 2);

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << pos.score;
			std::string s = stream.str();
			cv::rectangle(mRGB, cv::Rect(pos.bbox.x, pos.bbox.y - 10, pos.bbox.width - 5, 10), cv::Scalar(0, 0, 255), -1);
			cv::putText(mRGB, s, cv::Point(pos.bbox.x, pos.bbox.y - 2), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);

			if (imgs[1].rows > 0 && imgs[1].cols > 0) {
				cv::Mat mask = getMask(imgs[1](pos.bbox));
				for (int j = 0; j < mask.rows; j++)
				{
					for (int i = 0; i < mask.cols; i++)
					{
						if (mask.at<uchar>(j, i) > 128)
							maskOverlay.at<Vec3b>(pos.bbox.y + j, pos.bbox.x + i) = cv::Vec3b(0, 0, 255);
					}
				}
			}
		}

		for (auto& pos : truepositiveregions) {
			cv::rectangle(mRGB, pos.bbox, cv::Scalar(0, 255, 0), 2);

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << pos.score;
			std::string s = stream.str();
			cv::rectangle(mRGB, cv::Rect(pos.bbox.x, pos.bbox.y - 10, pos.bbox.width - 5, 10), cv::Scalar(0, 255, 0), -1);
			cv::putText(mRGB, s, cv::Point(pos.bbox.x, pos.bbox.y - 2), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);


			if (imgs[1].rows > 0 && imgs[1].cols > 0) {
				cv::Mat mask = getMask(imgs[1](pos.bbox));
				for (int j = 0; j < mask.rows; j++)
				{
					for (int i = 0; i < mask.cols; i++)
					{
						if (mask.at<uchar>(j, i) > 128)
							maskOverlay.at<Vec3b>(pos.bbox.y + j, pos.bbox.x + i) = cv::Vec3b(0, 255, 0);
					}
				}
			}
			//	cv::imshow("Mask", mask);
			//	cv::waitKey(0);
		}
		//	cv::imshow("Mask", maskOverlay);
		mRGB = mRGB + 0.5 * maskOverlay;


		for (auto& l : labelsPerNumber[i]) {
			if (!l.isDontCareArea()) {

				std::string category = dataSet.getCategory(&l);
				if (category == "easy")
					cv::rectangle(mRGB, l.getBbox(), cv::Scalar(255, 255, 0), 2);
				else if (category == "moderate")
					cv::rectangle(mRGB, l.getBbox(), cv::Scalar(255, 128, 0), 2);
				else if (category == "hard")
					cv::rectangle(mRGB, l.getBbox(), cv::Scalar(255, 0, 0), 2);
				else
					cv::rectangle(mRGB, l.getBbox(), cv::Scalar(128, 0, 0), 2);
			}
		}
		//	auto nmsresult = applyNonMaximumSuppression(truepositiveregions, 0.2);
		//	for (auto& pos : nmsresult) {
		//		cv::rectangle(nms, pos.bbox, cv::Scalar(255, 255, 0), 2);
		//	}

		cv::rectangle(mRGB, cv::Rect(0, 0, mRGB.cols, 20), cv::Scalar(255, 255, 255), -1);
		std::string str = "FP: " + std::to_string(eval.falsePositivesPerImage[i]) + ", missed: " + std::to_string(nrMissed) + " #windows : " + std::to_string(nrOfWindowsEvaluated) + " (#skipped: " + std::to_string(nrOfWindowsSkipped) + "). Eval time: " + std::to_string(evaluationTime) + "ms " + "(decision shift : " + std::to_string(valueShift) + ")";
		cv::putText(mRGB, str, cv::Point(10, 10), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 0), 1, CV_AA);

		//cv::imshow("TestNMS", nms);
		m.lock();
		double posPercentage = 100.0 * nrOfWindowsPositive / (nrOfWindowsEvaluated - nrOfWindowsSkipped);
		std::cout << "Image: " << i << " Number of windows evaluated: " << nrOfWindowsEvaluated << " (skipped " << nrOfWindowsSkipped << ") and " << nrOfWindowsPositive << " positive (" << std::setw(2) << posPercentage << "%) " << evaluationTime << "ms (value shift: " << valueShift << ")" << std::endl;
		// this will leak because creators are never disposed!
		m.unlock();

		cv::imwrite(std::to_string(i) + "_hddHOGrgb.png", mRGB);

		//	cv::imwrite(std::to_string(i) + "_hddHOGrgb_mask.png", maskOverlay);
	//	cv::imshow("Test", mRGB);
	//	cv::waitKey(0);
		//nr++;
	});
}


void testFeature() {
	int nr = 0;
	while (true) {
		char nrStr[7];
		sprintf(nrStr, "%06d", nr);

		cv::Mat tp = cv::imread(kittiDatasetPath + "\\regions\\tp\\rgb" + std::to_string(nr) + ".png");
		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\regions\\tp\\rgb" + std::to_string(nr) + ".png");
		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\rgb\\000000.png", CV_LOAD_IMAGE_UNCHANGED);
		//cv::Mat tp = cv::imread("D:\\test.png", CV_LOAD_IMAGE_ANYDEPTH);
		//tp.convertTo(tp, CV_32FC1, 1.0 / 0xFFFF, 0);

		cv::Mat tn = cv::imread(kittiDatasetPath + "\\regions\\tn\\rgb" + std::to_string(nr) + ".png");

		std::function<void(cv::Mat&, std::string)> func = [&](cv::Mat& img, std::string msg) -> void {

			cv::Mat rgb = img.clone();
			cv::cvtColor(img, img, CV_BGR2HLS);
			img.convertTo(img, CV_32FC1, 1.0);
			cv::Mat hsl[3];
			cv::split(img, hsl);

			//hsl[0].convertTo(hsl[0], CV_32FC1, 1.0 / 360);

			cv::imshow(msg + "_input", hsl[0] / 180);


			cv::Mat hue = hsl[0] / 180;
			auto cells = getCoOccurenceMatrix(hue, patchSize, 16);

			cv::Mat m = createFullCoOccurrenceMatrixImage(rgb, cells, patchSize);
			cv::imshow(msg, m);




			//HOG::HOGResult result =  HOG::getHistogramsOfDepthDifferences(img, patchSize, binSize, true, true);
			/*	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
			std::vector<cv::KeyPoint> keypoints;


			cv::Mat descriptors;
			detector->detect(img, keypoints);


			//detector->compute(img, keypoints, descriptors);

			cv::Mat imgKeypoints;
			cv::drawKeypoints(img, keypoints, imgKeypoints);
			*/

			/*	cv::Mat depth;
				int d = img.depth();
				if (img.type() != CV_32FC1) {
					img.convertTo(depth, CV_32FC1, 1, 0);
				}
				else
					depth = img;*/
					/*
					Mat normals(depth.rows, depth.cols, CV_32FC3, cv::Scalar(0));
					Mat angleMat(depth.rows, depth.cols, CV_32FC3, cv::Scalar(0));

					//depth = depth * 255;
					for (int y = 1; y < depth.rows; y++)
					{
						for (int x = 1; x < depth.cols; x++)
						{


							float r = x + 1 >= depth.cols ? depth.at<float>(y, x) : depth.at<float>(y, x + 1);
							float l = x - 1 < 0 ? depth.at<float>(y, x) : depth.at<float>(y, x - 1);

							float b = y + 1 >= depth.rows ? depth.at<float>(y, x) : depth.at<float>(y + 1, x);
							float t = y - 1 < 0 ? depth.at<float>(y, x) : depth.at<float>(y - 1, x);


							float dzdx = (r - l) / 2.0;
							float dzdy = (b - t) / 2.0;

							Vec3f d(-dzdx, -dzdy, 0.0f);

							Vec3f tt(x, y - 1, depth.at<float>(y - 1, x));
							Vec3f ll(x - 1, y, depth.at<float>(y, x - 1));
							Vec3f c(x, y, depth.at<float>(y, x));

							Vec3f d2 = (ll - c).cross(tt - c);


							Vec3f n = normalize(d);

							double azimuth = atan2(-d2[1], -d2[0]); // -pi -> pi
							if (azimuth < 0)
								azimuth += 2 * CV_PI;

							double zenith = atan(sqrt(d2[1] * d2[1] + d2[0] * d2[0]));

							cv::Vec3f angles(azimuth / (2 * CV_PI), (zenith + CV_PI / 2) / CV_PI, 0);


							angleMat.at<cv::Vec3f>(y, x) = cv::Vec3f(dzdx, dzdy, 0);

							normals.at<Vec3f>(y, x) = n;
						}
					}

					normalize(abs(angleMat), angleMat, 0, 1, cv::NormTypes::NORM_MINMAX);

					auto& result = HOG::get2DHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, 9, true, false);



					//				cv::imshow(msg, normals);

								//cvtColor(img, img, CV_BGR2GRAY);
								//cv::Mat lbp = Algorithms::OLBP(img);
								//lbp.convertTo(lbp, CV_32FC1, 1 / 255.0, 0);

								//cv::Mat padded;
								//int padding = 1;
								//padded.create(img.rows,img.cols, lbp.type());
								//padded.setTo(cv::Scalar::all(0));
								//lbp.copyTo(padded(Rect(padding, padding, lbp.cols, lbp.rows)));

								//
								//auto& result = HOG::getHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), padded, patchSize, 20, true, false);

					cv::Mat tmp;
					img.convertTo(tmp, CV_8UC3, 255, 0);
					cv::imshow(msg, result.combineHOGImage(img));*/

					/*cv::Mat magnitude(img.size(), CV_32FC1, cv::Scalar(0));
					cv::Mat angle(img.size(), CV_32FC1, cv::Scalar(0));


					for (int j = 0; j < depth.rows; j++)
					{
						for (int i = 0; i < depth.cols ; i++)
						{

							float r = i + 1 >= depth.cols ? depth.at<float>(j, i) : depth.at<float>(j, i + 1);
							float l = i - 1 < 0 ? depth.at<float>(j, i) : depth.at<float>(j, i - 1);

							float b = j + 1 >= depth.rows ? depth.at<float>(j, i) : depth.at<float>(j + 1, i);
							float t = j - 1 < 0 ? depth.at<float>(j, i) : depth.at<float>(j - 1, i);

							float dx = (r - l) / 2;
							float dy = (b - t) / 2;

							double anglePixel = atan2(dy, dx);
							// don't limit to 0-pi, but instead use 0-2pi range
							anglePixel = (anglePixel < 0 ? anglePixel + 2 * CV_PI : anglePixel) + CV_PI / 2;
							if (anglePixel > 2 * CV_PI) anglePixel -= 2 * CV_PI;

							double magPixel = sqrt((dx*dx) + (dy*dy));
							magnitude.at<float>(j, i) = magPixel;
							angle.at<float>(j, i) = anglePixel / (2 * CV_PI);
						}
					}

					cv::normalize(abs(magnitude), magnitude, 0, 255, cv::NormTypes::NORM_MINMAX);

					auto& result =  HOG::getHistogramsOfX(magnitude, angle, patchSize, binSize, true, false);
					cv::Mat tmp;
					img.convertTo(tmp, CV_8UC3, 255, 0);
					cv::imshow(msg, magnitude);*/


					//Mat gray;
					//cv::cvtColor(img, gray, CV_BGR2GRAY);

					//Mat lbp = Algorithms::OLBP(gray);

					//lbp.convertTo(lbp, CV_32FC1, 1 / 255.0, 0);
					//cv::Mat padded;
					//int padding = 1;
					//padded.create(img.rows, img.cols, lbp.type());
					//padded.setTo(cv::Scalar::all(0));
					//lbp.copyTo(padded(Rect(padding, padding, lbp.cols, lbp.rows)));


					//auto& result = HOG::getHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), padded, patchSize, binSize, true, false);
					//cv::imshow(msg, result.combineHOGImage(img));

		};

		func(tp, "TP");

		func(tn, "TN");

		cv::waitKey(0);
		nr++;
	}

}

void testSlidingWindow(EvaluationSettings& settings) {

	KITTIDataSet dataSet(settings.kittiDataSetPath);

	auto labelsPerNumber = dataSet.getLabelsPerNumber();

	std::mutex lock;
	ClassifierEvaluation eval;

	std::map<std::string, ClassifierEvaluation> evalPerRisk;
	for (auto& c : RiskAnalysis::getRiskCategories())
		evalPerRisk[c] = ClassifierEvaluation();

	std::map<int, std::vector<SlidingWindowRegion>> predictedPositiveregionsPerImage;

	int nrEvaluated = 0;
	dataSet.iterateDataSetWithSlidingWindow(settings.windowSizes, settings.baseWindowStride, settings.refWidth, settings.refHeight,

		settings.testCriteria,

		[&](int imgNr) -> void {
		// start of image
		predictedPositiveregionsPerImage[imgNr] = std::vector<SlidingWindowRegion>();
	}, [&](int imgNr, double scale, cv::Mat& fullRGBScale, cv::Mat& fullDepthScale, cv::Mat& fullThermalScale) -> void {

	}, [&](int idx, int resultClass, int imageNumber, int scale, cv::Rect2d& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string("Testing sliding window"), 1.0 * nrEvaluated / dataSet.getNrOfImages(), std::string("Evaluating sliding window (") + std::to_string(nrEvaluated) + ")");

		bool predictedPositive = false;

		for (auto& l : labelsPerNumber[imageNumber]) {
			if (!l.isDontCareArea()) {
				// true positive


				double iou = getIntersectionOverUnion(scaledRegion, l.getBbox());
				predictedPositive = iou > 0.5;
				if (predictedPositive) {
					predictedPositiveregionsPerImage[imageNumber].push_back(SlidingWindowRegion(imageNumber, scaledRegion, iou));

					break;
				}
			}
		}

	}, [&](int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositives) -> void {
		// end of image

		auto predictedPositives = applyNonMaximumSuppression(predictedPositiveregionsPerImage[imageNumber]);

		lock.lock();
		std::vector<DataSetLabel> truePositiveLabels;
		std::vector<cv::Rect2d> truePositiveRegions;
		for (auto& l : labelsPerNumber[imageNumber]) {
			if (!l.isDontCareArea()) {
				truePositiveRegions.push_back(l.getBbox());
				truePositiveLabels.push_back(l);
			}
		}
		int tp = 0;
		for (auto& predpos : predictedPositives) {

			int tpIndex = getOverlapIndex(predpos.bbox, truePositiveRegions);
			if (tpIndex != -1) {

				eval.nrOfTruePositives++;
			}
			else {
				eval.nrOfFalsePositives++;
			}
		}


		std::vector<cv::Rect2d> predictedPosRegions;
		for (auto& r : predictedPositives)
			predictedPosRegions.push_back(r.bbox);


		std::map<std::string, int> nrOfTruePositivesPerRiskCategory;
		std::map<std::string, int> nrOfFalseNegativesPerRiskCategory;
		for (auto& c : RiskAnalysis::getRiskCategories()) {
			nrOfTruePositivesPerRiskCategory[c] = 0;
			nrOfFalseNegativesPerRiskCategory[c] = 0;
		}

		for (auto& l : truePositiveLabels) {
			std::string category = RiskAnalysis::getRiskCategory(l.z_3d, l.x_3d, settings.vehicleSpeedKMh, settings.tireRoadFriction);
			nrOfTruePositivesPerRiskCategory[category]++;
		}
		for (int tp = 0; tp < truePositiveRegions.size(); tp++)
		{
			if (!overlaps(truePositiveRegions[tp], predictedPosRegions)) {
				// missed a true positive
				eval.nrOfFalseNegatives++;

				DataSetLabel& l = truePositiveLabels[tp];
				std::string category = RiskAnalysis::getRiskCategory(l.z_3d, l.x_3d, settings.vehicleSpeedKMh, settings.tireRoadFriction);
				nrOfTruePositivesPerRiskCategory[category]--;
				nrOfFalseNegativesPerRiskCategory[category]++;
			}
		}

		for (auto& c : RiskAnalysis::getRiskCategories()) {
			evalPerRisk[c].nrOfTruePositives += nrOfTruePositivesPerRiskCategory[c];
			evalPerRisk[c].nrOfFalseNegatives += nrOfFalseNegativesPerRiskCategory[c];
		}

		nrEvaluated++;

		lock.unlock();
	}, settings.slidingWindowParallelization);


	std::cout << "Evaluation of ground truth with sliding window done, results: " << std::endl;
	eval.print(std::cout);

	for (auto& c : RiskAnalysis::getRiskCategories()) {

		std::cout << "Evaluation of ground truth with sliding window for category " << c << std::endl;
		std::cout << "Min. miss rate for category: " << evalPerRisk[c].getMissRate() << std::endl;
	}

}



struct DistanceBetween {
	SlidingWindowRegion tpRegion;
	SlidingWindowRegion tnRegion;

	double dot;

	DistanceBetween(double dot, SlidingWindowRegion& tpRegion, SlidingWindowRegion& tnRegion) : dot(dot), tpRegion(tpRegion), tnRegion(tnRegion) {

	}
	bool operator<(const DistanceBetween& other) const {
		return abs(this->dot) < abs(other.dot);
	}
};

void checkDistanceBetweenTPAndTN(std::string& trainingFile, EvaluationSettings& settings, std::string& outputFile) {
	KITTIDataSet dataSet(kittiDatasetPath);
	TrainingDataSet tSet(&dataSet);

	tSet.load(trainingFile);

	FeatureSet fset;
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HOG(Depth)"), IFeatureCreator::Target::Depth, patchSize, binSize, settings.refWidth, settings.refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HOG(RGB)"), IFeatureCreator::Target::RGB, patchSize, binSize, settings.refWidth, settings.refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, settings.refWidth, settings.refHeight)));
	//fset.addCreator(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(std::string("HDD"), patchSize, binSize, refWidth, refHeight)));


	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;

	std::vector<SlidingWindowRegion> truePositiveRegions;
	std::vector<SlidingWindowRegion> trueNegativeRegions;

	// don't use prepared data and roi for training data set
	std::vector<std::unique_ptr<IPreparedData>> preparedData;
	cv::Rect roi;

	tSet.iterateDataSet([&](int idx) -> bool { return (idx + 1) % 10 == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string("Min distance between TN/TP"), 1.0 * imageNumber / tSet.getNumberOfImages(), std::string("Building feature vectors (") + std::to_string(imageNumber) + ")");

		FeatureVector v = fset.getFeatures(rgb, depth, thermal, roi, preparedData);
		if (resultClass == 1) {
			truePositiveFeatures.push_back(v);
			truePositiveRegions.push_back(SlidingWindowRegion(imageNumber, region, 0));
		}
		else {
			trueNegativeFeatures.push_back(v);
			trueNegativeRegions.push_back(SlidingWindowRegion(imageNumber, region, 0));
		}
	}, settings.addFlippedInTrainingSet, settings.refWidth, settings.refHeight);

	int featureSize = truePositiveFeatures[0].size();

	// build mean and sigma vector
	int N = truePositiveFeatures.size() + trueNegativeFeatures.size();
	auto meanVector = std::vector<float>(featureSize, 0);
	auto sigmaVector = std::vector<float>(featureSize, 0);

	for (auto& featureVector : truePositiveFeatures) {
		for (int f = 0; f < featureSize; f++)
			meanVector[f] += featureVector[f];
	}
	for (auto& featureVector : trueNegativeFeatures) {
		for (int f = 0; f < featureSize; f++)
			meanVector[f] += featureVector[f];
	}

	for (int f = 0; f < featureSize; f++)
		meanVector[f] /= N;

	std::vector<double> sigmaSum(featureSize, 0);
	for (auto& featureVector : truePositiveFeatures) {
		for (int f = 0; f < featureSize; f++)
			sigmaSum[f] += (featureVector[f] - meanVector[f]) * (featureVector[f] - meanVector[f]);
	}
	for (auto& featureVector : trueNegativeFeatures) {
		for (int f = 0; f < featureSize; f++)
			sigmaSum[f] += (featureVector[f] - meanVector[f]) * (featureVector[f] - meanVector[f]);
	}

	for (int f = 0; f < featureSize; f++)
		sigmaVector[f] = sigmaSum[f] / (N - 1);

	// now apply the normalization on the feature arrays
	for (auto& featureVector : truePositiveFeatures) {
		featureVector.applyMeanAndVariance(meanVector, sigmaVector);
		featureVector.normalize();
	}

	for (auto& featureVector : trueNegativeFeatures) {
		featureVector.applyMeanAndVariance(meanVector, sigmaVector);
		featureVector.normalize();
	}


	std::ofstream str(outputFile);
	str << std::fixed;
	str << "idx;similarity;TPIndex;TNImage;TNx;TNy;TNwidth;TNheight;TPImage;TPx;TPy;TPwidth;TPheight;" << std::endl;


	std::set<DistanceBetween> distances;
	for (int j = 0; j < trueNegativeFeatures.size(); j++) {

		if (j % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string("Min distance between TN/TP"), 1.0 * j / trueNegativeFeatures.size(), std::string("Calculating min distance between TP and TN (") + std::to_string(j) + ")");

		auto& tn = trueNegativeFeatures[j];

		double minDistance = std::numeric_limits<double>().max();
		double minSimilarity = -1;
		int minTP = -1;


		for (int i = 0; i < truePositiveFeatures.size(); i++)
		{
			double tpNorm = truePositiveFeatures[i].norm();
			double similarity = tn.dot(truePositiveFeatures[i]);
			double abssimilarity = abs(similarity);
			if (minDistance > abssimilarity) {
				minDistance = abssimilarity;
				minSimilarity = similarity;
				minTP = i;
			}
		}

		auto& tpregion = truePositiveRegions[minTP];
		auto& tnregion = trueNegativeRegions[j];
		str << j << ";" << minSimilarity << "; " << minTP << "; " <<
			tnregion.imageNumber << ";" << tnregion.bbox.x << ";" << tnregion.bbox.y << ";" << tnregion.bbox.width << ";" << tnregion.bbox.height << ";" <<
			tpregion.imageNumber << ";" << tpregion.bbox.x << ";" << tpregion.bbox.y << ";" << tpregion.bbox.width << ";" << tpregion.bbox.height << ";"
			<< std::endl;

		DistanceBetween db(minSimilarity, tpregion, tnregion);
		distances.emplace(db);
	}

	str.flush();
	str.close();

	KITTIDataSet ds(kittiDatasetPath);
	for (auto& db : distances) {
		cv::Mat tn = ds.getImagesForNumber(db.tnRegion.imageNumber)[0];
		cv::Mat tp = ds.getImagesForNumber(db.tpRegion.imageNumber)[0];

		cv::rectangle(tp, db.tpRegion.bbox, cv::Scalar(0, 255, 0));
		cv::rectangle(tn, db.tnRegion.bbox, cv::Scalar(0, 0, 255));

		cv::putText(tp, std::to_string(db.dot), cv::Point(20, 20), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, CV_AA);

		cv::imshow("TruePos", tp);
		cv::imshow("TrueNeg", tn);


		cv::waitKey(0);
	}
}



void browseThroughTrainingSet(std::string& trainingFile, DataSet* dataSet) {

	TrainingDataSet tSet(dataSet);

	tSet.load(trainingFile);

	cv::namedWindow("TrainingSet");
	tSet.iterateDataSetImages([](int imgNr, cv::Mat& rgb, cv::Mat& depth, const std::vector<TrainingRegion>& regions) -> void {

		cv::Mat tmp = rgb.clone();

		for (auto& r : regions) {
			if (r.regionClass == 1)
				cv::rectangle(tmp, r.region, cv::Scalar(0, 255, 0));
			else if (r.regionClass == -1)
				cv::rectangle(tmp, r.region, cv::Scalar(0, 0, 255));
			else if (r.regionClass == 0) // mask out don't care regions
				cv::rectangle(tmp, r.region, cv::Scalar(192, 192, 192), -1);

		}
		cv::imshow("TrainingSet", tmp);
		cv::waitKey(0);
	});
}


void printHeightVerticalAvgDepthRelation(std::string& trainingFile, std::ofstream& str) {
	KITTIDataSet dataSet(kittiDatasetPath);
	TrainingDataSet tSet(&dataSet);
	tSet.load(trainingFile);

	str << std::fixed;
	str << "height" << ";" << "avgdepth" << std::endl;

	tSet.iterateDataSetImages([&](int imgNr, cv::Mat& rgb, cv::Mat& depth, const std::vector<TrainingRegion>& regions) -> void {

		cv::Mat tmp = rgb.clone();



		ProgressWindow::getInstance()->updateStatus(std::string("Height/Depth correlation"), 1.0 * imgNr / tSet.getNumberOfImages(), "Iterating true positives of training set (" + std::to_string(imgNr) + ")");

		for (auto& r : regions) {
			if (r.regionClass == -1) {
				// correlate the region height with the average depth

				int xOffset = r.region.x + r.region.width / 2;
				double depthSum = 0;
				int depthCount = 0;
				for (int y = r.region.y; y < r.region.y + r.region.height; y++)
				{
					for (int i = xOffset - 1; i <= xOffset; i++)
					{
						depthSum += depth.at<float>(y, i);
						depthCount++;
					}
				}
				double depthAvg = depthSum / depthCount;
				str << r.region.height << ";" << depthAvg << std::endl;
			}
		}
	});

	str.flush();
}

void generateFinalForEachRound(FeatureTester* tester, EvaluationSettings& settings) {
	std::set<std::string> set = { "HOG(RGB)", "HDD" };

	auto featureSet = tester->getFeatureSet(set);

	KITTIDataSet dataSet(kittiDatasetPath);

	int nrOfEvaluations = 300;
	std::function<bool(int)> testCriteria = [](int imageNumber) -> bool { return imageNumber % 20 == 1; };

	for (int i = 0; i <= 2; i++)
	{

		std::string featureSetName = std::string("HDD+HOG(RGB)_" + std::to_string(i));
		EvaluatorCascade cascade(featureSetName);
		cascade.load(std::string("models\\HDD+HOG(RGB)_cascade_" + std::to_string(i) + ".xml"), std::string("models"));
		std::cout << "Started final evaluation on test set with sliding window and NMS " << std::endl;
		FinalEvaluationSlidingWindowResult finalresult;
		long elapsedEvaluationSlidingTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			finalresult = cascade.evaluateWithSlidingWindowAndNMS(settings, &dataSet, *featureSet, testCriteria);
		});
		std::cout << "Evaluation with sliding window and NMS complete after " << elapsedEvaluationSlidingTime << "ms" << std::endl;

		std::string finalEvaluationSlidingFile = "results\\" + std::string("HDD+HOG(RGB)_") + std::to_string(i) + ".csv";

		std::ofstream str = std::ofstream(finalEvaluationSlidingFile);
		str << "Name" << ";";
		ClassifierEvaluation().toCSVLine(str, true);
		str << std::endl;
		for (auto& result : finalresult.evaluations["easy"]) {
			str << featureSetName << "[S][E]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.evaluations["moderate"]) {
			str << featureSetName << "[S][M]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.evaluations["hard"]) {
			str << featureSetName << "[S][H]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}

		for (auto& result : finalresult.combinedEvaluations) {
			str << featureSetName << "[S]" << ";";
			result.toCSVLine(str, false);
			str << std::endl;
		}
	}
}


void drawRiskOnDepthDataSet(DataSet* set, bool dumpHeightsPerCategory = false) {

	if (!set->getFullfillsRequirements()[1])
		throw new std::exception("The dataset does not have the required depth information");

	float imgHeight = 400; // px
	float imgWidth = 800; //px
	float max_depth = 80.0; // m
	float tireroadFriction = 0.7;
	float gravity = 9.81; // m/s²

	float max_pedestrian_speed = 0 * 1000 / 3600.0; // m/s
	float vehicleSpeedKmH = 50; // km/h
	float t = 1;
	float max_seconds_remaining = 10;


	std::vector<std::vector<DataSetLabel>> labelsPerNumber = set->getLabelsPerNumber();

	std::map<std::string, std::vector<cv::Rect2d>> truePositivesPerCategory;
	for (auto& c : RiskAnalysis::getRiskCategories()) {
		truePositivesPerCategory[c] = std::vector<cv::Rect2d>();
	}

	for (int imgNr = 0; imgNr < set->getNrOfImages(); imgNr++)
	{
		auto& labels = labelsPerNumber[imgNr];
		if (labels.size() > 0) {
			auto imgs = set->getImagesForNumber(imgNr);

			for (auto& l : labels) {
				if (!l.isDontCareArea()) {
					std::string category = RiskAnalysis::getRiskCategory(l.z_3d, l.x_3d, vehicleSpeedKmH, tireroadFriction);
					truePositivesPerCategory[category].push_back(l.getBbox());
				}
			}

			cv::Mat topdown = RiskAnalysis::getTopDownImage(imgWidth, imgHeight, labels, max_depth, vehicleSpeedKmH, tireroadFriction);
			cv::Mat img = RiskAnalysis::getRGBImage(imgs[0], imgs[1], labels, 50, tireroadFriction);

			cv::imshow("TopDown", topdown);
			cv::imshow("RGB", img);
			cv::waitKey(0);
		}
	}

	if (dumpHeightsPerCategory) {
		for (auto& pair : truePositivesPerCategory) {

			double minHeight = std::numeric_limits<int>().max();
			double maxHeight = 0;

			std::ofstream str(pair.first + "_bboxheight.csv");

			for (auto& bbox : pair.second) {
				if (bbox.height < minHeight) minHeight = bbox.height;
				if (bbox.height > maxHeight) maxHeight = bbox.height;

				str << bbox.height << std::endl;
			}
			str.close();
			std::cout << "Bounding box range for category " << pair.first << ": " << minHeight << " ~ " << maxHeight << std::endl;

			std::cout.flush();
		}
	}
}




void explainModel(FeatureTester& tester, EvaluationSettings& settings) {



	std::set<std::string> set;
	set = { "HOG(RGB)", "SDDG(Depth)" };

	std::string featureSetName("");
	for (auto& name : set) {
		if (name != *(set.begin()))
			featureSetName += "+" + name;
		else
			featureSetName += name;
	}

	auto fset = tester.getFeatureSet(set);

	EvaluatorCascade cascade(featureSetName);
	cascade.load(std::string("models\\KITTI_" + featureSetName + "_cascade.xml"), std::string("models"));

	std::vector<int> classifierHits(cascade.size(), 0);

	classifierHits = cascade.getClassifierHitCount(); // { 376310, 16947, 14280, 12743, 10857, 11272, 42062 };

	int classifierHitSum = 0;
	for (auto val : classifierHits) classifierHitSum += val;

	int rounds = cascade.size();
	int padding = 5;

	// don't initialize directly or it will point to the same data
	std::vector<cv::Mat> imgs;// (set.size(), cv::Mat(cv::Size(refWidth, refHeight * 4), CV_32FC1, cv::Scalar(0)));
	for (int i = 0; i < set.size(); i++)
		imgs.push_back(cv::Mat(cv::Size((settings.refWidth + padding)*rounds + 4 * padding + settings.refWidth, settings.refHeight), CV_32FC1, cv::Scalar(0)));


	std::vector<cv::Mat> totalImgs;
	for (int i = 0; i < set.size(); i++)
		totalImgs.push_back(cv::Mat(cv::Size(settings.refWidth, settings.refHeight), CV_32FC1, cv::Scalar(0)));

	cv::Mat rgb(128, 64, CV_8UC3, cv::Scalar(0));
	cv::Mat depth(128, 64, CV_32FC1, cv::Scalar(0));



	for (int i = 0; i < rounds; i++)
	{
		ModelEvaluator model = cascade.getModelEvaluator(i);
		//	model.loadModel(std::string("models\\" + featureSetName + " round " + std::to_string(i) + ".xml"));


		auto cur = model.explainModel(fset, settings.refWidth, settings.refHeight);

		for (int j = 0; j < cur.size(); j++) {

			if (cur[j].cols > 0 && cur[j].rows > 0) {
				cv::normalize(cur[j], cur[j], 0, 1, cv::NormTypes::NORM_MINMAX);

				totalImgs[j] += cur[j] * (1.0 * classifierHits[j] / classifierHitSum);


				cv::Mat& dst = imgs[j](cv::Rect(i*(settings.refWidth + padding), 0, settings.refWidth, settings.refHeight));
				cur[j].copyTo(dst);
			}


			//	cv::imshow("Test" + std::to_string(j), toHeatMap(cur[j]));
			//	cv::waitKey(0);
		}
	}


	auto it = set.begin();
	for (int i = 0; i < imgs.size(); i++) {
		cv::Mat img;

		img = imgs[i];
		//	cv::normalize(img, img, 0, 1, cv::NormTypes::NORM_MINMAX);
			//imgs[i].convertTo(img, CV_8UC1, 255);

		cv::imshow("explaingray_" + *it, img);
		img = heatmap::toHeatMap(img);


		cv::Mat totalimage = totalImgs[i];
		cv::normalize(totalimage, totalimage, 0, 1, cv::NormTypes::NORM_MINMAX);
		cv::Mat heatmapTotalImage = heatmap::toHeatMap(totalimage);
		heatmapTotalImage.copyTo(img(cv::Rect(imgs[i].cols - settings.refWidth, 0, settings.refWidth, settings.refHeight)));

		cv::imshow("explain_" + *it, img);



		cv::imshow("explaintotal_" + *it, heatmap::toHeatMap(totalimage));


		it++;
	}
	cv::waitKey(0);
}


void testSpeed(FeatureTester& tester, DataSet* dataSet) {

	auto imgs = dataSet->getImagesForNumber(0);
	cv::Mat rgbScale = imgs[0];
	cv::Mat depthScale = imgs[1];
	cv::Mat thermalScale = depthScale; // temporarily just to evaluate speed

	cv::Rect bbox(64, 128, 64, 128);

	cv::Mat regionRGB;
	if (rgbScale.cols > 0 && rgbScale.rows > 0)
		regionRGB = rgbScale(bbox);

	cv::Mat regionDepth;
	if (depthScale.cols > 0 && depthScale.rows > 0)
		regionDepth = depthScale(bbox);

	cv::Mat regionThermal = regionDepth; // temporarily just to evaluate speed
	/*if (thermalScale.cols > 0 && thermalScale.rows > 0)
		regionThermal = thermalScale(bbox);*/


	std::ofstream str("speed.csv");
	str << "Name;PreparationTime;WindowEvaluationTime" << std::endl;

	if (fileExists("inputsets.txt")) {
		std::ifstream istr("inputsets.txt");
		std::string line;
		while (std::getline(istr, line)) {

			if (line.length() > 0 && line[0] != '#') {
				std::set<std::string> set;
				auto parts = splitString(line, '+');
				for (auto& p : parts)
					set.emplace(p);

				auto fset = tester.getFeatureSet(set);

				std::vector<std::unique_ptr<IPreparedData>> preparedData;
				int n = 100;
				long preparationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
					for (int i = 0; i < n; i++)
					{
						preparedData = fset->buildPreparedDataForFeatures(rgbScale, depthScale, thermalScale);
					}
				});

				double avgPreparationTime = 1.0 * preparationTime / n;

				n = 1000;
				long evaluationTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
					for (int i = 0; i < n; i++)
					{
						fset->getFeatures(regionRGB,regionDepth, regionThermal, bbox, preparedData);
					}
				});

				double avgEvaluationTime = 1.0 * evaluationTime / n;

				str << fset->getFeatureSetName() << ";" << avgPreparationTime << ";" << avgEvaluationTime << std::endl;
				std::cout << fset->getFeatureSetName() << ": prep time: " << avgPreparationTime << "ms, window eval time: " << avgEvaluationTime << "ms" << std::endl;
			}
		}
		istr.close();
	}
	else {
		std::cout << "Input sets file does not exist" << std::endl;
	}
}


void buildTesterJobsFromInputSets(FeatureTester& tester, DataSet* dataSet, EvaluationSettings& settings) {



	tester.nrOfConcurrentJobs = 4;
	if (fileExists("inputsets.txt")) {
		std::ifstream istr("inputsets.txt");
		std::string line;
		while (std::getline(istr, line)) {

			if (line.length() > 0 && line[0] != '#') {
				std::set<std::string> set;
				auto parts = splitString(line, '+');
				for (auto& p : parts)
					set.emplace(p);

				tester.addJob(set, dataSet, settings);
			}
		}
		istr.close();
	}
	else {
		std::cout << "Input sets file does not exist" << std::endl;
	}


	// --------- Build initial training set file if it does not exist ----------
	std::string initialTrain0File = std::string("trainingsets") + PATH_SEPARATOR + dataSet->getName() + "_" + "train0.txt";
	if (!fileExists(initialTrain0File)) {
		std::cout << "The initial train0 file does not exist, building training set";
		TrainingDataSet initialSet = buildInitialTrainingSet(settings, dataSet);
		initialSet.save(initialTrain0File);
	}

}


void evaluateClusterSize(FeatureTester& tester, EvaluationSettings settings) {

	KITTIDataSet dataSet(kittiDatasetPath);

	std::vector<std::string> clustersToTest = { "ORB(RGB)", "SIFT(RGB)", "CenSurE(RGB)", "MSD(RGB)", "FAST(RGB)" };
	std::set<std::string> set;

	settings.nrOfTrainingRounds = 1;

	for (int k = 10; k <= 100; k += 10)
	{
		tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, k, IFeatureCreator::Target::RGB))); }));
		set = { std::string("ORB(RGB)_") + std::to_string(k) };
		tester.addJob(set, &dataSet, settings);
	}
	for (int k = 10; k <= 100; k += 10)
	{
		tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, k, IFeatureCreator::Target::RGB))); }));
		set = { std::string("SIFT(RGB)_") + std::to_string(k) };
		tester.addJob(set, &dataSet, settings);
	}
	for (int k = 10; k <= 100; k += 10)
	{
		tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, k, IFeatureCreator::Target::RGB))); }));
		set = { std::string("CenSurE(RGB)_") + std::to_string(k) };
		tester.addJob(set, &dataSet, settings);
	}
	for (int k = 10; k <= 100; k += 10)
	{
		tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, k, IFeatureCreator::Target::RGB))); }));
		set = { std::string("MSD(RGB)_") + std::to_string(k) };
		tester.addJob(set, &dataSet, settings);
	}
	for (int k = 10; k <= 100; k += 10)
	{
		tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, k, IFeatureCreator::Target::RGB))); }));
		set = { std::string("FAST(RGB)_") + std::to_string(k) };
		tester.addJob(set, &dataSet, settings);
	}
	tester.runJobs();
}


void testKAISTROI(EvaluationSettings& settings) {
	KAISTDataSet kaist(settings.kaistDataSetPath);
	auto labelsPerNumber = kaist.getLabelsPerNumber();

	for (int i = 0; i < kaist.getNrOfImages(); i++)
	{
		bool show = false;
		for (auto& l : labelsPerNumber[i]) {
			if (!l.isDontCareArea()) {
				show = true;
				break;
			}
		}
		if (show) {

			double w = 12;
			double alpha = 2;
			//double beta = 8;


			auto imgs = kaist.getImagesForNumber(i);
			imgs[2].convertTo(imgs[2], CV_8UC1, 255);


			std::vector<float> scales = { 0.5, 0.75, 1 };

			std::vector<cv::Rect> candidates;

			for (auto& scale : scales) {

				cv::Mat mThermal;
				cv::resize(imgs[2], mThermal, cv::Size(imgs[2].cols * scale, imgs[2].rows * scale));

				cv::Mat mRGB;
				cv::resize(imgs[0], mRGB, cv::Size(imgs[0].cols * scale, imgs[0].rows * scale));

				cv::Mat dest = mThermal.clone();
				for (int j = 0; j < mThermal.rows; j++)
				{
					for (int i = 0; i < mThermal.cols; i++)
					{
						char value = mThermal.at<char>(j, i);

						double tl = 0;

						int count = 0;
						for (int x = i - w; x <= i + w; x++) {
							if (x >= 0 && x < mThermal.cols) {
								tl += mThermal.at<char>(j, x);
								count++;
							}
						}
						tl = tl / count + alpha;

						double t3 = max(1.06 * (tl - alpha), tl + 2);
						double t2 = min(t3, tl + 8);
						double t1 = min(t2, 230.0);
						double th = max(t1, tl);
						//  let th = tl + beta;


						if (value > th)
							dest.at<char>(j, i) = 255;
						else if (value < tl)
							dest.at<char>(j, i) = 0;

						else if (i - 1 >= 0 && dest.at<char>(j, i - 1) > 0)
							dest.at<char>(j, i) = 255;
						else
							dest.at<char>(j, i) = 0;
					}
				}


				cv::dilate(dest, dest, cv::Mat());
				cv::erode(dest, dest, cv::Mat());

				std::vector<std::vector<cv::Point> > contours;
				std::vector<cv::Rect> boundRect(contours.size());
				cv::findContours(dest, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				for (int i = 0; i < contours.size(); i++)
				{
					int minX = std::numeric_limits<int>().max();
					int maxX = std::numeric_limits<int>().min();

					int minY = std::numeric_limits<int>().max();
					int maxY = std::numeric_limits<int>().min();
					for (auto& p : contours[i]) {
						if (p.x > maxX) maxX = p.x;
						if (p.x < minX) minX = p.x;
						if (p.y > maxY) maxY = p.y;
						if (p.y < minY) minY = p.y;
					}

					cv::Rect r = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);

					if (1.0 * r.height / r.width >= 1 && 1.0 * r.height / r.width < 6 && r.width >= 8 && r.height >= 16) {
						candidates.push_back(cv::Rect(r.x / scale, r.y / scale, r.width / scale, r.height / scale));
					}
				}
			}


			cv::Mat mThermal = imgs[2].clone();
			cv::Mat mRGB = imgs[0].clone();


			for (auto& r : candidates) {
				cv::rectangle(mThermal, r, cv::Scalar(255), 1);
				cv::rectangle(mRGB, r, cv::Scalar(255, 255, 255), 1);
			}

			for (DataSetLabel& l : labelsPerNumber[i]) {
				if (!l.isDontCareArea()) {
					cv::rectangle(mThermal, l.getBbox(), cv::Scalar(127), 1);
					cv::rectangle(mRGB, l.getBbox(), cv::Scalar(0, 255, 0), 1);

				}
			}


			cv::imshow("Thermal", mThermal);
			cv::imshow("RGB", mRGB);


			cv::waitKey(0);

		}
	}

}

int main()
{
	EvaluationSettings settings;
	settings.read(std::string("settings.ini"));

	FeatureTester tester;
	tester.nrOfConcurrentJobs = 4;

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOG(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, IFeatureCreator::Target::RGB, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("S2HOG(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGHistogramVarianceFeatureCreator(name, IFeatureCreator::Target::RGB, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOG(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, IFeatureCreator::Target::Depth, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOG(Thermal)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, IFeatureCreator::Target::Thermal, patchSize, binSize, settings.refWidth, settings.refHeight))); }));

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, IFeatureCreator::Target::RGB))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, IFeatureCreator::Target::Depth))); }));

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Histogram(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HistogramDepthFeatureCreator(name))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, IFeatureCreator::Target::RGB))); }));
	///tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 90, IFeatureCreator::Target::RGB))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 10, IFeatureCreator::Target::RGB))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 30, IFeatureCreator::Target::RGB))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, IFeatureCreator::Target::RGB))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 90, IFeatureCreator::Target::RGB))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HDD"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(name, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("LBP(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(name, patchSize, 20, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HONV"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HONVFeatureCreator(name, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CoOccurrence(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CoOccurenceMatrixFeatureCreator(name, patchSize, 8))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("RAW(RGB)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new RAWRGBFeatureCreator(name, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOI"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOIFeatureCreator(name, patchSize, binSize, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SDDG(Thermal)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SDDGFeatureCreator(name, IFeatureCreator::Target::Thermal, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SDDG(Depth)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SDDGFeatureCreator(name, IFeatureCreator::Target::Depth, settings.refWidth, settings.refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("RAW(LUV)"), [&](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new RAWLUVFeatureCreator(name, settings.refWidth, settings.refHeight))); }));

	//for (auto& f : tester.getFeatureCreatorFactories()) {
	//	std::set<std::string> set = { f };
	//	auto fset = tester.getFeatureSet(set);
	//	std::cout << f << " " << fset->getNumberOfFeatures() << std::endl;
	//}
	// show progress window
	ProgressWindow* wnd = ProgressWindow::getInstance();
	wnd->run();

	//	testKAISTROI(settings);
	//testSlidingWindow(settings);

	/*cv::Mat testImage;
	testImage = cv::imread("D:\\test.jpg");

	LBPFeatureCreator hogcreator(std::string("HOG"), IFeatureCreator::Target::RGB);
	auto result = hogcreator.getFeatures((testImage, patchSize, binSize, nullptr, true, false);
	cv::Mat imgResult = result.combineHOGImage(testImage);
	cv::imshow("HOG", imgResult);

	auto preparedData = hogcreator.buildPreparedDataForFeatures(testImage, cv::Mat(), cv::Mat());
	auto result2 = hogcreator.getHistogramsOfOrientedGradient(testImage, patchSize, binSize, preparedData, true, false);
	cv::Mat imgResult2 = result2.combineHOGImage(testImage);
	cv::imshow("HOG2", imgResult2);
	cv::waitKey(0);*/


	//KITTIDataSet kittiDataSet(settings.kittiDataSetPath);
	//drawRiskOnDepthDataSet(&kittiDataSet);


	//std::set<std::string> set = { "SDDG" };
	//settings.trainingCriteria = [=](int imageNumber) -> bool { return imageNumber % 20 == 0; };
	//settings.testCriteria = [=](int imageNumber) -> bool { return imageNumber % 20 == 1; };

	//KAISTDataSet kaistDataSet(settings.kaistDataSetPath);
	//tester.addJob(set, &kaistDataSet,settings);
	//tester.runJobs();


	//KITTIDataSet kittiDataSet(settings.kittiDataSetPath);
	//browseThroughTrainingSet(std::string("trainingsets\\KITTI_HDD+HOG(RGB)_train1.txt"), &kittiDataSet);


	//explainModel(tester, settings);
	/*KITTIDataSet kittiDataSet(settings.kittiDataSetPath);
	testSpeed(tester, &kittiDataSet);
*/

	//testClassifier(tester, settings);


	if (settings.kittiDataSetPath != "") {
		KITTIDataSet dataSet(settings.kittiDataSetPath);
		buildTesterJobsFromInputSets(tester, &dataSet, settings);
		tester.runJobs();
	}


	if (settings.kaistDataSetPath != "") {
		KAISTDataSet dataSet(settings.kaistDataSetPath);
		buildTesterJobsFromInputSets(tester, &dataSet, settings);
		tester.runJobs();
	}



	//testFeature();


	//generateFinalForEachRound(&tester);

	//checkDistanceBetweenTPAndTN(std::string("trainingsets\\LBP(RGB)_train3.txt"), std::string("tptnsimilarity_lbp_train3.csv"));



	/*std::ofstream str("heightdepthvaluesTN.csv");
	printHeightVerticalAvgDepthRelation(std::string("trainingsets\\train0.txt"), str);*/

	//browseThroughDataSet(std::string("trainingsets\\train0.txt"));

	//	testSlidingWindow();


	//trainDetailedClassifier();
	//testFeature();

	std::cout << "--------------------- New console session -----------------------" << std::endl;

	getchar();
	return 0;
}
