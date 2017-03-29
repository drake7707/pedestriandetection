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

#include "EvaluatorCascade.h"

#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"

#include "CoOccurenceMatrix.h"

#include "KITTIDataSet.h"
#include "DataSet.h"

#include "JetHeatMap.h"

std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
std::string baseDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti\\regions";

//std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";
//std::string baseDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti\\regions";

int patchSize = 8;
int binSize = 9;
int refWidth = 64;
int refHeight = 128;

std::vector<cv::Size> windowSizes = {
	/*cv::Size(24,48),*/
	cv::Size(32,64),
	cv::Size(48,96),
	cv::Size(64,128),
	cv::Size(80,160),
	cv::Size(96,192),
	cv::Size(104,208),
	cv::Size(112,224),
	cv::Size(120,240),
	cv::Size(128,256)
};


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

TrainingDataSet saveTNTP() {


	TrainingDataSet trainingSet(kittiDatasetPath);

	KITTIDataSet dataSet = KITTIDataSet(kittiDatasetPath);

	srand(7707);

	ProgressWindow::getInstance()->updateStatus(std::string("Initial training set (train0)"), 0, "Loading KITTI dataset labels");

	std::vector<DataSetLabel> labels = dataSet.getLabels();

	std::vector<std::vector<DataSetLabel>> labelsPerNumber(dataSet.getNrOfImages(), std::vector<DataSetLabel>());
	for (auto& l : labels)
		labelsPerNumber[atoi(l.getNumber().c_str())].push_back(l);

	int sizeVariance = 8; // from 0.25 to 2 times the refWidth and refHeight ( so anything between 16x32 - 32x64 - 64x128 - 128x256, more scales might be evaluated later )
	int nrOfTNPerImage = 2;

	for (int i = 0; i < labelsPerNumber.size(); i++)
	{
		TrainingImage tImg;
		tImg.number = i;


		std::vector<cv::Mat> currentImages = dataSet.getImagesForNumber(i);
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
				double width = refWidth * sizeMultiplier;
				double height = refHeight * sizeMultiplier;
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

//
//void nms(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& resRects, float thresh) {
//	resRects.clear();
//
//	const size_t size = srcRects.size();
//	if (!size)
//	{
//		return;
//	}
//
//	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
//	std::multimap<int, size_t> idxs;
//	for (size_t i = 0; i < size; ++i)
//	{
//		idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
//	}
//
//	// keep looping while some indexes still remain in the indexes list
//	while (idxs.size() > 0)
//	{
//		// grab the last rectangle
//		auto lastElem = --std::end(idxs);
//		const cv::Rect& rect1 = srcRects[lastElem->second];
//
//		resRects.push_back(rect1);
//
//		idxs.erase(lastElem);
//
//		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
//		{
//			// grab the current rectangle
//			const cv::Rect& rect2 = srcRects[pos->second];
//
//			float intArea = (rect1 & rect2).area();
//			float unionArea = rect1.area() + rect2.area() - intArea;
//			float overlap = intArea / unionArea;
//
//			// if there is sufficient overlap, suppress the current bounding box
//			if (overlap > thresh)
//			{
//				pos = idxs.erase(pos);
//			}
//			else
//			{
//				++pos;
//			}
//		}
//	}
//}
//



void testClassifier(FeatureTester& tester) {

	std::set<std::string> set = { "HDD", "HOG(RGB)" };

	auto fset = tester.getFeatureSet(set);

	EvaluatorCascade cascade(std::string("Test"));
	cascade.load(std::string("models\\HDD+HOG(RGB)_cascade.xml"), std::string("models"));

	double valueShift = -9.4;


	//ModelEvaluator modelFinal(std::string("Test"));
	//modelFinal.loadModel(std::string("models\\HOG(Depth)+HOG(RGB)+LBP(RGB) round 3.xml"));



	/*ClassifierEvaluation eval = model.evaluateDataSet(1, false)[0];
	eval.print(std::cout);*/

	KITTIDataSet dataSet(kittiDatasetPath);

	//	cv::namedWindow("Test");

	auto& entries = cascade.getEntries();

	std::mutex m;
	int nr = 0;
	//while (true) {

	std::vector<DataSetLabel> labels = dataSet.getLabels();

	std::vector<std::vector<DataSetLabel>> labelsPerNumber(dataSet.getNrOfImages(), std::vector<DataSetLabel>());
	for (auto& l : labels)
		labelsPerNumber[atoi(l.getNumber().c_str())].push_back(l);


	ClassifierEvaluation eval(dataSet.getNrOfImages());

	parallel_for(0, 1000, 6, [&](int i) -> void {
		ProgressWindow::getInstance()->updateStatus(std::string("Testing classifier"), 1.0 * i / 1000, std::to_string(i));

		auto imgs = dataSet.getImagesForNumber(i);
		cv::Mat mRGB = imgs[0];
		cv::Mat mDepth = imgs[1];

		int nrOfWindowsEvaluated = 0;
		int nrOfWindowsSkipped = 0;
		long nrOfWindowsPositive = 0;


		std::vector<SlidingWindowRegion> predictedPositiveRegions;

		std::vector<SlidingWindowRegion> truepositiveregions;
		std::vector<SlidingWindowRegion> falsepositiveregions;


		long slidingWindowTime = measure<std::chrono::milliseconds>::execution([&]() -> void {



			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {
				cv::Mat regionRGB;
				cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));

				cv::Mat regionDepth;
				cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));

				bool mustContinue = true;



				double depthSum = 0;
				int depthCount = 0;
				int xOffset = bbox.x + bbox.width / 2;
				for (int y = bbox.y; y < bbox.y + bbox.height; y++)
				{
					for (int i = xOffset - 1; i <= xOffset + 1; i++)
					{
						float depth = mDepth.at<float>(y, i);
						depthSum += depth;
						depthCount++;
					}
				}
				double depthAvg = (depthSum / depthCount);

				if (dataSet.isWithinValidDepthRange(bbox.height, depthAvg)) {
					// falls within range where TP can lie, continue
				}
				else {
					// reject outright, will most likely never be TP
					mustContinue = false;
					nrOfWindowsSkipped++;
				}

				double result;
				bool predictedPositive = false;
				if (mustContinue) {
					FeatureVector v = fset->getFeatures(regionRGB, regionDepth);
					result = cascade.evaluateFeatures(v);
					if ((result + valueShift > 0 ? 1 : -1) == 1) {
						nrOfWindowsPositive++;
						predictedPositive = true;
						predictedPositiveRegions.push_back(SlidingWindowRegion(i, bbox, abs(result)));
					}
				}


				nrOfWindowsEvaluated++;
			}, windowSizes, 16);

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

		cv::Mat nms = mRGB.clone();



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


		for (auto& pos : falsepositiveregions) {
			cv::rectangle(mRGB, pos.bbox, cv::Scalar(0, 0, 255), 2);

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << pos.score;
			std::string s = stream.str();
			cv::rectangle(mRGB, cv::Rect(pos.bbox.x, pos.bbox.y - 10, pos.bbox.width - 5, 10), cv::Scalar(0, 0, 255), -1);
			cv::putText(mRGB, s, cv::Point(pos.bbox.x, pos.bbox.y - 2), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);
		}

		for (auto& pos : truepositiveregions) {
			cv::rectangle(mRGB, pos.bbox, cv::Scalar(0, 255, 0), 2);

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << pos.score;
			std::string s = stream.str();
			cv::rectangle(mRGB, cv::Rect(pos.bbox.x, pos.bbox.y - 10, pos.bbox.width - 5, 10), cv::Scalar(0, 255, 0), -1);
			cv::putText(mRGB, s, cv::Point(pos.bbox.x, pos.bbox.y - 2), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);
		}

		for (auto& l : labelsPerNumber[i]) {
			if (!l.isDontCareArea()) {

				std::string category = l.getCategory();
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
		std::string str = "FP: " + std::to_string(eval.falsePositivesPerImage[i]) + ", missed: " + std::to_string(nrMissed) + " #windows : " + std::to_string(nrOfWindowsEvaluated) + " (#skipped with depth check: " + std::to_string(nrOfWindowsSkipped) + "). Eval time: " + std::to_string(slidingWindowTime) + "ms " + "(decision shift : " + std::to_string(valueShift) + ")";
		cv::putText(mRGB, str, cv::Point(10, 10), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 0), 1, CV_AA);


		//	cv::imshow("Test", mRGB);
			//cv::imshow("TestNMS", nms);
		m.lock();
		double posPercentage = 100.0 * nrOfWindowsPositive / (nrOfWindowsEvaluated - nrOfWindowsSkipped);
		std::cout << "Image: " << i << " Number of windows evaluated: " << nrOfWindowsEvaluated << " (skipped " << nrOfWindowsSkipped << ") and " << nrOfWindowsPositive << " positive (" << std::setw(2) << posPercentage << "%) " << slidingWindowTime << "ms (value shift: " << valueShift << ")" << std::endl;
		// this will leak because creators are never disposed!
		m.unlock();

		cv::imwrite(std::to_string(i) + "_hddHOGrgb.png", mRGB);
		//cv::waitKey(0);
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

void testSlidingWindow() {

	TrainingDataSet tSet(kittiDatasetPath);
	tSet.load(std::string("trainingsets\\train0.txt"));

	cv::namedWindow("Test");

	int oldNumber = -1;
	cv::Mat tmp;


	float minScaleReduction = 0.5;
	float maxScaleReduction = 4; // this is excluded, so 64x128 windows will be at most scaled to 32x64 with 4, or 16x32 with 8
	int baseWindowStride = 16;

	tSet.getDataSet()->iterateDataSetWithSlidingWindow(windowSizes, baseWindowStride, refWidth, refHeight,

		[&](int idx) -> bool { return true; },

		[&](int imageNumber) -> void {
		// start of image
	},
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (imageNumber != oldNumber) {
			if (tmp.rows > 0) {
				cv::imshow("Test", tmp);
				cv::waitKey(0);
			}
			tmp = fullrgb.clone();
			oldNumber = imageNumber;
		}


		if (resultClass == 1)
			cv::rectangle(tmp, region, cv::Scalar(0, 255, 0), 3);
		else {
			if (!overlapsWithTruePositive)
				cv::rectangle(tmp, region, cv::Scalar(0, 0, 255), 1);
		}
	}, [&](int imageNumber, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositives) -> void {
		// end of image
	}, 1);
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

void checkDistanceBetweenTPAndTN(std::string& trainingFile, std::string& outputFile) {
	TrainingDataSet tSet(kittiDatasetPath);
	tSet.load(trainingFile);

	FeatureSet fset;
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HOG(Depth)"), true, patchSize, binSize, refWidth, refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HOG(RGB)"), false, patchSize, binSize, refWidth, refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), false, patchSize, 20, refWidth, refHeight)));
	//fset.addCreator(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(std::string("HDD"), patchSize, binSize, refWidth, refHeight)));


	std::vector<FeatureVector> truePositiveFeatures;
	std::vector<FeatureVector> trueNegativeFeatures;

	std::vector<SlidingWindowRegion> truePositiveRegions;
	std::vector<SlidingWindowRegion> trueNegativeRegions;

	tSet.iterateDataSet([&](int idx) -> bool { return (idx + 1) % 10 == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string("Min distance between TN/TP"), 1.0 * imageNumber / tSet.getNumberOfImages(), std::string("Building feature vectors (") + std::to_string(imageNumber) + ")");

		FeatureVector v = fset.getFeatures(rgb, depth);
		if (resultClass == 1) {
			truePositiveFeatures.push_back(v);
			truePositiveRegions.push_back(SlidingWindowRegion(imageNumber, region, 0));
		}
		else {
			trueNegativeFeatures.push_back(v);
			trueNegativeRegions.push_back(SlidingWindowRegion(imageNumber, region, 0));
		}
	});

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

void browseThroughDataSet(std::string& trainingFile) {
	TrainingDataSet tSet(kittiDatasetPath);
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
	TrainingDataSet tSet(kittiDatasetPath);
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

void generateFinalForEachRound(FeatureTester* tester) {
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
			finalresult = cascade.evaluateWithSlidingWindowAndNMS(windowSizes, &dataSet, *featureSet, nrOfEvaluations, testCriteria);

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





void explainModel(FeatureTester* tester) {



	std::set<std::string> set;
	set = { "S2HOG(RGB)" };

	std::string featureSetName("");
	for (auto& name : set) {
		if (name != *(set.begin()))
			featureSetName += "+" + name;
		else
			featureSetName += name;
	}

	auto fset = tester->getFeatureSet(set);

	EvaluatorCascade cascade(featureSetName);
	cascade.load(std::string("models\\" + featureSetName + "_cascade.xml"), std::string("models"));

	std::vector<int> classifierHits(cascade.size(), 0);

	classifierHits = cascade.getClassifierHitCount(); // { 376310, 16947, 14280, 12743, 10857, 11272, 42062 };

	int classifierHitSum = 0;
	for (auto val : classifierHits) classifierHitSum += val;

	int rounds = cascade.size();
	int padding = 5;

	// don't initialize directly or it will point to the same data
	std::vector<cv::Mat> imgs;// (set.size(), cv::Mat(cv::Size(refWidth, refHeight * 4), CV_32FC1, cv::Scalar(0)));
	for (int i = 0; i < set.size(); i++)
		imgs.push_back(cv::Mat(cv::Size((refWidth + padding)*rounds + 4 * padding + refWidth, refHeight), CV_32FC1, cv::Scalar(0)));


	std::vector<cv::Mat> totalImgs;
	for (int i = 0; i < set.size(); i++)
		totalImgs.push_back(cv::Mat(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0)));

	cv::Mat rgb(128, 64, CV_8UC3, cv::Scalar(0));
	cv::Mat depth(128, 64, CV_32FC1, cv::Scalar(0));



	for (int i = 0; i < rounds; i++)
	{
		ModelEvaluator model(featureSetName);
		model.loadModel(std::string("models\\" + featureSetName + " round " + std::to_string(i) + ".xml"));
		fset->getFeatures(rgb, depth);

		auto cur = model.explainModel(fset, refWidth, refHeight);

		for (int j = 0; j < cur.size(); j++) {

			cv::normalize(cur[j], cur[j], 0, 1, cv::NormTypes::NORM_MINMAX);

			totalImgs[j] += cur[j] * (1.0 * classifierHits[j] / classifierHitSum);


			cv::Mat& dst = imgs[j](cv::Rect(i*(refWidth + padding), 0, refWidth, refHeight));
			cur[j].copyTo(dst);



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
		heatmapTotalImage.copyTo(img(cv::Rect(imgs[i].cols - refWidth, 0, refWidth, refHeight)));

		cv::imshow("explain_" + *it, img);



		cv::imshow("explaintotal_" + *it, heatmap::toHeatMap(totalimage));


		it++;
	}
	cv::waitKey(0);
}

int main()
{

	/*std::vector<std::pair<float, SlidingWindowRegion>> testregions;
	testregions.push_back(std::pair<float, SlidingWindowRegion>(1, SlidingWindowRegion(0, cv::Rect(50, 50, 100, 100))));
	testregions.push_back(std::pair<float, SlidingWindowRegion>(2, SlidingWindowRegion(0, cv::Rect(58, 50, 100, 100))));
	testregions.push_back(std::pair<float, SlidingWindowRegion>(1.5, SlidingWindowRegion(0, cv::Rect(66, 50, 80, 80))));
	testregions.push_back(std::pair<float, SlidingWindowRegion>(0, SlidingWindowRegion(0, cv::Rect(46, 30, 150, 150))));


	auto result = applyNonMaximumSuppression(testregions);

	for (auto& r : result) {
		std::cout << r.first << " - " << r.second.bbox.x << "," << r.second.bbox.y << " " << r.second.bbox.width << "x" << r.second.bbox.height << std::endl;
	}
*/
	FeatureTester tester;
	tester.nrOfConcurrentJobs = 4;

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("S2HOG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGHistogramVarianceFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HOG(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, true, patchSize, binSize, refWidth, refHeight))); }));

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, true))); }));

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Histogram(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HistogramDepthFeatureCreator(name))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, false))); }));
	///tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 90, false))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 10, false))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 30, false))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, false))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 90, false))); }));
	//tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HDD"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(name, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("LBP(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(name, false, patchSize, 20, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HONV"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HONVFeatureCreator(name, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CoOccurrence(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CoOccurenceMatrixFeatureCreator(name, patchSize, 8))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("RAW(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new RAWRGBFeatureCreator(name, refWidth, refHeight))); }));



	//testClassifier(tester);
	//testFeature();
	// show progress window
	ProgressWindow* wnd = ProgressWindow::getInstance();
	wnd->run();

	explainModel(&tester);

	//generateFinalForEachRound(&tester);

	//checkDistanceBetweenTPAndTN(std::string("trainingsets\\LBP(RGB)_train3.txt"), std::string("tptnsimilarity_lbp_train3.csv"));

	//browseThroughDataSet(std::string("trainingsets\\train0.txt"));




	/*std::ofstream str("heightdepthvaluesTN.csv");
	printHeightVerticalAvgDepthRelation(std::string("trainingsets\\train0.txt"), str);*/

	//browseThroughDataSet(std::string("trainingsets\\train0.txt"));
//	testSlidingWindow();

	/*TrainingDataSet testTrainingSet(kittiDatasetPath);
	testTrainingSet.load(std::string("trainingsets\\HOG(Depth)+HOG(RGB)+LBP(RGB)_train1.txt"));


	testTrainingSet.iterateDataSet([](int idx) -> bool { return true; }, [&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {
		if(resultClass == 1)
			cv::imshow("TP", rgb);
		else
			cv::imshow("TN", rgb);

		cv::waitKey(0);

	});*/
	//trainDetailedClassifier();
	//testFeature();





	//testFeature();
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	//testClassifier();
	//saveTNTP();
	//return 0;

	int nrOfEvaluations = 300;






	// --------- Build initial training set file if it does not exist ----------
	std::string initialTrain0File = std::string("trainingsets") + PATH_SEPARATOR + "train0.txt";
	if (!FileExists(initialTrain0File)) {
		std::cout << "The initial train0 file does not exist, building training set";
		TrainingDataSet initialSet = saveTNTP();
		initialSet.save(initialTrain0File);
	}



	//std::vector<std::string> clustersToTest = { "ORB(RGB)", "SIFT(RGB)", "CenSurE(RGB)", "MSD(RGB)", "FAST(RGB)" };
	//for (int k = 10; k <= 100; k += 10)
	//{
	//	tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, k, false))); }));
	//	set = { std::string("ORB(RGB)_") + std::to_string(k) };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//for (int k = 10; k <= 100; k += 10)
	//{
	//	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, k, false))); }));
	//	set = { std::string("SIFT(RGB)_") + std::to_string(k) };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//for (int k = 10; k <= 100; k += 10)
	//{
	//	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, k, false))); }));
	//	set = { std::string("CenSurE(RGB)_") + std::to_string(k) };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//for (int k = 10; k <= 100; k += 10)
	//{
	//	tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, k, false))); }));
	//	set = { std::string("MSD(RGB)_") + std::to_string(k) };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//for (int k = 10; k <= 100; k += 10)
	//{
	//	tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(RGB)_") + std::to_string(k), [=](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, k, false))); }));
	//	set = { std::string("FAST(RGB)_") + std::to_string(k) };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//tester.runJobs();
	//return 0;


	//evaluate each creator individually, but don't do a sliding window evaluation yet
	//tester.nrOfConcurrentJobs = 6;
	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	set = { name };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1, false);
	//}
	//tester.runJobs();

	tester.nrOfConcurrentJobs = 1;

	std::set<std::string> set;
	set = { "HOG(RGB)", "HDD" };
	tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 8);
	tester.runJobs();
	//
	//set = { "HOG(RGB)", "RAW(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)", "HOG(Depth)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)", "HONV" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)", "LBP(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)", "S2HOG(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)", "CoOccurrence(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);
	//tester.runJobs();

	//set = { "CoOccurrence(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "LBP(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HONV" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HDD" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(Depth)", "HOG(RGB)","LBP(RGB)" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);

	//set = { "HOG(RGB)","LBP(RGB)", "HDD" };
	//tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);


	//tester.runJobs();

	/*set = { "HOG(Depth)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);

	set = { "LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);
	set = { "HDD" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);
	set = { "S2HOG(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);

	tester.runJobs();*/

	/*for (auto& name : tester.getFeatureCreatorFactories()) {
		set = { "HOG(RGB)", name };
		tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);
	}
	tester.runJobs();

	for (auto& name : tester.getFeatureCreatorFactories()) {
		set = { name };
		tester.addJob(set, windowSizes, kittiDatasetPath, nrOfEvaluations, 4);
	}
	tester.runJobs();

	*/

	//// evaluate each creator combined with HOG(RGB)
	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HOG(RGB)") {
	//		set = { "HOG(RGB)", name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HOG(RGB)" && name != "HOG(Depth)") {
	//		set = { "HOG(Depth)", "HOG(RGB)", name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HOG(RGB)" && name != "HOG(Depth)" && name != "LBP(RGB)") {
	//		set = { "HOG(Depth)", "HOG(RGB)", "LBP(RGB)",  name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	/*

		set = { "HOG(RGB)", "Corner(RGB)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HOG(RGB)","Histogram(Depth)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HOG(RGB)", "S2HOG(RGB)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HOG(RGB)", "HOG(Depth)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HOG(RGB)",  "S2HOG(RGB)",  "HOG(Depth)" };
		tester.addJob(set, nrOfEvaluations);

	*/







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


	getchar();
	return 0;
}
