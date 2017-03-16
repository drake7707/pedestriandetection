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

#include "ProgressWindow.h"

#include "Helper.h"
#include "TrainingDataSet.h"

#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"

#include "ModelEvaluator.h"
#include "IFeatureCreator.h"
#include "HoGFeatureCreator.h"
#include "HoGHistogramVarianceFeatureCreator.h"
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

#include "EvaluatorCascade.h"

#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"

#include "CoOccurenceMatrix.h"

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

bool overlaps(cv::Rect2d r, std::vector<cv::Rect2d> selectedRegions) {
	for (auto& region : selectedRegions) {
		if ((r & region).area() > 0)
			return true;
	}

	return false;
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
			if (r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {

				TrainingRegion tr;
				tr.region = l.getBbox();
				tr.regionClass = 1;
				tImg.regions.push_back(tr);
				selectedRegions.push_back(r);
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
			} while (iteration++ < 10000 && (overlaps(rTN, selectedRegions) || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= currentImages[0].cols || rTN.y + rTN.height >= currentImages[0].rows));


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


void testClassifier() {


	FeatureSet fset;
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(std::string("HDD"), patchSize, binSize, refWidth, refHeight)));
	//fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight)));
	//fset.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight)));

	EvaluatorCascade cascade(std::string("Test"));
	cascade.load(std::string("models\\HDD+HoG(RGB)_cascade.xml"), std::string("models"));

	//ModelEvaluator modelFinal(std::string("Test"));
	//modelFinal.loadModel(std::string("models\\HoG(Depth)+HoG(RGB)+LBP(RGB) round 3.xml"));

	cascade.updateLastModelValueShift(1.44);

	/*ClassifierEvaluation eval = model.evaluateDataSet(1, false)[0];
	eval.print(std::cout);*/

	KITTIDataSet dataSet(kittiDatasetPath);

	cv::namedWindow("Test");

	auto& entries = cascade.getEntries();
	double valueShift = entries[entries.size() - 1].valueShift;

	int nr = 0;
	while (true) {

		auto imgs = dataSet.getImagesForNumber(nr);
		cv::Mat mRGB = imgs[0];
		cv::Mat mDepth = imgs[1];

		int nrOfWindowsEvaluated = 0;
		int nrOfWindowsSkipped = 0;
		long nrOfWindowsPositive = 0;
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



				if (mustContinue) {
					FeatureVector v = fset.getFeatures(regionRGB, regionDepth);
					double result = cascade.evaluateFeatures(v);
					if ((result > 0 ? 1 : -1) == 1) {
						cv::rectangle(mRGB, bbox, cv::Scalar(0, 255, 0), 2);
						nrOfWindowsPositive++;
					}
				}
				nrOfWindowsEvaluated++;
			}, 0.5, 4, 16);

		});

		cv::imshow("Test", mRGB);
		double posPercentage = 100.0 * nrOfWindowsPositive / (nrOfWindowsEvaluated - nrOfWindowsSkipped);
		std::cout << "Number of windows evaluated: " << nrOfWindowsEvaluated << " (skipped " << nrOfWindowsSkipped << ") and " << nrOfWindowsPositive << " positive (" << std::setw(2) << posPercentage << ") " << slidingWindowTime << "ms" << std::endl;
		// this will leak because creators are never disposed!
		cv::waitKey(0);
		nr++;
	}
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

			auto matrix = getCoOccurenceMatrixOfPatch(hue, 16);
			cv::Mat m = createFullCoOccurrenceMatrixImage(img, cells, patchSize);
			cv::imshow(msg, m);




			//hog::HOGResult result =  hog::getHistogramsOfDepthDifferences(img, patchSize, binSize, true, true);
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

					auto& result = hog::get2DHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, 9, true, false);



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
								//auto& result = hog::getHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), padded, patchSize, 20, true, false);

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

					auto& result =  hog::getHistogramsOfX(magnitude, angle, patchSize, binSize, true, false);
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


					//auto& result = hog::getHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), padded, patchSize, binSize, true, false);
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
	tSet.iterateDataSetWithSlidingWindow([&](int idx) -> bool { return true; },
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
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight)));
	fset.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight)));
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
			truePositiveRegions.push_back(SlidingWindowRegion(imageNumber, region));
		}
		else {
			trueNegativeFeatures.push_back(v);
			trueNegativeRegions.push_back(SlidingWindowRegion(imageNumber, region));
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
			else
				cv::rectangle(tmp, r.region, cv::Scalar(0, 0, 255));
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

int main()
{
	//testClassifier();
	//testFeature();
	// show progress window
	ProgressWindow* wnd = ProgressWindow::getInstance();
	wnd->run();


	//checkDistanceBetweenTPAndTN(std::string("trainingsets\\LBP(RGB)_train3.txt"), std::string("tptnsimilarity_lbp_train3.csv"));

	//browseThroughDataSet(std::string("trainingsets\\LBP(RGB)_train4.txt"));
	//testClassifier();




	/*std::ofstream str("heightdepthvaluesTN.csv");
	printHeightVerticalAvgDepthRelation(std::string("trainingsets\\train0.txt"), str);*/
	//testClassifier();
	//browseThroughDataSet(std::string("trainingsets\\train0.txt"));
//	testSlidingWindow();

	/*TrainingDataSet testTrainingSet(kittiDatasetPath);
	testTrainingSet.load(std::string("trainingsets\\HoG(Depth)+HoG(RGB)+LBP(RGB)_train1.txt"));


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

	int nrOfEvaluations = 500;
	std::set<std::string> set;





	// --------- Build initial training set file if it does not exist ----------
	std::string initialTrain0File = std::string("trainingsets") + PATH_SEPARATOR + "train0.txt";
	if (!FileExists(initialTrain0File)) {
		std::cout << "The initial train0 file does not exist, building training set";
		TrainingDataSet initialSet = saveTNTP();
		initialSet.save(initialTrain0File);
	}

	FeatureTester tester;
	tester.nrOfConcurrentJobs = 4;

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HoG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("S2HoG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGHistogramVarianceFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HoG(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, true, patchSize, binSize, refWidth, refHeight))); }));
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
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("LBP(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(name, patchSize, 20, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HONV"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HONVFeatureCreator(name, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CoOccurrence(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CoOccurenceMatrixFeatureCreator(name, patchSize, 8))); }));


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

	set = { "CoOccurrence(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "CoOccurrence(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();


	set = { "LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(Depth)", "HoG(RGB)","LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HONV" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HDD" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(Depth)", "HoG(RGB)","LBP(RGB)", "HDD" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "HoG(Depth)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "HDD" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "HONV" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	set = { "HoG(RGB)", "S2HoG(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();



	/*set = { "HoG(Depth)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);

	set = { "LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);
	set = { "HDD" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);
	set = { "S2HoG(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 1);

	tester.runJobs();*/

	for (auto& name : tester.getFeatureCreatorFactories()) {
		set = { name };
		tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	}
	tester.runJobs();

	//// evaluate each creator combined with HOG(RGB)
	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HoG(RGB)") {
	//		set = { "HoG(RGB)", name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HoG(RGB)" && name != "HoG(Depth)") {
	//		set = { "HoG(Depth)", "HoG(RGB)", name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	if (name != "HoG(RGB)" && name != "HoG(Depth)" && name != "LBP(RGB)") {
	//		set = { "HoG(Depth)", "HoG(RGB)", "LBP(RGB)",  name };
	//		tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//	}
	//}
	//tester.runJobs();

	/*

		set = { "HoG(RGB)", "Corner(RGB)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HoG(RGB)","Histogram(Depth)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HoG(RGB)", "S2HoG(RGB)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HoG(RGB)", "HoG(Depth)" };
		tester.addJob(set, nrOfEvaluations);

		set = { "HoG(RGB)",  "S2HoG(RGB)",  "HoG(Depth)" };
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




























using namespace cv;
using namespace std;


Mat createSegmentationDisplay(Mat & segments, int numOfSegments, Mat & image)
{
	//create a new image
	Mat wshed(segments.size(), CV_8UC3);

	//Create color tab for coloring the segments
	vector<Vec3b> colorTab;
	for (int i = 0; i < numOfSegments; i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);

		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	//assign different color to different segments
	for (int i = 0; i < segments.rows; i++)
	{
		for (int j = 0; j < segments.cols; j++)
		{
			int index = segments.at<int>(i, j);
			if (index == -1)
				wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			else if (index <= 0 || index > numOfSegments)
				wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			else
				wshed.at<Vec3b>(i, j) = colorTab[index - 1];
		}
	}

	//If the original image available then merge with the colors of segments
	if (image.dims > 0)
		wshed = wshed*0.5 + image*0.5;

	return wshed;
}


/**
* This is an example method showing how to use this implementation.
*
* @param input The original image.
* @return wshedWithImage A merged image of the original and the segments.
*/
Mat watershedWithMarkers(Mat input) {

	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	for (int x = 0; x < input.rows; x++) {
		for (int y = 0; y < input.cols; y++) {
			if (input.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
				input.at<Vec3b>(x, y)[0] = 0;
				input.at<Vec3b>(x, y)[1] = 0;
				input.at<Vec3b>(x, y)[2] = 0;
			}
		}
	}
	// Show output image
	//imshow("Black Background Image", input);

	// Create a kernel that we will use for accuting/sharpening our image
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); // an approximation of second derivative, a quite strong kernel

				  // do the laplacian filtering as it is
				  // well, we need to convert everything in something more deeper then CV_8U
				  // because the kernel has some negative values,
				  // and we can expect in general to have a Laplacian image with negative values
				  // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
				  // so the possible negative number will be truncated
	Mat imgLaplacian;
	Mat sharp = input; // copy source image to another temporary one
	filter2D(sharp, imgLaplacian, CV_32F, kernel);
	input.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	//imshow( "New Sharped Image", imgResult );

	input = imgResult; // copy back
					   // Create binary image from source image
	Mat bw;
	cvtColor(input, bw, CV_BGR2GRAY);
	threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("Binary Image", bw);

	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, CV_DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 1., NORM_MINMAX);
	//imshow("Distance Transform Image", dist);

	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(dist, dist, kernel1);
	//imshow("Peaks", dist);

	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);

	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(dist.size(), CV_32SC1);

	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++)
		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);

	// Draw the background marker
	circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
	//imshow("Markers", markers*10000);

	// Perform the watershed algorithm
	watershed(input, markers);
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark);
	//imshow("Markers_v2", mark); // uncomment this if you want to see how the mark image looks like at that point

	int numOfSegments = contours.size();
	Mat wshed = createSegmentationDisplay(markers, numOfSegments, input);

	return wshed;
}


/*



*/