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
	testSet.addCreator(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight));
	testSet.addCreator(new HOGHistogramVarianceFeatureCreator(std::string("S2HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight));
	testSet.addCreator(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight));

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


void testFeature() {
	int nr = 0;
	while (true) {
		char nrStr[7];
		sprintf(nrStr, "%06d", nr);

		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\rgb\\000000.png");// kittiDatasetPath + "\\regions\\tp\\depth" + std::to_string(nr) + ".png");
		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\regions\\tp\\rgb" + std::to_string(nr) + ".png");
		cv::Mat tp = cv::imread(kittiDatasetPath + "\\depth\\000000.png", CV_LOAD_IMAGE_ANYDEPTH);
		//cv::Mat tp = cv::imread("D:\\test.png", CV_LOAD_IMAGE_ANYDEPTH);
		tp.convertTo(tp, CV_32FC1, 1.0 / 0xFFFF, 0);

		cv::Mat tn = cv::imread(kittiDatasetPath + "\\regions\\tn\\depth" + std::to_string(nr) + ".png");

		std::function<void(cv::Mat&, std::string)> func = [&](cv::Mat& img, std::string msg) -> void {



			//hog::HOGResult result =  hog::getHistogramsOfDepthDifferences(img, patchSize, binSize, true, true);
			/*	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
			std::vector<cv::KeyPoint> keypoints;


			cv::Mat descriptors;
			detector->detect(img, keypoints);


			//detector->compute(img, keypoints, descriptors);

			cv::Mat imgKeypoints;
			cv::drawKeypoints(img, keypoints, imgKeypoints);
			*/

			cv::Mat depth;
			int d = img.depth();
			if (img.type() != CV_32FC1) {
				img.convertTo(depth, CV_32FC1, 1, 0);
			}
			else
				depth = img;

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

					Vec3f d(-dzdx, -dzdy, 1.0f);

					Vec3f tt(x, y - 1, depth.at<float>(y - 1, x));
					Vec3f ll(x - 1, y, depth.at<float>(y, x - 1));
					Vec3f c(x, y, depth.at<float>(y, x));

					Vec3f d2 = (ll - c).cross(tt - c);


					Vec3f n = normalize(d);

					double azimuth = atan2(-d2[1], -d2[0]); // -pi -> pi
					if (azimuth < 0)
						azimuth += 2 * CV_PI;

					double zenith = atan(sqrt(d2[1] * d2[1] + d2[0] * d2[0]));

					cv::Vec3f angles(azimuth / (2 * CV_PI), (zenith + CV_PI / 2) / CV_PI, 1);
					angleMat.at<cv::Vec3f>(y, x) = angles;

					//normals.at<Vec3f>(y, x) = n;
				}
			}

			auto& result = hog::get2DHistogramsOfX(cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(1)), angleMat, patchSize, 9, true, false);



			//	cv::imshow(msg, normals);

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
			angleMat.convertTo(tmp, CV_8UC3, 255, 0);
			cv::imshow(msg, result.combineHOGImage(tmp));
		};

		func(tp, "TP");

		func(tn, "TN");

		cv::waitKey(0);
		nr++;
	}

}

int main()
{
	//testClassifier();
	//testFeature();
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	//testClassifier();
	//saveTNTP();
	//return 0;

	int nrOfEvaluations = 100;
	std::set<std::string> set;


	//for (int i = 10; i < 100; i+=5)
	//{
	//	tester.addAvailableCreator(std::string("SURF(RGB)_") + std::to_string(i), new SURFFeatureCreator(std::string("SURF(RGB)_") + std::to_string(i), i));

	//	set = { std::string("SURF(RGB)_") + std::to_string(i) };
	//	tester.addJob(set, nrOfEvaluations);
	//}
	//


	FeatureTester tester(baseDatasetPath);
	tester.addAvailableCreator(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(new HOGHistogramVarianceFeatureCreator(std::string("S2HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(new CornerFeatureCreator(std::string("Corner(RGB)"), false));
	tester.addAvailableCreator(new CornerFeatureCreator(std::string("Corner(Depth)"), true));
	tester.addAvailableCreator(new HistogramDepthFeatureCreator(std::string("Histogram(Depth)")));

	tester.addAvailableCreator(new SURFFeatureCreator(std::string("SURF(RGB)"), 80, false));
	tester.addAvailableCreator(new SURFFeatureCreator(std::string("SURF(Depth)"), 80, true));
	tester.addAvailableCreator(new ORBFeatureCreator(std::string("ORB(RGB)"), 80, false));
	tester.addAvailableCreator(new ORBFeatureCreator(std::string("ORB(Depth)"), 80, true));
	tester.addAvailableCreator(new SIFTFeatureCreator(std::string("SIFT(RGB)"), 80, false));
	tester.addAvailableCreator(new SIFTFeatureCreator(std::string("SIFT(Depth)"), 80, true));
	tester.addAvailableCreator(new CenSurEFeatureCreator(std::string("CenSurE(RGB)"), 80, false));
	tester.addAvailableCreator(new CenSurEFeatureCreator(std::string("CenSurE(Depth)"), 80, true));
	tester.addAvailableCreator(new MSDFeatureCreator(std::string("MSD(RGB)"), 80, false));
	tester.addAvailableCreator(new MSDFeatureCreator(std::string("MSD(Depth)"), 80, true));
	/*tester.addAvailableCreator(new BRISKFeatureCreator(std::string("BRISK(RGB)"), 80, false)); // way too damn slow
	tester.addAvailableCreator(new BRISKFeatureCreator(std::string("BRISK(Depth)"), 80, true));*/
	tester.addAvailableCreator(new FASTFeatureCreator(std::string("FAST(RGB)"), 80, false));
	tester.addAvailableCreator(new FASTFeatureCreator(std::string("FAST(Depth)"), 80, true));

	tester.addAvailableCreator(new HDDFeatureCreator(std::string("HDD"), patchSize, binSize, refWidth, refHeight));
	tester.addAvailableCreator(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight));
	tester.addAvailableCreator(new HONVFeatureCreator(std::string("HONV"), patchSize, binSize, refWidth, refHeight));



	//evaluate each creator individually
	for (auto& creator : tester.getAvailableCreators()) {
		set = { creator->getName() };
		tester.addJob(set, nrOfEvaluations);
	}
	tester.runJobs(std::string("results\\individualresults.csv"));

	// evaluate each creator combined with HOG(RGB)
	for (auto& creator : tester.getAvailableCreators()) {
		if (creator->getName() != "HoG(RGB)") {
			set = { "HoG(RGB)", creator->getName() };
			tester.addJob(set, nrOfEvaluations);
		}
	}
	tester.runJobs(std::string("results\\hogrgb_results.csv"));

	for (auto& creator : tester.getAvailableCreators()) {
		if (creator->getName() != "HoG(RGB)" && creator->getName() != "HoG(Depth)") {
			set = { "HoG(Depth)", "HoG(RGB)", creator->getName() };
			tester.addJob(set, nrOfEvaluations);
		}
	}
	tester.runJobs(std::string("results\\hogrgbhogdepth_results.csv"));


	for (auto& creator : tester.getAvailableCreators()) {
		if (creator->getName() != "HoG(RGB)" && creator->getName() != "HoG(Depth)" && creator->getName() != "LBP(RGB)") {
			set = { "HoG(Depth)", "HoG(RGB)", "LBP(RGB)",  creator->getName() };
			tester.addJob(set, nrOfEvaluations);
		}
	}
	tester.runJobs(std::string("results\\hogrgbhogdepthlbprgb_results.csv"));



	for (auto& creator : tester.getAvailableCreators()) {
		if (creator->getName() != "HoG(RGB)" && creator->getName() != "HoG(Depth)" && creator->getName() != "LBP(RGB)" && creator->getName() != "Histogram(Depth)") {
			set = { "HoG(Depth)", "HoG(RGB)", "LBP(RGB)", "Histogram(Depth)",  creator->getName() };
			tester.addJob(set, nrOfEvaluations);
		}
	}
	tester.runJobs(std::string("results\\hogrgbhogdepthlbprgbhistogramdepth_results.csv"));


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


	//getchar();
	return 0;
}

