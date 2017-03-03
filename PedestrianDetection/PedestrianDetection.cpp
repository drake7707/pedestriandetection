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


#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"


#include "KITTIDataSet.h"
#include "DataSet.h"

std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";
std::string baseDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti\\regions";

//std::string kittiDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti";
//std::string baseDatasetPath = "C:\\Users\\dwight\\Downloads\\dwight\\kitti\\regions";

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


	std::vector<DataSetLabel> labels = dataSet.getLabels();

	std::vector<std::vector<DataSetLabel>> labelsPerNumber(dataSet.getNrOfImages(), std::vector<DataSetLabel>());
	for (auto& l : labels)
		labelsPerNumber[atoi(l.getNumber().c_str())].push_back(l);

	int sizeVariance = 4;
	int nrOfTNPerImage = 2;

	for (int i = 0; i < labelsPerNumber.size(); i++)
	{
		TrainingImage tImg;
		tImg.number = i;

		
		std::vector<cv::Mat> currentImages = dataSet.getImagesForNumber(i);
		std::vector<cv::Rect2d> selectedRegions;

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
				double sizeMultiplier = (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
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

	return trainingSet;
}



void trainDetailedClassifier() {

	int trainingRound = 1;
	TrainingDataSet tSet(kittiDatasetPath);
	tSet.load(std::string("trainingsets\\train0.txt"));

	std::string featureSetName = "HoG(Depth)+HoG(RGB)+LBP(RGB)";
	FeatureSet setFinal;

	setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight)));
	setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight)));
	setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight)));
	
	ModelEvaluator modelFinal(featureSetName, tSet, setFinal);
	modelFinal.loadModel(std::string("models\\" + featureSetName + ".xml"));


	
	// parameters, number of additional TN/TP
	std::ofstream str("results\\" + featureSetName + "_round" + std::to_string(trainingRound) + ".csv");
	EvaluationSlidingWindowResult result = modelFinal.evaluateWithSlidingWindow(500, trainingRound, 0, 1000);
	for (auto& result : result.evaluations) {
		result.toCSVLine(str, false);
		str << std::endl;
	}

	TrainingDataSet newTrainingSet = tSet;
	for (auto& swregion : result.worstFalsePositives) {

		TrainingRegion r;
		r.region = swregion.bbox;
		r.regionClass = -1; // it was a negative but positive was specified
		newTrainingSet.addTrainingRegion(swregion.imageNumber, r);
	}

	for (auto& swregion : result.worstFalseNegatives) {

		TrainingRegion r;
		r.region = swregion.bbox;
		r.regionClass = 1; // it was a positive but negative was specified
		newTrainingSet.addTrainingRegion(swregion.imageNumber, r);
	}
	newTrainingSet.save("trainingsets\\" + featureSetName + "_" + "train" + std::to_string(trainingRound) + ".txt");
}

void testClassifier() {

//	TrainingDataSet tSet(kittiDatasetPath);
//	tSet.load(std::string("train0.txt"));
//
//
//	std::vector<FeatureSet> cascadeFeatureSets;
//	std::vector<ModelEvaluator> cascadeEvaluators;
//	std::vector<double> valueShifts;
//
//	//FeatureSet set2;
//	//set2.addCreator(new HistogramDepthFeatureCreator(std::string("Histogram(Depth)")));
//	//cascadeFeatureSets.push_back(set2);
//	//ModelEvaluator model2(tSet, set2);
//	//model2.loadModel(std::string("models\\Histogram(Depth).xml"));
//	//cascadeEvaluators.push_back(model2);
//	//valueShifts.push_back(7);
//
//
//
//	//FeatureSet set1;
//	//set1.addCreator(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight));
//	//cascadeFeatureSets.push_back(set1);
//	//ModelEvaluator model1(tSet, set1);
//	//model1.loadModel(std::string("models\\LBP(RGB).xml"));
//	//cascadeEvaluators.push_back(model1);
//	//valueShifts.push_back(7.4);
//
//
//	FeatureSet setFinal;
//	/*setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(Depth)"), true, patchSize, binSize, refWidth, refHeight)));
//	setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(std::string("HoG(RGB)"), false, patchSize, binSize, refWidth, refHeight)));
//	setFinal.addCreator(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(std::string("LBP(RGB)"), patchSize, 20, refWidth, refHeight)));
//*/
//	cascadeFeatureSets.push_back(std::move(setFinal));
//	ModelEvaluator modelFinal(tSet, setFinal);
//	modelFinal.loadModel(std::string("models\\HoG(Depth)+HoG(RGB)+LBP(RGB).xml"));
//	cascadeEvaluators.push_back(modelFinal);
//	valueShifts.push_back(2.2);
//
//	/*ClassifierEvaluation eval = model.evaluateDataSet(1, false)[0];
//	eval.print(std::cout);*/
//
//	int nr = 0;
//	while (true) {
//		char nrStr[7];
//		sprintf(nrStr, "%06d", nr);
//
//		cv::Mat mRGB = cv::imread(kittiDatasetPath + "\\rgb\\" + nrStr + ".png");
//		cv::Mat mDepth = cv::imread(kittiDatasetPath + "\\depth\\" + nrStr + ".png");
//		mDepth.convertTo(mDepth, CV_32FC1, 1.0 / 0xFFFF, 0);
//
//
//		int nrOfWindowsEvaluated = 0;
//		long slidingWindowTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
//
//
//
//			slideWindow(mRGB.cols, mRGB.rows, [&](cv::Rect bbox) -> void {
//				cv::Mat regionRGB;
//				cv::resize(mRGB(bbox), regionRGB, cv::Size2d(refWidth, refHeight));
//
//				cv::Mat regionDepth;
//				cv::resize(mDepth(bbox), regionDepth, cv::Size2d(refWidth, refHeight));
//
//				bool isPositive = true;
//				for (int i = 0; i < cascadeEvaluators.size(); i++)
//				{
//					FeatureVector v = cascadeFeatureSets[i].getFeatures(regionRGB, regionDepth);
//					EvaluationResult result = cascadeEvaluators[i].evaluateWindow(regionRGB, regionDepth, valueShifts[i]);
//					if (result.resultClass == -1) {
//						isPositive = false;
//						break;
//					}
//
//				}
//
//				if (isPositive)
//					cv::rectangle(mRGB, bbox, cv::Scalar(0, 255, 0), 2);
//
//				nrOfWindowsEvaluated++;
//			}, 0.5, 2, 16);
//
//		});
//
//		cv::imshow("Test", mRGB);
//
//		std::cout << "Number of windows evaluated: " << nrOfWindowsEvaluated << " in " << slidingWindowTime << "ms" << std::endl;
//		// this will leak because creators are never disposed!
//		cv::waitKey(0);
//		nr++;
//	}
}


void testFeature() {
	int nr = 0;
	while (true) {
		char nrStr[7];
		sprintf(nrStr, "%06d", nr);

		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\rgb\\000000.png");// kittiDatasetPath + "\\regions\\tp\\depth" + std::to_string(nr) + ".png");
		//cv::Mat tp = cv::imread(kittiDatasetPath + "\\regions\\tp\\rgb" + std::to_string(nr) + ".png");
		cv::Mat tp = cv::imread(kittiDatasetPath + "\\depth\\000000.png", CV_LOAD_IMAGE_UNCHANGED);
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
	//trainDetailedClassifier();
	//testFeature();
	/*TrainingDataSet tSet = saveTNTP();

	tSet.save(std::string("train0.txt"));
*/

	ProgressWindow* wnd = ProgressWindow::getInstance();
	wnd->run();

	std::cout << std::endl;
	std::cout.flush();
	TrainingDataSet tSet(kittiDatasetPath);
	tSet.load(std::string("trainingsets\\train0.txt"));

	//testFeature();
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	//testClassifier();
	//saveTNTP();
	//return 0;

	int nrOfEvaluations = 500;
	std::set<std::string> set;


	//for (int i = 10; i < 100; i+=5)
	//{
	//	tester.addAvailableCreator(std::string("SURF(RGB)_") + std::to_string(i), new SURFFeatureCreator(std::string("SURF(RGB)_") + std::to_string(i), i));

	//	set = { std::string("SURF(RGB)_") + std::to_string(i) };
	//	tester.addJob(set, nrOfEvaluations);
	//}
	//


	FeatureTester tester;
	tester.nrOfConcurrentJobs = 4;

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HoG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("S2HoG(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGHistogramVarianceFeatureCreator(name, false, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HoG(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HOGFeatureCreator(name, true, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Corner(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CornerFeatureCreator(name, true))); }));

	tester.addFeatureCreatorFactory(FactoryCreator(std::string("Histogram(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HistogramDepthFeatureCreator(name))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SURF(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SURFFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("ORB(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new ORBFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("SIFT(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new SIFTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("CenSurE(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new CenSurEFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("MSD(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new MSDFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 80, false))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("FAST(Depth)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new FASTFeatureCreator(name, 80, true))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HDD"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HDDFeatureCreator(name, patchSize, binSize, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("LBP(RGB)"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new LBPFeatureCreator(name, patchSize, 20, refWidth, refHeight))); }));
	tester.addFeatureCreatorFactory(FactoryCreator(std::string("HONV"), [](std::string& name) -> std::unique_ptr<IFeatureCreator> { return std::move(std::unique_ptr<IFeatureCreator>(new HONVFeatureCreator(std::string("HONV"), patchSize, binSize, refWidth, refHeight))); }));


	set = { "HoG(Depth)", "HoG(RGB)","LBP(RGB)" };
	tester.addJob(set, kittiDatasetPath, nrOfEvaluations, 4);
	tester.runJobs();

	//evaluate each creator individually
	//for (auto& name : tester.getFeatureCreatorFactories()) {
	//	set = { name };
	//	tester.addJob(set, kittiDatasetPath, nrOfEvaluations);
	//}
	//tester.runJobs();

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


