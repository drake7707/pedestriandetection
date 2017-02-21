// PedestrianDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

#include "ModelEvaluator.h"
#include "IFeatureCreator.h"
#include "HOGRGBFeatureCreator.h"
#include "FeaturesSet.h"


#include "KITTIDataSet.h"
#include "DataSet.h"
int refWidth = 64;
int refHeight = 128;
std::string kittiDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti";


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
			} while (iteration++ < 100 && (overlaps(pair.second, rTN, selectedTNRegions) || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= currentImages[0].cols || rTN.y + rTN.height >= currentImages[0].rows));


			if (iteration < 100) {
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


int main()
{
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	saveTNTP();
	//return 0;

	FeaturesSet set;
	set.addCreator(&HOGRGBFeatureCreator());

	std::string baseDatasetPath = "D:\\PedestrianDetectionDatasets\\kitti\\regions";
	ModelEvaluator evaluator(baseDatasetPath, set);

	evaluator.train();

	ClassifierEvaluation eval = evaluator.evaluate();

	eval.print(std::cout);



	getchar();
	return 0;
}

