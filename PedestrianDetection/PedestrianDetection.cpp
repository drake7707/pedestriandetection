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

void saveTNTP() {

	KITTIDataSet* dataSet = new KITTIDataSet(kittiDatasetPath);

	srand(7707);


	std::vector<DataSetLabel> labels = dataSet->getLabels();

	std::string currentNumber = "";
	std::vector<cv::Mat> currentImages;
	int sizeVariance = 4;
	int tpIdxRGB = 0;
	int tnIdxRGB = 0;

	int tpIdxDepth = 0;
	int tnIdxDepth = 0;

	int idx = 0;
	for (auto& l : labels) {


		if (currentNumber != l.getNumber()) {
			currentNumber = l.getNumber();
			currentImages = dataSet->getImagesForNumber(currentNumber);
		}

		// get true positive and true negative image
		// -----------------------------------------

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
				if (i == 0)
					tpIdxRGB++;
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
				if (i == 0)
					tpIdxRGB++;
				else
					tpIdxDepth++;

				// take a number of true negative patches that don't overlap
				for (int k = 0; k < 2; k++)
				{
					// take an equivalent patch at random for a true negative
					cv::Mat rgbTN;
					cv::Rect2d rTN;

					int iteration = 0;
					do {
						double sizeMultiplier = (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
						double width = l.getBbox().width * sizeMultiplier;
						double height = l.getBbox().height * sizeMultiplier;
						rTN = cv::Rect2d(randBetween(0, img.cols - width), randBetween(0, img.rows - height), width, height);
					} while (iteration++ < 100 && ((rTN & l.getBbox()).area() > 0 || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= img.cols || rTN.y + rTN.height >= img.rows));


					if (iteration < 100) {
						img(rTN).copyTo(rgbTN);
						cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));
						//tnFunc(rgbTN);
						path = "D:\\PedestrianDetectionDatasets\\kitti\\regions\\tn\\";
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
		idx++;
	}
}


int main()
{
	std::cout << "--------------------- New console session -----------------------" << std::endl;
	//saveTNTP();
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

