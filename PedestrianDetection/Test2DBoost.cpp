#include "Test2DBoost.h"
#include "opencv2\opencv.hpp"
#include "Detector.h"
#include "DetectorCascade.h"


std::ostream & operator<<(std::ostream &os, const Point2D& p)
{
	return os << "<" << p.x << "," << p.y << ">";
}



Test2DBoost::~Test2DBoost()
{
	if (sc != nullptr)
		delete sc;

	for (auto& c : classifiers)
		delete c;
}


void SVMClassifier::train(const std::set<Point2D>& tpSet, const std::set<Point2D>& tnSet) {

	int trainingSize = tpSet.size() + tnSet.size();
	int featureSize = 2;

	cv::Mat trainingMat(trainingSize, featureSize, CV_32FC1);
	cv::Mat trainingLabels(trainingSize, 1, CV_32SC1);

	int i = 0;
	for (auto& p : tpSet) {
		trainingMat.at<float>(i, 0) = (float)p.x;
		trainingMat.at<float>(i, 1) = (float)p.y;
		trainingLabels.at<float>(i, 0) = getActualClass(p);
		i++;
	}
	for (auto& p : tnSet) {
		trainingMat.at<float>(i, 0) = (float)p.x;
		trainingMat.at<float>(i, 1) = (float)p.y;
		trainingLabels.at<float>(i, 0) = getActualClass(p);
		i++;
	}



	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR); // LINEAR KERNEL but in order
	svm->setC(0.01);

	cv::Mat classWeights(2, 1, CV_32FC1);
	classWeights.at<float>(0, 0) = 0.5;
	classWeights.at<float>(1, 0) = 0.5;
	svm->setClassWeights(classWeights);

	svm->train(trainingMat, cv::ml::ROW_SAMPLE, trainingLabels);

	std::vector<float> alpha;
	std::vector<float> svidx;
	cv::Mat sv = svm->getSupportVectors();
	b = svm->getDecisionFunction(0, alpha, svidx);
	wT = cv::Mat(1, sv.cols, CV_32F, cv::Scalar(0));
	for (int r = 0; r < sv.rows; ++r)
	{
		for (int c = 0; c < sv.cols; ++c)
			wT.at<float>(0, c) += alpha[r] * sv.at<float>(r, c);
	}

	/*double svPointingTo = (wT.dot(sv) - b);

	std::cout << svPointingTo << std::endl;*/
	//if (svPointingTo < 0) {
	//	// flipping required
	//	wT = wT * (-1);
	//	b = -b;
	//}
}


int SVMClassifier::evaluate(const Point2D& p) const {

	int featureSize = 2;
	cv::Mat testMat(1, featureSize, CV_32F);

	testMat.at<float>(0, 0) = (float)p.x;
	testMat.at<float>(0, 1) = (float)p.y;
	double value = -(wT.dot(testMat) - b);

	//	double value = svm->predict(testMat);// (wT.dot(testMat) - b);

	if (value > 0)
		return 1;
	else
		return -1;
}


void SVMClassifier::drawHyperPlane(const cv::Mat& m, const cv::Scalar& color) {


	float wx = wT.at<float>(0, 0);
	float wy = wT.at<float>(0, 1);

	float x0 = 0;
	float y0 = (b - wx*x0) / wy;

	float x1 = 1;
	float y1 = (b - wx*x1) / wy;


	cv::line(m, cv::Point2f(x0 * m.cols, m.rows - y0 * m.rows), cv::Point(x1 * m.cols, m.rows - y1 * m.rows), color, 2);

}





// -----------------------------------------------------------------------------------


void Test2DBoost::buildTrainingSet(int trainingSize) {
	tpSet.clear();
	tnSet.clear();

	/*for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			Point2D p(1.0 * i / width, 1.0 * j / height);
			if (SVMClassifier::getActualClass(p) == 1)
				tpSet.emplace(p);
			else
				tnSet.emplace(p);
		}
	}
	*/
	for (int i = 0; i < trainingSize; i++)
	{
		double val0 = (rand() / (double)RAND_MAX);
		double val1 = (rand() / (double)RAND_MAX);

		Point2D p(val0, val1);

		int pClass = SVMClassifier::getActualClass(p);
		if (pClass == 1)
			tpSet.emplace(p);
		else
			tnSet.emplace(p);
	}

	// equalize the training set
	while (tpSet.size() < tnSet.size()) {
		double val0 = (rand() / (double)RAND_MAX);
		double val1 = (rand() / (double)RAND_MAX);

		Point2D p(val0, val1);
		int pClass = SVMClassifier::getActualClass(p);
		if (pClass == 1) 			tpSet.emplace(p);
	}
	while (tnSet.size() < tpSet.size()) {
		double val0 = (rand() / (double)RAND_MAX);
		double val1 = (rand() / (double)RAND_MAX);

		Point2D p(val0, val1);
		int pClass = SVMClassifier::getActualClass(p);
		if (pClass == -1) 			tnSet.emplace(p);
	}
	
}

void Test2DBoost::buildModel() {


	std::vector<Point2D> points;
	DM_AG::Labels labels;

	for (auto& p : tpSet) {
		points.push_back(p);
		labels.push_back(1);
	}
	for (auto& p : tnSet) {
		points.push_back(p);
		labels.push_back(-1);
	}


	classifiers.push_back(new SplitClassifier(false, 0.25, false));
	classifiers.push_back(new SplitClassifier(false, 0.75, true));
	classifiers.push_back(new SplitClassifier(true, 0.25, false));
	classifiers.push_back(new SplitClassifier(true, 0.75, true));

	/*classifiers.push_back(new SplitClassifier(false, 0.25, true));
	classifiers.push_back(new SplitClassifier(false, 0.75, true));
	classifiers.push_back(new SplitClassifier(true, 0.25, true));
	classifiers.push_back(new SplitClassifier(true, 0.75, true));*/

	
	int oldSizeTN = -1;
	int oldSizeTP = -1;
	int iteration = 0;
	while (iteration < 10) {
		//while (tnSet.size() > 0 && tpSet.size() > 0 && (oldSizeTN != tnSet.size() || oldSizeTP != tpSet.size()) && iteration < 10) {
		oldSizeTN = tnSet.size();
		oldSizeTP = tpSet.size();

		SVMClassifier*  c1 = new SVMClassifier();
		c1->train(tpSet, tnSet);
		classifiers.push_back(c1);
		//evaluateAndRemoveTN(c1);

		buildTrainingSet(1000);
		/*	cv::Mat testImage(width, height, CV_8UC3, cv::Scalar(0));
			for (auto& p : tpSet)
				testImage.at<cv::Vec3b>(p.y * testImage.rows, p.x * testImage.cols) = cv::Vec3b(0, 255, 0);
			for (auto& p : tnSet)
				testImage.at<cv::Vec3b>(p.y * testImage.rows, p.x * testImage.cols) = cv::Vec3b(0, 0, 255);

			imshow(std::to_string(iteration), testImage);*/
		iteration++;

	}

	DM_AG::ClassificationResults weights = ada.ada_boost(classifiers, points, labels, 50);

	sc = new DM_AG::StrongClassifier<Point2D>(weights, &classifiers, labels);

}

void Test2DBoost::evaluateAndRemoveTN(SVMClassifier* classifier) {
	std::set<Point2D>::iterator it = tnSet.begin();
	while (it != tnSet.end()) {
		// copy the current iterator then increment it
		std::set<Point2D>::iterator current = it++;
		Point2D p = *current;

		int pClass = classifier->evaluate(p);
		if (pClass == -1)
			tnSet.erase(p);
	}
	// equalize the training set 
	//while(tpSet.size() > tnSet.size()) {
	//	int diff = tpSet.size() - tnSet.size();
	//	int i = 0;
	//	auto& it = tpSet.begin();
	//	while (tpSet.size() > tnSet.size() && it != tpSet.end()) {
	//		std::set<Point2D>::iterator current = it++;
	//		Point2D p = *current;
	//		if (i % (1+tpSet.size()/diff) == 0)
	//			tpSet.erase(p);

	//		i++;
	//	}

	//}
	//
}

void Test2DBoost::evaluateBoost() {

	cv::Mat testImage(width, height, CV_8UC3);
	cv::Mat correctImage(width, height, CV_8UC3);

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{

			Point2D p(1.0 * i / width, 1.0 * j / height);
			float value = sc->analyze(p);

			testImage.at<cv::Vec3b>(height - 1 - j, i) = value > 0 ? cv::Vec3b(0, 255, 0) : cv::Vec3b(0, 0, 255);
			correctImage.at<cv::Vec3b>(width - 1 - j, i) = SVMClassifier::getActualClass(p) > 0 ? cv::Vec3b(0, 255, 0) : cv::Vec3b(0, 0, 255);

		}
	}


	/*for (auto& c : classifiers) {
		((SVMClassifier*)c)->drawHyperPlane(testImage, cv::Scalar(255, 0, 0));
		((SVMClassifier*)c)->drawHyperPlane(correctImage, cv::Scalar(255, 0, 0));
	}*/
	
	int cIdx = 0;
	for (auto& c : classifiers) {

		ClassifierEvaluation eval;

		cv::Mat cMat(width, height, CV_8UC3);
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				Point2D p(1.0 * i / width, 1.0 * j / height);
				float value = c->analyze(p);


				if (value > 0) {
					cMat.at<cv::Vec3b>(height - 1 - j, i) = cv::Vec3b(0, 255, 0);
					if (SVMClassifier::getActualClass(p) == 1)
						eval.nrOfTruePositives++;
					else {
						eval.nrOfFalsePositives++;
						cMat.at<cv::Vec3b>(height - 1 - j, i) = cv::Vec3b(0, 255, 255);
					}
				}
				else {
					cMat.at<cv::Vec3b>(height - 1 - j, i) = cv::Vec3b(0, 0, 255);
					if (SVMClassifier::getActualClass(p) == -1)
						eval.nrOfTrueNegatives++;
					else {
						eval.nrOfFalseNegatives++;
						cMat.at<cv::Vec3b>(height - 1 - j, i) = cv::Vec3b(255, 255, 0);
					}
				}
			}
		}

		std::cout << "Classifier " + std::to_string(cIdx) << std::endl;
		eval.print(std::cout);

		//((SVMClassifier*)c)->drawHyperPlane(cMat, cv::Scalar(255, 0, 0));
		cIdx++;

		cv::namedWindow("Classifier " + std::to_string(cIdx));
		cv::imshow("Classifier " + std::to_string(cIdx), cMat);
	}


	cv::namedWindow("Test Image");
	cv::namedWindow("Actual Image");
	cv::imshow("Test Image", testImage);
	cv::imshow("Actual Image", correctImage);
	cv::waitKey(0);

}