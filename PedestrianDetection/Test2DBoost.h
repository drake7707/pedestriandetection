#pragma once

#include <set>
#include <opencv2\opencv.hpp>
#include "adaboost\adaboost.hpp"



class SVMClassifier : public DM_AG::Classifier<Point2D> {


private:

	cv::Ptr<cv::ml::SVM> svm;
	cv::Mat wT;
	double b;


	int analyze(const Point2D& p) const {
		return evaluate(p);

	}

public:

	void train(const std::set<Point2D>& tpSet, const std::set<Point2D>& tnSet);
	int evaluate(const Point2D& p) const;

	void drawHyperPlane(const cv::Mat& m, const cv::Scalar& color);

	static int getActualClass(const Point2D p) {
		if ((p.x - 0.5)* (p.x - 0.5) + (p.y - 0.5)*(p.y - 0.5) < 0.3*0.3)
			return 1;
		else
			return -1;
	}
};

class SplitClassifier : public DM_AG::Classifier<Point2D> {

public:
	SplitClassifier(bool splitHorizontally, float value, bool flipped) : splitHorizontally(splitHorizontally), value(value), flipped(flipped) {

	}
private:
	bool splitHorizontally;
	bool flipped;
	float value;


	int analyze(const Point2D& p) const {
		if (flipped) {
			if (splitHorizontally)
				return p.y < value ? 1 : -1;
			else
				return p.x < value ? 1 : -1;
		}
		else {
			if (splitHorizontally)
				return p.y < value ? -1 : 1;
			else
				return p.x < value ? -1 : 1;
		}
	}
};

class Test2DBoost
{

	std::set<Point2D> tpSet;
	std::set<Point2D> tnSet;
	DM_AG::ADA<Point2D> ada;

	DM_AG::Classifier<Point2D>::CollectionClassifiers classifiers;

	DM_AG::StrongClassifier<Point2D>* sc;


	int width;
	int height;


private:

	void buildTrainingSet(int trainingSize);

	void buildModel();

	void evaluateAndRemoveTN(SVMClassifier* classifier);

	void evaluateBoost();

public:

	void run(int trainingSize) {
		buildTrainingSet(trainingSize);
		buildModel();
		evaluateBoost();
	}
	Test2DBoost(int width, int height) : width(width), height(height) { }
	~Test2DBoost();
};

