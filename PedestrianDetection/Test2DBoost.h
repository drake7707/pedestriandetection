#pragma once

#include <set>
#include <opencv2\opencv.hpp>
#include "adaboost\adaboost.hpp"


struct Point2D {
	float x;
	float y;
	Point2D(float x, float y) : x(x), y(y) { }

	bool operator<(const Point2D& p) const {
		if (x < p.x)
			return true;
		else if (x > p.x)
			return false;
		else
		{
			if (y < p.y)
				return true;
			else
				return false;
		}
	}

	bool operator==(const Point2D& p) const {
		return p.x == x && p.y == y;
	}
};

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
		if ((p.x - 0.5)* (p.x - 0.5) + (p.y - 0.5)*(p.y - 0.5) < 0.25*0.25)
			return 1;
		else
			return -1;
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

