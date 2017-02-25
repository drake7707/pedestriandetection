#pragma once
#include <vector>
#include "opencv2\opencv.hpp"

class FeatureVector : public std::vector<float>
{
public:
	FeatureVector() {

	}

	FeatureVector(int size, float defaultValue) : std::vector<float>(size, defaultValue) {

	}

	FeatureVector(const FeatureVector& v1, const FeatureVector& v2) {
		reserve(v1.size() + v2.size());
		for (float v : v1)
			push_back(v);
		for (float v : v2)
			push_back(v);
	}

	void normalize() {
		double sum = 0;
		for (int i = 0; i < this->size(); i++)
			sum += at(i) * at(i);

		double norm = sqrt(sum);
		if (norm > 0) {
			for (int i = 0; i < this->size(); i++)
				at(i) = at(i) / norm;
		}
	}

	double distanceToSquared(const FeatureVector& v) {
		if (v.size() != this->size())
			throw std::exception("Can't calculate distance between 2 vectors of varying dimensions");

		double sumsquares = 0;
		for (int f = 0; f < size(); f++)
		{
			sumsquares += (v[f] - at(f)) * (v[f] - at(f));
		}
		return sumsquares;
	}

	void applyMeanAndVariance(std::vector<float> meanVector, std::vector<float> sigmaVector) {
		for (int f = 0; f < this->size(); f++)
			at(f) = sigmaVector[f] != 0 ? (at(f) - meanVector[f]) / sigmaVector[f] : 0;
	}

	cv::Mat toMat() {
		cv::Mat imgSample(1, this->size(), CV_32FC1);
		for (int i = 0; i < this->size(); i++)
			imgSample.at<float>(0, i) = at(i);
		return imgSample;
	}

	~FeatureVector();
};

