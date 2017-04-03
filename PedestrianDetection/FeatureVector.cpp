#include "FeatureVector.h"

FeatureVector::FeatureVector() {

}

FeatureVector::FeatureVector(int size, float defaultValue) : std::vector<float>(size, defaultValue) {

}

FeatureVector::FeatureVector(const FeatureVector& v1, const FeatureVector& v2) {
	reserve(v1.size() + v2.size());
	for (float v : v1)
		push_back(v);
	for (float v : v2)
		push_back(v);
}

void FeatureVector::normalize() {
	double n = norm();
	if (n > 0) {
		for (int i = 0; i < this->size(); i++)
			at(i) = at(i) / n;
	}
}

double FeatureVector::norm() const {
	double sum = 0;
	for (int i = 0; i < this->size(); i++)
		sum += at(i) * at(i);

	double norm = sqrt(sum);
	return norm;
}


double FeatureVector::distanceToSquared(const FeatureVector& v) const {
	if (v.size() != this->size())
		throw std::exception("Can't calculate distance between 2 vectors of varying dimensions");

	double sumsquares = 0;
	for (int f = 0; f < size(); f++)
	{
		sumsquares += (v[f] - at(f)) * (v[f] - at(f));
	}
	return sumsquares;
}

void FeatureVector::applyMeanAndVariance(const std::vector<float>& meanVector, const std::vector<float>& sigmaVector) {
	for (int f = 0; f < this->size(); f++)
		at(f) = sigmaVector[f] != 0 ? (at(f) - meanVector[f]) / sigmaVector[f] : 0;
}


double FeatureVector::dot(FeatureVector& v) {
	if (v.size() != this->size())
		throw std::exception("Can't calculate distance between 2 vectors of varying dimensions");

	double sum = 0;
	for (int i = 0; i < this->size(); i++)
	{
		sum += at(i) * v.at(i);
	}
	return sum;
}

cv::Mat FeatureVector::toMat() {
	cv::Mat imgSample(1, this->size(), CV_32FC1);
	for (int i = 0; i < this->size(); i++)
		imgSample.at<float>(0, i) = at(i);
	return imgSample;
}


FeatureVector::~FeatureVector()
{
}
