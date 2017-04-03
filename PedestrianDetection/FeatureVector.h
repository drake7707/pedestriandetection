#pragma once
#include <vector>
#include "opencv2\opencv.hpp"

class FeatureVector : public std::vector<float>
{
public:
	FeatureVector();

	FeatureVector(int size, float defaultValue);

	FeatureVector(const FeatureVector& v1, const FeatureVector& v2);
	
	/// <summary>
	/// L2-Normalizes the vector
	/// </summary>
	void normalize();

	/// <summary>
	/// Calculates the L2-norm
	/// </summary>
	double norm() const;

	/// <summary>
	/// Returns the squared distance between this and given vector
	/// </summary>
	double distanceToSquared(const FeatureVector& v) const;

	/// <summary>
	/// Applies the mean and standard deviation on each dimension
	/// </summary>
	void applyMeanAndVariance(const std::vector<float>& meanVector, const std::vector<float>& sigmaVector);

	/// <summary>
	/// Calculates the dot product between this and the given vector
	/// </summary>
	double dot(FeatureVector& v);

	/// <summary>
	/// Converts the feature vector to a 1D Mat
	/// </summary>
	cv::Mat toMat();


	~FeatureVector();
};

