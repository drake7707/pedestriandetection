#pragma once
#include <vector>
class Histogram : public std::vector<float>
{
public:
	Histogram() {

	}

	Histogram(int size, float defaultValue) : std::vector<float>(size, defaultValue) {

	}

	/// <summary>
	/// Calculates the S^2 variance of the histogram
	/// </summary>
	float getS2() const;
};

