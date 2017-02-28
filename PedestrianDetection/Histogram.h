#pragma once
#include <vector>
class Histogram : public std::vector<float>
{
public:
	Histogram() {

	}

	Histogram(int size, float defaultValue) : std::vector<float>(size, defaultValue) {

	}

	float getS2() const;
};

