#pragma once
#include <vector>
#include "Histogram.h"
class Histogram2D : public std::vector<std::vector<float>>
{
public:

	Histogram2D();

	Histogram2D(int size, float defaultValue);


	Histogram flatten() const;
};

