#pragma once
#include "IPreparedData.h"
#include "IntegralHistogram.h"

/// <summary>
/// HOG and derivatives can use integral histograms to calculate the sum of each bin at every pixel
/// to easily obtain the histogram for any region, massively speeding up sliding window evaluation
/// </summary>
class IntHistPreparedData :
	public IPreparedData
{
public:

	IntegralHistogram integralHistogram;

	IntHistPreparedData();
	virtual ~IntHistPreparedData();
};

