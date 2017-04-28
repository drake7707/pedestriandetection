#pragma once
#include "IPreparedData.h"
#include "IntegralHistogram.h"

/// <summary>
/// HOG and derivatives can use integral histograms to calculate the sum of each bin at every pixel
/// to easily obtain the histogram for any region, massively speeding up sliding window evaluation
/// </summary>
class HOG1DPreparedData :
	public IPreparedData
{
public:

	IntegralHistogram integralHistogram;

	HOG1DPreparedData();
	virtual ~HOG1DPreparedData();
};

