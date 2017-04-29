#pragma once
#include "IPreparedData.h"
#include "IntegralHistogram2D.h"

/// <summary>
/// HOG and derivatives can use integral histograms to calculate the sum of each bin at every pixel
/// to easily obtain the histogram for any region, massively speeding up sliding window evaluation
/// </summary>
class IntHist2DPreparedData :
	public IPreparedData
{
public:

	IntegralHistogram2D integralHistogram;

	IntHist2DPreparedData();
	virtual ~IntHist2DPreparedData();
};

