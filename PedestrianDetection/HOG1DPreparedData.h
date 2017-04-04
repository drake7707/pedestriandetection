#pragma once
#include "IPreparedData.h"
#include "IntegralImage.h"

class HOG1DPreparedData :
	public IPreparedData
{
public:

	IntegralHistogram integralHistogram;

	HOG1DPreparedData();
	virtual ~HOG1DPreparedData();
};

