#pragma once
#include "opencv2/opencv.hpp"
#include "Histogram.h"

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 


void showHistogram(Histogram& hist);

int randBetween(int min, int max);

int ceilTo(double val, double target);