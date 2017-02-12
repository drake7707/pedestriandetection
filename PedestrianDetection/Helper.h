#pragma once
#include "opencv2/opencv.hpp"
#include "Histogram.h"

void showHistogram(Histogram& hist);

int randBetween(int min, int max);

int ceilTo(double val, double target);