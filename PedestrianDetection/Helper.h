#pragma once
#include "opencv2/opencv.hpp"

typedef std::vector<float> Histogram;

void showHistogram(Histogram& hist);

int randBetween(int min, int max);

int ceilTo(double val, double target);