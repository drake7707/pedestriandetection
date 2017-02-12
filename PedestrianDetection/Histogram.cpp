#include "Histogram.h"


float Histogram::getS2() {
	double avg = 0;
	for (float el : *this) {
		avg += el;
	}
	avg /= this->size();


	double sumvar = 0;
	for (float el : *this) {
		sumvar += (el - avg) * (el - avg);
	}

	// don't divide by N, this results in worse performance
	return sumvar;
}