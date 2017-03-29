#pragma once

namespace heatmap {

	double interpolate(double val, double y0, double x0, double y1, double x1);

	double base(double val);

	double red(double gray);
	double green(double gray);

	double blue(double gray);


};