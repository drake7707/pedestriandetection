#include "opencv2/opencv.hpp"

using namespace cv;

// From https://github.com/bytefish/opencv/blob/master/lbp/lbp.cpp


namespace Algorithms {

	
	template <typename _Tp>
	void OLBP_(const Mat& src, Mat& dst, _Tp minDiff) {
		dst = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);

		// faster
		_Tp
			*row_m1,
			*row = (_Tp*)src.ptr<_Tp>(0),
			*row_p1 = (_Tp*)src.ptr<_Tp>(1);

		for (int i = 1; i<src.rows - 1; i++) {
			unsigned char *dst_row = dst.ptr<unsigned char>(i - 1);
			row_m1 = row;
			row = row_p1;
			row_p1 = (_Tp*)src.ptr<_Tp>(i + 1);

			for (int j = 1; j<src.cols - 1; j++) {
				_Tp center = row[j] + minDiff;
				unsigned char code = 0;
				code |= (row_m1[j - 1] > center) << 7;
				code |= (row_m1[j]   > center) << 6;
				code |= (row_m1[j + 1] > center) << 5;
				code |= (row[j + 1] > center) << 4;
				code |= (row_p1[j + 1] > center) << 3;
				code |= (row_p1[j]   > center) << 2;
				code |= (row_p1[j - 1] > center) << 1;
				code |= (row[j - 1] > center) << 0;
				dst_row[j - 1] = code;
			}
		}
		/*
		for (int i = 1; i<src.rows - 1; i++) {
		for (int j = 1; j<src.cols - 1; j++) {
		_Tp center = src.at<_Tp>(i, j)- + minDiff;
		unsigned char code = 0;
		code |= (src.at<_Tp>(i - 1, j - 1) > center) << 7;
		code |= (src.at<_Tp>(i - 1, j) > center) << 6;
		code |= (src.at<_Tp>(i - 1, j + 1) > center) << 5;
		code |= (src.at<_Tp>(i, j + 1) > center) << 4;
		code |= (src.at<_Tp>(i + 1, j + 1) > center) << 3;
		code |= (src.at<_Tp>(i + 1, j) > center) << 2;
		code |= (src.at<_Tp>(i + 1, j - 1) > center) << 1;
		code |= (src.at<_Tp>(i, j - 1) > center) << 0;
		dst.at<unsigned char>(i - 1, j - 1) = code;
		}
		}*/
	}

	template <typename _Tp>
	void ELBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
		neighbors = max(min(neighbors, 31), 1); // set bounds...
												// Note: alternatively you can switch to the new OpenCV Mat_
												// type system to define an unsigned int matrix... I am probably
												// mistaken here, but I didn't see an unsigned int representation
												// in OpenCV's classic typesystem...
		dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
		for (int n = 0; n < neighbors; n++) {
			// sample points
			float x = static_cast<float>(radius) * cos(2.0*CV_PI*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius) * -sin(2.0*CV_PI*n / static_cast<float>(neighbors));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			// iterate through your data
			for (int i = radius; i < src.rows - radius; i++) {
				for (int j = radius; j < src.cols - radius; j++) {
					float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
					// we are dealing with floating point precision, so add some little tolerance
					dst.at<unsigned int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) && (abs(t - src.at<_Tp>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}

	template <typename _Tp>
	void VARLBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
		max(min(neighbors, 31), 1); // set bounds
		dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32FC1); //! result
																				  // allocate some memory for temporary on-line variance calculations
		Mat _mean = Mat::zeros(src.rows, src.cols, CV_32FC1);
		Mat _delta = Mat::zeros(src.rows, src.cols, CV_32FC1);
		Mat _m2 = Mat::zeros(src.rows, src.cols, CV_32FC1);
		for (int n = 0; n < neighbors; n++) {
			// sample points
			float x = static_cast<float>(radius) * cos(2.0*CV_PI*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius) * -sin(2.0*CV_PI*n / static_cast<float>(neighbors));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			// iterate through your data
			for (int i = radius; i < src.rows - radius; i++) {
				for (int j = radius; j < src.cols - radius; j++) {
					float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
					_delta.at<float>(i, j) = t - _mean.at<float>(i, j);
					_mean.at<float>(i, j) = (_mean.at<float>(i, j) + (_delta.at<float>(i, j) / (1.0*(n + 1)))); // i am a bit paranoid
					_m2.at<float>(i, j) = _m2.at<float>(i, j) + _delta.at<float>(i, j) * (t - _mean.at<float>(i, j));
				}
			}
		}
		// calculate result
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				dst.at<float>(i - radius, j - radius) = _m2.at<float>(i, j) / (1.0*(neighbors - 1));
			}
		}
	}

	// now the wrapper functions
	void OLBP(const Mat& src, Mat& dst, uchar minDiff = 0) {
		switch (src.type()) {
		case CV_8SC1: OLBP_<char>(src, dst, minDiff); break;
		case CV_8UC1: OLBP_<unsigned char>(src, dst, minDiff); break;
		case CV_16SC1: OLBP_<short>(src, dst, minDiff); break;
		case CV_16UC1: OLBP_<unsigned short>(src, dst, minDiff); break;
		case CV_32SC1: OLBP_<int>(src, dst, minDiff); break;
		case CV_32FC1: OLBP_<float>(src, dst, minDiff); break;
		case CV_64FC1: OLBP_<double>(src, dst, minDiff); break;
		}
	}

	void ELBP(const Mat& src, Mat& dst, int radius, int neighbors) {
		switch (src.type()) {
		case CV_8SC1: ELBP_<char>(src, dst, radius, neighbors); break;
		case CV_8UC1: ELBP_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1: ELBP_<short>(src, dst, radius, neighbors); break;
		case CV_16UC1: ELBP_<unsigned short>(src, dst, radius, neighbors); break;
		case CV_32SC1: ELBP_<int>(src, dst, radius, neighbors); break;
		case CV_32FC1: ELBP_<float>(src, dst, radius, neighbors); break;
		case CV_64FC1: ELBP_<double>(src, dst, radius, neighbors); break;
		}
	}

	void VARLBP(const Mat& src, Mat& dst, int radius, int neighbors) {
		switch (src.type()) {
		case CV_8SC1: VARLBP_<char>(src, dst, radius, neighbors); break;
		case CV_8UC1: VARLBP_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1: VARLBP_<short>(src, dst, radius, neighbors); break;
		case CV_16UC1: VARLBP_<unsigned short>(src, dst, radius, neighbors); break;
		case CV_32SC1: VARLBP_<int>(src, dst, radius, neighbors); break;
		case CV_32FC1: VARLBP_<float>(src, dst, radius, neighbors); break;
		case CV_64FC1: VARLBP_<double>(src, dst, radius, neighbors); break;
		}
	}

	// now the Mat return functions
	Mat OLBP(const Mat& src) {
		Mat dst;
		OLBP(src, dst);
		return dst;
	}

	Mat ELBP(const Mat& src, int radius, int neighbors) {
		Mat dst;
		ELBP(src, dst, radius, neighbors);
		return dst;
	}
	Mat VARLBP(const Mat& src, int radius, int neighbors) {
		Mat dst;
		VARLBP(src, dst, radius, neighbors);
		return dst;
	}

}