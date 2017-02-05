// PedestrianDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "KITTIDataSet.h"
#include "HistogramOfOrientedGradients.h"

using namespace cv;

int main()
{


	/*KITTIDataSet dataset("D:\\PedestrianDetectionDatasets\\kitti");

	for (auto& l : dataset.getLabels()) {
		std::cout << l.getNumber() << "   " << l.getBbox().x << "," << l.getBbox().y << " / " << l.getBbox().width << "x" << l.getBbox().height << std::endl;
	}
*/

	//auto img = imread("D:\\circle.png");
	auto img = imread("D:\\test.jpg");
	namedWindow("Test");
	imshow("Test", img);


	auto result = getHistogramsOfOrientedGradient(img, 8, 18, true);

	
	namedWindow("HoG");
	imshow("HoG", result.hogImage);


	waitKey(0);


	getchar();

    return 0;
}

