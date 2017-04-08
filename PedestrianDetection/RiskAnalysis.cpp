#include "RiskAnalysis.h"

namespace RiskAnalysis {

	std::vector<std::pair<float, float>> riskTimes = {
		std::pair<float,float>(-100000 , 0), // critical
		std::pair<float,float>(0.0f, 0.5f), // high
		std::pair<float,float>(0.5f, 1.0f), // moderate
		std::pair<float,float>(1.0f, 1.5f), // low
	};

	std::vector<cv::Vec3b> riskColors = {
		cv::Vec3b(64, 0, 128),
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 127, 255),
		cv::Vec3b(0, 255, 0)
	};


	std::vector<std::string> riskCategories = {
		std::string("Critical"),
		std::string("High"),
		std::string("Moderate"),
		std::string("Low")
	};


	float gravity = 9.81; // m/s²
	float maxPedestrianSpeed = 0; // assume standing still

	double getStoppingDistance(double vehicleSpeedForRating, double tireroadFriction) {
		double stoppingDistance = vehicleSpeedForRating * vehicleSpeedForRating / (2 * tireroadFriction * gravity);
		return stoppingDistance;
	}

	double getRemainingTimeToHitPedestrian(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction, int t) {
		float vehicleSpeedForRating = vehicleSpeedKMh / 3.6f;
		double stoppingDistance = getStoppingDistance(vehicleSpeedForRating, tireroadFriction);
		// vehicle on origin point, so - 0
		double distanceToPedestrian = sqrt(pedestrianX * pedestrianX + pedestrianDepth * pedestrianDepth); // m
		double remainingDistanceToAct = distanceToPedestrian - (maxPedestrianSpeed*t) - (stoppingDistance); // m
																											// remainingTime * vehicleSpeedForRating = remainingDistanceToAct;
		double remainingTime = remainingDistanceToAct / vehicleSpeedForRating;

		return remainingTime;
	}


	std::vector<std::string> getRiskCategories() {
		return riskCategories;
	}

	std::string getRiskCategory(double pedestrianDepth, double pedestrianX, float vehicleSpeedKMh, double tireroadFriction, int t) {
		double remainingTime = getRemainingTimeToHitPedestrian(pedestrianDepth, pedestrianX, vehicleSpeedKMh, tireroadFriction, t);
		for (int r = 0; r < riskTimes.size(); r++)
		{
			if (remainingTime >= riskTimes[r].first && remainingTime < riskTimes[r].second) {
				return riskCategories[r];
				break;
			}
		}
		return "";
	}

	cv::Mat getTopDownImage(int imgWidth, int imgHeight, std::vector<DataSetLabel>& labels, float maxDepth, float vehicleSpeedKMh, float tireroadFriction, float t) {

		float vehicleSpeedForRating = vehicleSpeedKMh / 3.6f;

		cv::Mat topdown(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));

		cv::Mat& topdownOverlay = topdown;// (imgHeight, imgWidth, CV_8UC3, cv::Scalar(0, 0, 0));

		for (int j = 0; j < topdownOverlay.rows; j++)
		{
			for (int i = 0; i < topdownOverlay.cols; i++)
			{
				float depth = (1 - 1.0 * j / topdownOverlay.rows) * maxDepth;
				// no pixel -> x in meters
				float x = ((1.0 * i / topdownOverlay.cols) * 2 - 1) * maxDepth;
				
				double remainingTime = getRemainingTimeToHitPedestrian(depth, x, vehicleSpeedKMh);

		
				for (int r = 0; r < riskTimes.size(); r++)
				{
					if (remainingTime >= riskTimes[r].first && remainingTime < riskTimes[r].second) {
						topdownOverlay.at<cv::Vec3b>(j, i) = riskColors[r];
						break;
					}
				}
			}
		}

		// draw car point and maximum horizontal viewing angle
		cv::Point2f carPoint = cv::Point2f(imgWidth / 2, imgHeight);
		cv::circle(topdown, cv::Point2f(imgWidth / 2, imgHeight), 10, cv::Scalar(255, 128, 128), -1);
		cv::line(topdown, carPoint, cv::Point(carPoint.x + imgWidth * cos(-CV_PI / 4), carPoint.y + imgWidth * sin(-CV_PI / 4)), cv::Scalar(255, 128, 128));
		cv::line(topdown, carPoint, cv::Point(carPoint.x + imgWidth * cos(-3 * CV_PI / 4), carPoint.y + imgWidth * sin(-3 * CV_PI / 4)), cv::Scalar(255, 128, 128));


		float t1secRadius = (vehicleSpeedForRating*t / maxDepth * imgHeight);
		double stoppingDistance = getStoppingDistance(vehicleSpeedForRating, tireroadFriction);
		double stoppingDistanceRadius = stoppingDistance / maxDepth * imgHeight;

		//cv::circle(topdown, cv::Point2f(imgWidth / 2, imgHeight), t1secRadius, cv::Scalar(255, 128, 128), 1);
		cv::circle(topdown, cv::Point2f(imgWidth / 2, imgHeight), stoppingDistanceRadius, cv::Scalar(0, 0, 0), 2, CV_AA);

		double maxSecondsRemaining = (maxDepth - stoppingDistance) / vehicleSpeedForRating;


		// draw 10 meter lines
		for (int i = 0; i <= maxDepth; i += 10)
		{
			int depth = i;
			float d = (1.0 * depth / maxDepth);

			float y = imgHeight * (1 - d);

			//cv::line(topdown, cv::Point2f(0, y), cv::Point2f(imgWidth, y), cv::Scalar(192, 192, 192));
			cv::circle(topdown, cv::Point(imgWidth / 2, imgHeight), d * imgHeight, cv::Scalar(128, 128, 128));

			cv::putText(topdown, std::to_string(i) + "m", cv::Point2f(0, y + 12), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, cv::Scalar(128, 128, 128));
		}

		for (auto& l : labels) {
			if (!l.isDontCareArea()) {

				double remainingTime = getRemainingTimeToHitPedestrian(l.z_3d, l.x_3d, vehicleSpeedKMh, tireroadFriction, t);

				float depth = (l.z_3d / maxDepth);
				float y = imgHeight * (1 - depth);
				float x = imgWidth / 2 + (l.x_3d / maxDepth) * imgWidth / 2;

				cv::putText(topdown, std::to_string(remainingTime), cv::Point2f(x, y), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));

				if (remainingTime < 0) {
					// already too late
					remainingTime = 0;
				}

				// from 0,0,255 -> 0,255,0
				float alpha = remainingTime / maxSecondsRemaining;

				cv::circle(topdown, cv::Point2f(x, y), 2, cv::Scalar(0, alpha * 255, (1 - alpha) * 255), -1, CV_AA);
				cv::circle(topdown, cv::Point2f(x, y), 3, cv::Scalar(0, alpha * 128, (1 - alpha) * 128), 1, CV_AA);
				float t1secRadiusPx = (maxPedestrianSpeed*t / maxDepth * imgHeight);
				cv::circle(topdown, cv::Point2f(x, y), t1secRadiusPx, cv::Scalar(0, alpha * 255, (1 - alpha) * 255), 1, CV_AA);
			}
		}
		return topdown;
	}

	cv::Mat getRGBImage(cv::Mat& mRGB, cv::Mat& mDepth, std::vector<DataSetLabel>& labels, float vehicleSpeedKMh, float tireroadFriction, float t) {
		cv::Mat img = mRGB.clone();
		cv::Mat depthImg = mDepth.clone();

		float vehicleSpeedForRating = vehicleSpeedKMh / 3.6f;
		float maxDepth = 80;

		// smooth out the depth for iso lines
		for (int i = 0; i < 10; i++)
			cv::dilate(depthImg, depthImg, cv::Mat());
		for (int i = 0; i < 10; i++)
			cv::erode(depthImg, depthImg, cv::Mat());
		cv::GaussianBlur(depthImg, depthImg, cv::Size(5, 5), 0, 0);

		cv::Mat overlay = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0));
		for (int j = 0; j < img.rows; j++)
		{
			for (int i = 0; i < img.cols; i++)
			{
				float depth = maxDepth - maxDepth * mDepth.at<float>(j, i);

				double angle = CV_PI / 2 - (1.0 * i / img.cols) * CV_PI / 2; // 90 -> 0;
				double actualAngle = angle + CV_PI / 4; // 45° offset of cone - 90° of middle split
				float x = cos(actualAngle) * depth; // adjacent times hypotenuse

													// no pixel -> x in meters
				double remainingTime = getRemainingTimeToHitPedestrian(depth, x, vehicleSpeedKMh);

				for (int r = 0; r < riskTimes.size(); r++)
				{
					if (remainingTime >= riskTimes[r].first && remainingTime < riskTimes[r].second) {
						overlay.at<cv::Vec3b>(j, i) = riskColors[r];
						break;
					}
				}
			}
		}
		img = img + 0.5 * overlay;

		// draw 10m iso depth lines
		for (int d = 0; d < maxDepth; d += 10) {

			Algorithms::marchingSquares([&](int j, int i) -> float { return maxDepth - maxDepth * depthImg.at<float>(j, i);  },
				[&](int x1, int y1, int x2, int y2) -> void {
				cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 1);
			},
				d, mDepth.cols, mDepth.rows, 100, 150);
		}


		for (auto& l : labels) {
			if (!l.isDontCareArea()) {

				double remainingTime = getRemainingTimeToHitPedestrian(l.z_3d, l.x_3d, vehicleSpeedKMh, tireroadFriction, t);
				cv::rectangle(img, l.getBbox(), cv::Scalar(255, 255, 0), 2);
				cv::putText(img, std::to_string(remainingTime), l.getBbox().tl(), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
			}
		}
		return img;
	}

}