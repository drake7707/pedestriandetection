#pragma once
#include <map>
#include <string>
#include "opencv2/opencv.hpp"
#include <mutex>
#include <thread>

struct ProgressStatus {
	std::string name = "";
	double percentage = 0;
	std::string text = "";
	ProgressStatus() {}

	ProgressStatus(std::string& name, double percentage, std::string& text) : name(name), percentage(percentage), text(text) {}
};

class ProgressWindow
{

private:
	std::map<std::string, ProgressStatus> elements;
	std::mutex lock;

	
public:

	ProgressWindow::ProgressWindow()
	{
	}


	ProgressWindow::~ProgressWindow()
	{
	}



	static ProgressWindow* getInstance() {
		static ProgressWindow* instance;
		if (instance == nullptr)
			instance = new ProgressWindow();
		return instance;
	}

	void updateStatus(std::string& name, double percentage, std::string text) {
		lock.lock();
		elements[name] = ProgressStatus(name, percentage, text);
		lock.unlock();
	}

	void run() {

		std::thread t([&]() -> void {
			cv::namedWindow("Progress");

			int rowHeight = 60;
			
			while (true) {
				int rows = elements.size() * rowHeight;
				if (rows > 0) {
					cv::Mat img(rows, 300, CV_8UC3, cv::Scalar(0));

					int padding = 5;

					int idx = 0;
					lock.lock();
					for (auto& el : elements) {

						int y = idx * rowHeight;

						//background
						cv::rectangle(img, cv::Rect(padding, y + padding, img.cols - 2 * padding, rowHeight - 2 * padding), cv::Scalar(255, 255, 255), -1);
						// percentage filled
						cv::rectangle(img, cv::Rect(padding, y + padding, el.second.percentage * (img.cols - 2 * padding), rowHeight - 2 * padding), cv::Scalar(255, 192, 64), -1);

						// text
						std::string text = el.second.name;
						cv::putText(img, text, cv::Point(padding, y + padding + 20), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0), 1, CV_AA);

						std::string text2 = el.second.text;
						cv::putText(img, text2, cv::Point(padding, y + padding + 40), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(0), 1, CV_AA);


						//border
						cv::rectangle(img, cv::Rect(padding, y + padding, img.cols - 2 * padding, rowHeight - 2 * padding), cv::Scalar(192, 192, 192), 1, 8);

						idx++;
					}
					lock.unlock();
					cv::imshow("Progress", img);
				}
				else
					cv::imshow("Progress", cv::Mat(100,300, CV_8UC3, cv::Scalar(255)));
				cv::waitKey(100);
			}
		});
		t.detach();
	}

	void finish(std::string& key) {
		lock.lock();
		elements.erase(key);
		lock.unlock();
	}
};

