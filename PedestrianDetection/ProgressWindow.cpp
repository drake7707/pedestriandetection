#include "ProgressWindow.h"


ProgressWindow::ProgressWindow()
{
}


ProgressWindow::~ProgressWindow()
{
}


ProgressWindow* ProgressWindow::getInstance() {
	static ProgressWindow* instance;
	if (instance == nullptr)
		instance = new ProgressWindow();
	return instance;
}

void ProgressWindow::updateStatus(std::string& name, double percentage, std::string text) {
	lock.lock();

	std::chrono::steady_clock::time_point start;
	if (elements.find(name) != elements.end())
		start = elements[name].startTime;
	else
		start = std::chrono::steady_clock::now();

	elements[name] = ProgressStatus(name, percentage, text);
	elements[name].startTime = start;

	lock.unlock();
}

void ProgressWindow::run() {

	std::thread t([&]() -> void {
		cv::namedWindow("Progress");

		int rowHeight = 60;

		while (true) {
			int rows = elements.size() * rowHeight;
			if (rows > 0) {
				cv::Mat img(rows, 400, CV_8UC3, cv::Scalar(0));

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


					if (el.second.percentage > 0) {
						double duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - el.second.startTime).count();
						// it took duration to get as far as el.second.percentage
						// which means for 1 - el.second.percentage, it will take  duration / el.second.percentage * ( 1 - el.second.percentage)

						long remainingMS = duration / el.second.percentage * (1 - el.second.percentage);
						int ms = remainingMS % 1000;
						remainingMS = (remainingMS - ms) / 1000;
						int secs = remainingMS % 60;
						remainingMS = (remainingMS - secs) / 60;
						int mins = remainingMS % 60;
						int hrs = (remainingMS - mins) / 60;

						std::ostringstream oss;
						oss << std::setfill('0')          // set field fill character to '0'
							<< std::setw(2) << hrs << ":"
							<< std::setw(2) << mins << ":"
							<< std::setw(2) << secs;
						std::string formattedRemainingTime(oss.str());

						int baseLine = 0;
						auto remainingTextSize = cv::getTextSize(formattedRemainingTime, cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.4, 1, &baseLine);
						cv::putText(img, formattedRemainingTime, cv::Point(padding + img.cols - 2 * padding - remainingTextSize.width, y + padding + 10), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(0), 1, CV_AA);
					}

					//border
					cv::rectangle(img, cv::Rect(padding, y + padding, img.cols - 2 * padding, rowHeight - 2 * padding), cv::Scalar(192, 192, 192), 1, 8);

					idx++;
				}
				lock.unlock();
				cv::imshow("Progress", img);
			}
			else
				cv::imshow("Progress", cv::Mat(100, 400, CV_8UC3, cv::Scalar(255, 255, 255)));
			cv::waitKey(100);
		}
	});
	t.detach();
}

void ProgressWindow::finish(std::string& key) {
	lock.lock();
	elements.erase(key);
	lock.unlock();
}