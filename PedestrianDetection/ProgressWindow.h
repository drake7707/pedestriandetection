#pragma once
#include <map>
#include <string>
#include "opencv2/opencv.hpp"
#include <mutex>
#include <thread>
#include <chrono>

struct ProgressStatus {
	std::string name = "";
	double percentage = 0;
	std::string text = "";

	std::chrono::steady_clock::time_point startTime;

	ProgressStatus() {}

	ProgressStatus(std::string& name, double percentage, std::string& text) : name(name), percentage(percentage), text(text) {}
};

class ProgressWindow
{

private:
	std::map<std::string, ProgressStatus> elements;
	std::mutex lock;


public:

	ProgressWindow::ProgressWindow();
	

	ProgressWindow::~ProgressWindow();
	


	static ProgressWindow* getInstance();

	void updateStatus(std::string& name, double percentage, std::string text);

	void run();

	void finish(std::string& key);
};

