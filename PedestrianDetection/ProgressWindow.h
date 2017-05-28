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
	


	/// <summary>
	/// Returns the singleton instance of the window
	/// </summary>
	static ProgressWindow* getInstance();


	/// <summary>
	/// Updates the percentage and displayed text of the progress bar with given name (creates a progress bar if it didn't exist yet) 
	/// </summary>
	void updateStatus(std::string& name, double percentage, std::string text);


	/// <summary>
	/// Shows and periodically updats the progress bar window
	/// </summary>
	void run();


	/// <summary>
	/// Finishes the progress bar associated with given name
	/// </summary>
	void finish(std::string& name);
};

