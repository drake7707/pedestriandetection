#pragma once
#include "opencv2/opencv.hpp"
#include "Histogram.h"
#include <chrono>

#include <mutex>
#include <condition_variable>



#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 


void showHistogram(Histogram& hist, std::string title = "Histogram");

int randBetween(int min, int max);

int ceilTo(double val, double target);

void slideWindow(int imgWidth, int imgHeight, std::function<void(cv::Rect2d bbox)> func, double minScaleReduction = 0.25, double maxScaleReduction = 4, int refWidth = 64, int refHeight = 128);


template<typename TimeT = std::chrono::milliseconds>
struct measure
{
	template<typename F, typename ...Args>
	static typename TimeT::rep execution(F func, Args&&... args)
	{
		auto start = std::chrono::system_clock::now();

		// Now call the function with all the parameters you need.
		func(std::forward<Args>(args)...);

		auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);

		return duration.count();
	}
};



class Semaphore {
public:
	Semaphore(int count_ = 0)
		: count(count_) {}

	inline void notify()
	{
		std::unique_lock<std::mutex> lock(mtx);
		count++;
		cv.notify_one();
	}

	inline void wait()
	{
		std::unique_lock<std::mutex> lock(mtx);

		while (count == 0) {
			cv.wait(lock);
		}
		count--;
	}

private:
	std::mutex mtx;
	std::condition_variable cv;
	int count;
};