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
#ifdef _WIN32
#include <io.h> 
#define access    _access_s
#else
#include <unistd.h>
#endif

void showHistogram(Histogram& hist, std::string title = "Histogram");

int randBetween(int min, int max);

int ceilTo(double val, double target);

void slideWindow(int imgWidth, int imgHeight, std::function<void(cv::Rect bbox)> func, double minScaleReduction = 0.25, double maxScaleReduction = 4, int slidingWindowStep = 8, float multiplicationFactor = 2, int refWidth = 64, int refHeight = 128);


struct SlidingWindowRegion {
	int imageNumber;
	cv::Rect bbox;
	SlidingWindowRegion(int imageNumber, cv::Rect bbox) : imageNumber(imageNumber), bbox(bbox) { }
};

std::vector < std::pair<float, SlidingWindowRegion>> applyNonMaximumSuppression(std::vector<std::pair<float, SlidingWindowRegion>>& windows, float iouTreshold = 0.5);



void iterateDataSet(const std::string& baseDatasetPath, std::function<bool(int idx)> canSelectFunc, std::function<void(int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth)> func);

void parallel_for(int from, int to, int nrOfThreads, std::function<void(int)> func);

double getIntersectionOverUnion(cv::Rect& r1, cv::Rect& r2);


bool FileExists(const std::string &Filename);


template <typename T, typename T2>
void parallel_foreach(const std::map<T, T2>& map, int nrOfThreads, std::function<void(std::pair<T, T2>&)> func);


template <typename T, typename T2>
void parallel_foreach(const std::map<T, T2>& map, int nrOfThreads, std::function<void(std::pair<T, T2>&)> func) {

	if (nrOfThreads == 1) {
		// just do it on the main thread
		auto it = map.begin();
		while (it != map.end()) {
			std::pair<T, T2> pair = *it;
			func(pair);
			it++;
		}
	}
	else {
		std::mutex lock;
		std::vector<std::thread> threads;
		threads.reserve(nrOfThreads);

		auto it = map.begin();
		for (int t = 0; t < nrOfThreads; t++)
		{
			threads.push_back(std::thread([&]() -> void {
				bool stop = false;
				while (!stop) {
					std::pair<T, T2> pair;
					lock.lock();
					if (it != map.end()) {
						pair = *it;
						it++;

						lock.unlock();
						func(pair);
					}
					else {
						lock.unlock();
						stop = true;
					}
				}
			}));
		}

		for (auto& t : threads)
			t.join();
	}
}




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