#include "VariableNumberFeatureCreator.h"
#include "Helper.h"


VariableNumberFeatureCreator::VariableNumberFeatureCreator(std::string& creatorName, int clusterSize) : creatorName(creatorName), clusterSize(clusterSize)
{
}


VariableNumberFeatureCreator::~VariableNumberFeatureCreator()
{
}

void VariableNumberFeatureCreator::prepare(std::string& datasetPath) {

	// try and load first
	loadCentroids(std::string(""));
	if (centroids.size() > 0)
		return;

	int k = clusterSize;
	cv::TermCriteria criteria(CV_TERMCRIT_ITER, 1000, 1.0);
	int attempts = 10;
	int flags = cv::KmeansFlags::KMEANS_PP_CENTERS; // use kmeans++

	std::vector<FeatureVector> samples;

	iterateDataSet(datasetPath, [](int idx) -> bool { return true; },
		[&](int idx, int resultClass, cv::Mat&rgb, cv::Mat&depth) -> void {

		// aggregate vectors

		std::vector<FeatureVector> localSamples = getVariableNumberFeatures(rgb, depth);
		for (auto& s : localSamples)
			samples.push_back(s);
	});

	if (samples.size() > 0) {
		int featureSize = samples[0].size();
		cv::Mat kmeansData(samples.size(), featureSize, CV_32FC1);

		for (int s = 0; s < samples.size(); s++)
		{
			for (int f = 0; f < featureSize; f++)
				kmeansData.at<float>(s, f) = samples[s][f];
		}

		cv::Mat labels;
		cv::Mat centers;
		double result = cv::kmeans(kmeansData, k, labels, criteria, attempts, flags, centers);

		for (int j = 0; j < centers.rows; j++)
		{
			FeatureVector centroid(centers.cols, 0);
			for (int i = 0; i < centers.cols; i++)
			{
				centroid[i] = centers.at<float>(j, i);
			}
			this->centroids.push_back(centroid);
		}




	//	// visualize to check if it's correct:
	//	cv::Mat test(128, 64, CV_8UC3, cv::Scalar(0));

	//	std::vector<cv::Scalar> colors;
	//	for (int i = 0; i < centroids.size(); i++)
	//	{
	//		colors.push_back(cv::Scalar(1.0 * rand() / RAND_MAX * 255, 1.0 * rand() / RAND_MAX * 255, 1.0 * rand() / RAND_MAX * 255));
	//	}

	//	for (int j = 0; j < samples.size(); j++)
	//	{
	//		float x = samples[j][0];
	//		float y = samples[j][1];

	//		double minDistance = std::numeric_limits<double>().max();
	//		int closestCentroidIndex = -1;
	//		for (int c = 0; c < centroids.size(); c++)
	//		{
	//			double distance = samples[j].distanceToSquared(centroids[c]);
	//			if (minDistance > distance) {
	//				closestCentroidIndex = c;
	//				minDistance = distance;
	//			}
	//		}
	//		cv::circle(test, cv::Point(x, y), 2, colors[closestCentroidIndex], -1);
	//	}

	//	for (int c = 0; c < centroids.size(); c++)
	//	{
	//		cv::circle(test, cv::Point(centroids[c][0], centroids[c][1]), 4, cv::Scalar(255, 0, 0), -1);
	//	}
	//	cv::imshow("test", test);
	//	cv::waitKey(0);
	
	
	}



	saveCentroids(std::string(""));
}

FeatureVector VariableNumberFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	std::vector<FeatureVector> localSamples = getVariableNumberFeatures(rgb, depth);


	// tally closest centroids
	FeatureVector histogram(centroids.size(), 0);
	for (auto& s : localSamples) {

		// find the centroid that's closest
		double minDistance = std::numeric_limits<double>().max();
		int closestCentroidIndex = -1;
		for (int c = 0; c < centroids.size(); c++)
		{
			double distance = s.distanceToSquared(centroids[c]);
			if (minDistance > distance) {
				closestCentroidIndex = c;
				minDistance = distance;
			}
		}

		histogram[closestCentroidIndex]++;
	}

	// because the amount of samples can vary the histogram must be normalized so the values can be compared
	if (localSamples.size() > 0) {
		for (int i = 0; i < histogram.size(); i++)
			histogram[i] /= localSamples.size();
	}
	return histogram;
}

int VariableNumberFeatureCreator::getNumberOfFeatures() const {
	return centroids.size();
}


std::string VariableNumberFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return creatorName + " centroid " + std::to_string(featureIndex) + " count ";
}


void VariableNumberFeatureCreator::saveCentroids(std::string& path) {

	cv::FileStorage fs(path + creatorName + ".xml", cv::FileStorage::WRITE);
	fs << "centroids" << centroids;
	fs.release();
}

void VariableNumberFeatureCreator::loadCentroids(std::string& path) {

	cv::FileStorage fsRead(path + creatorName + ".xml", cv::FileStorage::READ);
	fsRead["centroids"] >> centroids;
	fsRead.release();
}


