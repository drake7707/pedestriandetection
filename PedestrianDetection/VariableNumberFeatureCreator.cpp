#include "VariableNumberFeatureCreator.h"
#include "Helper.h"


VariableNumberFeatureCreator::VariableNumberFeatureCreator(std::string& creatorName, int clusterSize) : creatorName(creatorName), clusterSize(clusterSize)
{
}


VariableNumberFeatureCreator::~VariableNumberFeatureCreator()
{
}

void VariableNumberFeatureCreator::prepare(std::string& datasetPath) {

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
	}
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

			histogram[c]++;
		}
	}

	// because the amount of samples can vary the histogram must be normalized so the values can be compared
	for (int i = 0; i < histogram.size(); i++)
		histogram[i] /= localSamples.size();

	return histogram;
}

int VariableNumberFeatureCreator::getNumberOfFeatures() const {
	return clusterSize;
}


std::string VariableNumberFeatureCreator::explainFeature(int featureIndex, double featureValue) const {
	return creatorName + " centroid " + std::to_string(featureIndex) + " count ";
}